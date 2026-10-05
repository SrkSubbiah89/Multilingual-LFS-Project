"""Write additive corrections for review findings 4 and 9, without rerunning AI.

Raw CSVs and original manifests remain byte-identical. A method label alone
cannot prove which encoder executed, so historical encoder identity is marked
unresolved rather than silently replaced with the intended profile's encoder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from analyze import isco_accuracy, wilson_score_interval


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correct_artifacts(results_root: Path, output: Path) -> dict:
    index = {"correction_date": "2026-10-04", "raw_artifacts_modified": False,
             "top3_corrections": [], "provenance_corrections": []}
    for source in sorted(results_root.rglob("*.csv")):
        with source.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "gold_rank_in_pool" not in reader.fieldnames:
                continue
            rows = list(reader)
        rel = source.relative_to(results_root)
        original_hash = _sha(source)
        ranked = [r for r in rows if (r.get("gold_rank_in_pool") or "").strip()]
        corrected = isco_accuracy(rows)[-1]
        if ranked and corrected.n > len(ranked):
            hits = sum(int(r["gold_rank_in_pool"]) <= 3 for r in ranked)
            old_lo, old_hi = wilson_score_interval(hits, len(ranked))
            target = output / rel.with_suffix(".top3_corrected.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "finding": 4, "source_csv": source.as_posix(), "source_sha256": original_hash,
                "metric": "isco_top3_prererank_pool",
                "supersedes": {"n": len(ranked), "successes": hits,
                               "accuracy": round(hits / len(ranked), 4),
                               "ci_lo": round(old_lo, 4), "ci_hi": round(old_hi, 4)},
                "corrected": asdict(corrected),
                "note": "Recorded candidate-pool misses count as failures. Top-1 predictions and scores are unchanged.",
            }
            target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            index["top3_corrections"].append(target.relative_to(output).as_posix())

        labelled_large = [r for r in rows if "e5large" in (r.get("pred_method") or "")]
        bad_encoders = sorted({r.get("embedding_model_version", "") for r in labelled_large
                               if r.get("embedding_model_version") != "intfloat/multilingual-e5-large"})
        if bad_encoders:
            target = output / rel.with_suffix(".provenance_correction.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            manifests = [{"path": p.as_posix(), "sha256": _sha(p)}
                         for p in sorted((source.parent / "manifests").glob("manifest_*"))
                         if p.suffix in (".csv", ".jsonl", ".json")]
            payload = {
                "finding": 9, "source_csv": source.as_posix(), "source_sha256": original_hash,
                "recorded_embedding_model_versions": bad_encoders,
                "claimed_method_labels": sorted({r["pred_method"] for r in labelled_large}),
                "profile_expected_embedding_model": "intfloat/multilingual-e5-large",
                "executed_embedding_model": None,
                "provenance_status": "historical_execution_identity_unresolved",
                "config_hash_status": "does_not_reliably_identify_encoder_before_harness_fix",
                "affected_downstream_manifests": manifests,
                "downstream_manifest_correction": {
                    "model_versions.embedding": None,
                    "reason": "Original manifests do not resolve the executed encoder; retain this explicit uncertainty when consuming them.",
                },
                "accuracy_changed": False,
                "note": "The harness recorded a constant e5-small encoder. The intended profile label is insufficient to prove executed identity; independent startup/model or collection evidence is required.",
            }
            target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            index["provenance_corrections"].append(target.relative_to(output).as_posix())
        if _sha(source) != original_hash:
            raise RuntimeError(f"Raw source changed during correction: {source}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "corrections_20261004")
    args = parser.parse_args()
    index = correct_artifacts(args.results_root, args.out)
    print(f"Wrote {len(index['top3_corrections'])} top-3 and {len(index['provenance_corrections'])} provenance corrections to {args.out}")


if __name__ == "__main__":
    main()
