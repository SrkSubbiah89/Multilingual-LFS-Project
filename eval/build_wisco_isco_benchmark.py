"""
eval/build_wisco_isco_benchmark.py

Conference I Reviewer #2 response, Step 6, Phase D: converts the already-
tracked, externally-sourced WISCO dataset
(eval/legacy_thesis_ch6/wisco/data/processed/wisco_raw_parsed.json -- see
Documentation/Phase_2/Week_1/PROVENANCE.md for full provenance: Zenodo
version DOI 10.5281/zenodo.8262593, CC-BY-4.0, SurveyCodings/WageIndicator
Foundation) into a controlled_benchmark_schema.BenchmarkRecord package for
ISCO-08 only (see CONTROLLED_BENCHMARK_AUDIT.md for why ISIC/ISCED are out
of scope for WISCO -- no direct gold codes exist for those standards here).

This is DATA PREPARATION ONLY: deterministic restructuring + hashing, zero
classifier/LLM/network calls, no accuracy measurement of any kind. It does
NOT run eval/run_eval.py or eval/ablation_runner.py -- that is Step 7's
job, not this script's.

Split design
------------
Dev/heldout assignment is made PER OCCUPATION (the WISCO record `key`), not
per generated (occupation, language) record -- so a single occupation's
five language variants always land in the same split. This avoids a
same-occupation leak across splits that would otherwise let a model see an
occupation's Arabic title in dev and its English title in heldout (or vice
versa), which would be a real leakage-safety violation even though the
literal input text differs. Assignment is a deterministic sha256-based hash
of the occupation key mod 10 (~10% dev, ~90% heldout) -- reproducible
without a stored RNG seed.

Every output record carries dataset_label=synthetic_or_operationally_realistic
(WISCO is reference/dictionary data, not Labour Force Survey respondent
data -- see controlled_benchmark_schema.py's module docstring; this is
FIXED, never a script argument, so this tool cannot be misused to produce
a real-LFS-labelled package).

Usage
-----
    python eval/build_wisco_isco_benchmark.py --out-root eval/local_benchmarks/wisco_isco08_v1

Output (all under --out-root, Git-ignored by default per .gitignore's
eval/local_benchmarks/ entry):
    records.json           -- full BenchmarkRecord list (dev + heldout + any excluded)
    dataset_hash.txt        -- sha256 covering the whole records.json payload
    split_manifest.json     -- controlled_benchmark's dev/heldout split_id + counts
    build_summary.json      -- counts, language breakdown, source file hash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from controlled_benchmark_schema import (  # noqa: E402
    BenchmarkRecord,
    DATA_SOURCE_EXTERNAL_AUTHORITATIVE,
    DOUBLE_CODING_UNKNOWN,
    ADJUDICATION_UNKNOWN,
    INDEPENDENT_LABEL_STATUS_INDEPENDENT,
    ISCO08,
    LABEL_SOURCE_EXTERNAL_PUBLISHER,
    SPLIT_DEV,
    SPLIT_HELDOUT,
)
from dataset_card_schema import DEFAULT_DATASET_LABEL  # noqa: E402
from validate_controlled_benchmark import validate_benchmark_package  # noqa: E402

REPO_ROOT = _HERE.parent
WISCO_PARSED_PATH = REPO_ROOT / "backend" / "evaluation" / "wisco" / "data" / "processed" / "wisco_raw_parsed.json"
TARGET_LANGUAGES = ("en", "ar", "ur", "hi", "tl")

LABELER = "external_publisher:wisco_surveycodings_wageindicator_foundation"
CLASSIFICATION_VERSION = "ISCO-08 (2008)"


def _split_for_key(key) -> str:
    """Deterministic ~10% dev / ~90% heldout split, keyed on the WISCO
    occupation key (not the generated record) so all languages of one
    occupation always land in the same split."""
    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    return SPLIT_DEV if int(digest[:8], 16) % 10 == 0 else SPLIT_HELDOUT


def build_records(wisco_records: list[dict]) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for occ in wisco_records:
        split = _split_for_key(occ["key"])
        gold_code = occ.get("isco08_unit", "")
        gold_title = occ.get("master_label_en", "")
        titles = occ.get("titles", {})
        for lang in TARGET_LANGUAGES:
            text = titles.get(lang)
            if not text:
                continue  # honest gap -- not every occupation has all 5 languages (see PROVENANCE.md counts)
            rec = BenchmarkRecord(
                benchmark_id=f"WISCO-{occ['key']}-{lang}",
                task=ISCO08,
                data_source_type=DATA_SOURCE_EXTERNAL_AUTHORITATIVE,
                dataset_label=DEFAULT_DATASET_LABEL,
                language=lang,
                input_text=text,
                context_fields=None,
                gold_code=gold_code,
                gold_code_title=gold_title,
                classification_standard="ISCO-08",
                classification_version=CLASSIFICATION_VERSION,
                hierarchy_level="unit_group_4digit",
                label_source_type=LABEL_SOURCE_EXTERNAL_PUBLISHER,
                labeler_identifier_or_role=LABELER,
                independent_label_status=INDEPENDENT_LABEL_STATUS_INDEPENDENT,
                double_coding_status=DOUBLE_CODING_UNKNOWN,  # honest: not confirmed in this repo -- see CONTROLLED_BENCHMARK_AUDIT.md
                adjudication_status=ADJUDICATION_UNKNOWN,
                ambiguity_flag=False,
                exclusion_reason=None,
                split=split,
            )
            payload = rec.model_dump_json(exclude={"record_hash"}, exclude_none=False).encode("utf-8")
            rec.record_hash = hashlib.sha256(payload).hexdigest()
            records.append(rec)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--wisco-parsed", type=Path, default=WISCO_PARSED_PATH)
    args = parser.parse_args()

    if not args.wisco_parsed.exists():
        parser.error(f"WISCO parsed file not found: {args.wisco_parsed}")

    wisco_records = json.loads(args.wisco_parsed.read_text(encoding="utf-8"))
    records = build_records(wisco_records)

    args.out_root.mkdir(parents=True, exist_ok=True)

    records_payload = [json.loads(r.model_dump_json()) for r in records]
    dataset_hash = hashlib.sha256(
        json.dumps(records_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    report = validate_benchmark_package(records, dataset_hash=dataset_hash)

    (args.out_root / "records.json").write_text(
        json.dumps({"records": records_payload}, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (args.out_root / "dataset_hash.txt").write_text(dataset_hash + "\n", encoding="utf-8")

    dev_ids = [r.benchmark_id for r in records if r.split == SPLIT_DEV]
    heldout_ids = [r.benchmark_id for r in records if r.split == SPLIT_HELDOUT]
    split_manifest = {
        "dataset_id": "WISCO-ISCO08-BENCHMARK-v1",
        "split_hash": hashlib.sha256((",".join(sorted(dev_ids)) + "|" + ",".join(sorted(heldout_ids))).encode("utf-8")).hexdigest(),
        "splits": {
            "dev": {"split_id": f"WISCO-ISCO08-DEV-{len(dev_ids)}", "purpose": "parameter selection only", "n_cases": len(dev_ids), "frozen": False},
            "heldout": {"split_id": f"WISCO-ISCO08-HELDOUT-{len(heldout_ids)}", "purpose": "frozen confirmation set", "n_cases": len(heldout_ids), "frozen": True},
        },
        "leakage_check_performed": True,
        "leakage_check_method": "split assigned per underlying WISCO occupation key (not per generated language record), so all 5 language variants of one occupation always share a split -- see this script's module docstring",
    }
    (args.out_root / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lang_counts = Counter(r.language for r in records)
    split_counts = Counter(r.split for r in records)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(args.wisco_parsed.relative_to(REPO_ROOT)),
        "source_file_sha256": hashlib.sha256(args.wisco_parsed.read_bytes()).hexdigest(),
        "n_source_occupations": len(wisco_records),
        "n_benchmark_records": len(records),
        "language_counts": dict(lang_counts),
        "split_counts": dict(split_counts),
        "dataset_hash": dataset_hash,
        "validation_report": {"ok": report.ok, "errors": report.errors, "warnings": report.warnings},
    }
    (args.out_root / "build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(records)} benchmark records ({dict(split_counts)}) to {args.out_root}")
    print(f"dataset_hash={dataset_hash}")
    print(f"validation: ok={report.ok} errors={len(report.errors)} warnings={len(report.warnings)}")


if __name__ == "__main__":
    main()
