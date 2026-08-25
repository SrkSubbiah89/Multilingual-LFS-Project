"""
eval/build_manifests_from_existing_runs.py

2026-08-25. One-off (but real, reusable) script: builds proper
ExperimentRunManifest files for the real, already-committed evaluation
runs from the ISCO-08 catalogue-enrichment / e5-large investigation,
none of which went through eval/manifest.py at the time (they were run
directly via eval/run_eval.py, which produces a CaseResult CSV but not a
wrapped manifest_*.jsonl -- that wrapping is normally eval/
ablation_runner.py's job, and none of these runs used a named
ablation_runner.py config since they were testing new profiles that
config system doesn't know about).

This closes a real, disclosed gap: eval/figure_exports/export_evaluation_
results.py and export_latency_scalability.py only ever look for
manifest_*.jsonl files, so they were reporting "no_manifests_found: true"
even after real, decisive, full-scale results existed -- the data existed,
it just wasn't wrapped in the format those exporters read.

Builds manifests from case_rows read back from the real CSVs (build_manifest()'s
own documented supported input -- "or rows read back from a previously-
written CaseResult CSV"), not by re-running anything. dataset_label is
synthetic_or_operationally_realistic for every one of these (WISCO is
real, externally-sourced reference data, but not real LFS respondent
data) -- never approved_real_lfs_validation, consistent with every other
WISCO-based result in this project.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from eval.manifest import build_manifest, write_manifest
from eval.dataset_card_schema import SYNTHETIC_OR_OPERATIONALLY_REALISTIC

_HELDOUT_TEST_SET = Path("eval/local_benchmarks/wisco_isco08_v3_dev_validation_split/heldout_run_eval_format.csv")
_VALIDATION_TEST_SET = Path("eval/local_benchmarks/wisco_isco08_v3_dev_validation_split/validation_run_eval_format.csv")

_RUNS = [
    dict(
        csv_path=Path("eval/results/raw_runs/enriched_e5large_heldout_20260824/20260824T123941Z_flat.csv"),
        run_id="enriched_e5large_heldout_20260824",
        classifier_method="flat_isco08_official_ilo2021_v1_enriched_e5large",
        split_name="heldout",
        test_set=_HELDOUT_TEST_SET,
        retrieval_params={"system": "flat", "use_llm_reranker": False,
                           "isco_catalogue_profile": "official_ilo2021_v1_enriched_e5large"},
        llm_params={"reranker_model": ""},
    ),
    dict(
        csv_path=Path("eval/results/raw_runs/enriched_catalogue_heldout_20260824/20260824T113739Z_flat.csv"),
        run_id="enriched_catalogue_heldout_20260824",
        classifier_method="flat_isco08_official_ilo2021_v1_enriched",
        split_name="heldout",
        test_set=_HELDOUT_TEST_SET,
        retrieval_params={"system": "flat", "use_llm_reranker": False,
                           "isco_catalogue_profile": "official_ilo2021_v1_enriched"},
        llm_params={"reranker_model": ""},
    ),
    dict(
        csv_path=Path("eval/results/dev_selection/enriched_catalogue_with_rerank_check/20260824T120417Z_flat.csv"),
        run_id="enriched_catalogue_with_rerank_check_20260824",
        classifier_method="flat_isco08_official_ilo2021_v1_enriched",
        split_name="validation",
        test_set=_VALIDATION_TEST_SET,
        retrieval_params={"system": "flat", "use_llm_reranker": True,
                           "isco_catalogue_profile": "official_ilo2021_v1_enriched"},
        llm_params={"reranker_model": "groq/openai/gpt-oss-120b"},
    ),
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for run in _RUNS:
        csv_path: Path = run["csv_path"]
        if not csv_path.exists():
            print(f"SKIP (not found): {csv_path}")
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            case_rows = list(csv.DictReader(f))

        manifest = build_manifest(
            case_rows,
            run_id=run["run_id"],
            classifier_method=run["classifier_method"],
            split_name=run["split_name"],
            dataset_version_hash=_sha256_file(run["test_set"]),
            dataset_label=SYNTHETIC_OR_OPERATIONALLY_REALISTIC,
            retrieval_params=run["retrieval_params"],
            llm_params=run["llm_params"],
            evaluation_status="measured",
        )
        out_dir = csv_path.parent / "manifests"
        manifest_csv_path, jsonl_path = write_manifest(manifest, out_dir)
        print(f"Wrote manifest for {run['run_id']}: {jsonl_path}")
        print(f"  n_cases={manifest.n_cases} peak_memory_mb={manifest.peak_process_memory_mb} "
              f"latency_mean_ms={manifest.latency_mean_ms}")


if __name__ == "__main__":
    main()
