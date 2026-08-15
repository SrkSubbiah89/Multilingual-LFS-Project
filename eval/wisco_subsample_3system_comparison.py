"""
eval/wisco_subsample_3system_comparison.py

Runs backend/evaluation/evaluate.py's existing BM25 / flat / hierarchical
3-system comparison against a real WISCO subsample -- the specific gap
identified in Phase 0 of the WISCO validation task (BM25 has never been
run against WISCO at all; the existing flat-vs-hierarchical WISCO result
is a separate, already-published comparison using the official ILO 2021
catalogue profile).

Uses the WISCO benchmark's own pre-existing, deterministic 500-record
subset (eval/local_benchmarks/wisco_isco08_v2_group_split/
reranking_subset_500.json -- seed 42, stratified by language x ISCO
major group, drawn from the same frozen heldout split as the canonical
Tier-1 result) rather than constructing a new sample.

IMPORTANT METHODOLOGICAL DISCLOSURE: backend/evaluation/evaluate.py
constructs ISCOClassifier() with no isco_catalogue_profile argument,
i.e. the LEGACY catalogue profile (436 unit groups as of the 2026-08-12
primary-source fix) -- NOT the official_ilo2021_v1 profile used for the
canonical published Tier-1 result (21.1927% flat / 10.3537%
hierarchical). This script does not modify evaluate.py or ISCOClassifier
(per this task's explicit "don't tune/adjust the classifier" instruction),
so its flat/hierarchical numbers are NOT directly comparable to the
canonical Tier-1 numbers -- different catalogues. This run's real
contribution is the BM25 baseline data point, which does not exist
anywhere else against WISCO.

Usage
-----
    python eval/wisco_subsample_3system_comparison.py --out backend/evaluation/results_wisco.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.evaluation.evaluate import run_comparison, print_comparison_table, save_results_csv  # noqa: E402

_SUBSET_PATH = Path(__file__).resolve().parent / "local_benchmarks" / "wisco_isco08_v2_group_split" / "reranking_subset_500.json"
_RECORDS_PATH = Path(__file__).resolve().parent / "local_benchmarks" / "wisco_isco08_v2_group_split" / "records.json"


def load_wisco_subsample() -> list[tuple[str, str]]:
    subset = json.loads(_SUBSET_PATH.read_text(encoding="utf-8"))
    ids = set(subset["benchmark_ids"])
    all_records = json.loads(_RECORDS_PATH.read_text(encoding="utf-8"))["records"]
    by_id = {r["benchmark_id"]: r for r in all_records}

    missing = ids - set(by_id.keys())
    if missing:
        raise RuntimeError(f"{len(missing)} benchmark_ids from the subset are missing from records.json: {sorted(missing)[:5]}...")

    cases = [(by_id[bid]["input_text"], by_id[bid]["gold_code"]) for bid in subset["benchmark_ids"]]
    assert len(cases) == subset["actual_size"] == 500, f"expected 500 cases, got {len(cases)}"
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="backend/evaluation/results_wisco.csv")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    test_cases = load_wisco_subsample()
    print(f"Loaded {len(test_cases)} WISCO subsample cases (seed 42, stratified, heldout split)")

    results = run_comparison(top_k=args.top_k, systems=None, test_cases=test_cases)
    print_comparison_table(results)
    save_results_csv(results, args.out)


if __name__ == "__main__":
    main()
