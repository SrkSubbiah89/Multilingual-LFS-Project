"""
eval/figure_exports/export_evaluation_results.py

Exports a clean CSV/JSON summary of every experiment-run manifest found
under eval/results/ (Conference I Reviewer #2 response, Section I) for an
evaluation-results figure/table -- NOT a screenshot.

If no manifests exist yet (the default state until eval/ablation_runner.py
or eval/manifest.py has actually been run), this writes a structured
"no data yet" JSON/CSV rather than fabricating rows -- see the
no_manifests_found flag in the JSON output.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

_FIELDS = [
    "run_id", "classifier_method", "split_name", "dataset_label",
    "evaluation_status", "n_cases",
    "latency_mean_ms", "latency_p95_ms", "hitl_escalation_rate",
    "estimated_cost_usd", "git_commit", "utc_timestamp",
]


def find_manifests(results_dir: Path = _RESULTS_DIR) -> list[Path]:
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob("manifest_*.jsonl"))


def load_manifest_summaries(results_dir: Path = _RESULTS_DIR) -> list[dict]:
    summaries = []
    for path in find_manifests(results_dir):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            continue
        data = json.loads(lines[0])
        summaries.append({k: data.get(k) for k in _FIELDS})
    return summaries


def write_json(summaries: list[dict], path: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_manifests_found": len(summaries) == 0,
        "results": summaries,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(summaries: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--results-dir", type=Path, default=_RESULTS_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = load_manifest_summaries(args.results_dir)
    write_json(summaries, args.out / "evaluation_results.json")
    write_csv(summaries, args.out / "evaluation_results.csv")
    if summaries:
        print(f"Wrote {len(summaries)} result(s) to {args.out}")
    else:
        print(f"No manifests found under {args.results_dir} -- wrote a 'no data yet' export to {args.out}")


if __name__ == "__main__":
    main()
