"""
eval/figure_exports/export_latency_scalability.py

Exports latency/throughput/hardware figures for a latency-vs-scalability
chart (Conference I Reviewer #2 response, Section I, reviewer comment 4:
computational analysis). Reads the same experiment-run manifests as
export_evaluation_results.py but projects a different column set (latency
percentiles, throughput, hardware) -- kept as a separate script per the
Reviewer #2 response plan rather than merged, since the two serve
different figures (accuracy/result table vs. performance/scalability
chart) with different natural axes.

If no manifests exist yet, writes a structured "no data yet" JSON/CSV
rather than fabricating rows.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

_FIELDS = [
    "run_id", "classifier_method", "dataset_label", "evaluation_status", "n_cases",
    "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
    "throughput_cases_per_sec",
    "hardware_cpu_count", "hardware_ram_gb", "hardware_gpu_model",
    "peak_process_memory_mb", "peak_gpu_memory_mb",
]


def find_manifests(results_dir: Path = _RESULTS_DIR) -> list[Path]:
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob("manifest_*.jsonl"))


def load_latency_rows(results_dir: Path = _RESULTS_DIR) -> list[dict]:
    rows = []
    for path in find_manifests(results_dir):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            continue
        data = json.loads(lines[0])
        hw = data.get("hardware") or {}
        row = {k: data.get(k) for k in _FIELDS if not k.startswith("hardware_")}
        row["hardware_cpu_count"] = hw.get("cpu_count")
        row["hardware_ram_gb"] = hw.get("ram_gb")
        row["hardware_gpu_model"] = hw.get("gpu_model")
        rows.append(row)
    return rows


def write_json(rows: list[dict], path: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_manifests_found": len(rows) == 0,
        "results": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--results-dir", type=Path, default=_RESULTS_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_latency_rows(args.results_dir)
    write_json(rows, args.out / "latency_scalability.json")
    write_csv(rows, args.out / "latency_scalability.csv")
    if rows:
        print(f"Wrote {len(rows)} row(s) to {args.out}")
    else:
        print(f"No manifests found under {args.results_dir} -- wrote a 'no data yet' export to {args.out}")


if __name__ == "__main__":
    main()
