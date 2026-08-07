"""
eval/export_benchmark_to_run_eval_csv.py

Conference I Reviewer #2 response, Step 6, Phase G: reformats a
controlled_benchmark_schema.BenchmarkRecord package (eval/build_wisco_isco_
benchmark.py's output, or any other package matching that schema) into the
plain CSV schema eval/run_eval.py's --test-set already expects
(case_id,input_text,input_language,gold_isco_4digit,gold_isic,gold_isced).

This is a DETERMINISTIC REFORMATTING step only -- no classifier/LLM/network
call, no accuracy measurement. It exists solely so the Step 7 command
prepared in this Step 6 pass is actually executable; it does not itself run
that command (see Documentation/Conference_I_Reviewer_2/STEP_7_COMMAND.md).

Only records from ONE split (--split dev|heldout) are exported per call, so
a caller can never accidentally mix dev and heldout rows into one CSV.
Records with task != isco08 are skipped (gold_isic/gold_isced stay blank
for every row -- WISCO has no direct ISIC/ISCED gold labels; see
CONTROLLED_BENCHMARK_AUDIT.md).

Usage
-----
    python eval/export_benchmark_to_run_eval_csv.py \\
        --records eval/local_benchmarks/wisco_isco08_v1/records.json \\
        --split heldout \\
        --out eval/local_benchmarks/wisco_isco08_v1/heldout_run_eval_format.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from controlled_benchmark_schema import ISCO08, SPLITS  # noqa: E402

FIELDNAMES = ["case_id", "input_text", "input_language", "gold_isco_4digit", "gold_isic", "gold_isced"]


def export(records: list[dict], split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split!r}")
    rows = []
    for r in records:
        if r["split"] != split or r["task"] != ISCO08:
            continue
        rows.append({
            "case_id": r["benchmark_id"],
            "input_text": r["input_text"],
            "input_language": r["language"],
            "gold_isco_4digit": r["gold_code"],
            "gold_isic": "",  # honest gap -- no direct ISIC gold in this source, see CONTROLLED_BENCHMARK_AUDIT.md
            "gold_isced": "",  # same, for ISCED
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=sorted(SPLITS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.records.read_text(encoding="utf-8"))
    rows = export(data["records"], args.split)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
