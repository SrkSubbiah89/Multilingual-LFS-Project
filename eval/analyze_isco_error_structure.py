"""Error-structure analysis for ISCO-08 retrieval on the WISCO heldout split.

Exact four-digit accuracy says how often a configuration is completely right.
It says nothing about the shape of the remaining error, which is the part a
reader needs in order to judge whether the system understands occupations and
misses the precise unit group, or fails to recognise the occupation at all.

ISCO-08 is hierarchical: 1 digit is the major group (Professionals, Service and
Sales Workers, ...), 2 digits the sub-major, 3 digits the minor, 4 digits the
unit group. A prediction can therefore be wrong at four digits while still
landing in the correct occupational family. This script measures that directly:
accuracy at each level, the conditional structure of four-digit failures, the
per-language pattern, and the gold codes that absorb the most error.

Everything is recomputed from preserved per-case predictions using only the
standard library. No model is loaded, no network call is made, and no
configuration is selected or tuned here -- this describes runs that already
happened.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEVELS = ("1digit", "2digit", "3digit", "4digit")
LEVEL_NAMES = {
    "1digit": "major group",
    "2digit": "sub-major group",
    "3digit": "minor group",
    "4digit": "unit group",
}


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream)]
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def analyse(rows: list[dict]) -> dict:
    n = len(rows)
    report: dict = {"n": n}

    # Accuracy at each hierarchy level.
    levels = {}
    for level in LEVELS:
        gold_key, pred_key = f"gold_isco_{level}", f"pred_isco_{level}"
        correct = sum(
            1 for r in rows
            if r[gold_key].strip() and r[gold_key].strip() == r[pred_key].strip()
        )
        levels[level] = {
            "name": LEVEL_NAMES[level],
            "correct": correct,
            "accuracy_percent": 100 * correct / n,
        }
    report["accuracy_by_level"] = levels

    # Structure of the four-digit failures: how far off are they?
    failures = [r for r in rows if r["gold_isco_4digit"].strip() != r["pred_isco_4digit"].strip()]
    blank = sum(1 for r in failures if not r["pred_isco_4digit"].strip())
    same_major = sum(
        1 for r in failures
        if r["pred_isco_4digit"].strip()
        and r["gold_isco_1digit"].strip() == r["pred_isco_1digit"].strip()
    )
    same_minor = sum(
        1 for r in failures
        if r["pred_isco_4digit"].strip()
        and r["gold_isco_3digit"].strip() == r["pred_isco_3digit"].strip()
    )
    report["four_digit_failures"] = {
        "n": len(failures),
        "no_prediction_returned": blank,
        "right_major_group_wrong_unit": same_major,
        "right_minor_group_wrong_unit": same_minor,
        "right_major_group_percent_of_failures": 100 * same_major / len(failures) if failures else 0.0,
        "right_minor_group_percent_of_failures": 100 * same_minor / len(failures) if failures else 0.0,
    }

    # Per-language: exact accuracy and how much of the failure is near-miss.
    by_language = {}
    for language in sorted({r["input_language"] for r in rows}):
        subset = [r for r in rows if r["input_language"] == language]
        exact = sum(1 for r in subset if r["gold_isco_4digit"].strip() == r["pred_isco_4digit"].strip())
        major = sum(1 for r in subset if r["gold_isco_1digit"].strip() == r["pred_isco_1digit"].strip())
        sub_failures = [r for r in subset if r["gold_isco_4digit"].strip() != r["pred_isco_4digit"].strip()]
        near = sum(
            1 for r in sub_failures
            if r["pred_isco_4digit"].strip()
            and r["gold_isco_1digit"].strip() == r["pred_isco_1digit"].strip()
        )
        by_language[language] = {
            "n": len(subset),
            "exact_correct": exact,
            "exact_percent": 100 * exact / len(subset),
            "major_group_correct": major,
            "major_group_percent": 100 * major / len(subset),
            "near_miss_share_of_failures_percent": 100 * near / len(sub_failures) if sub_failures else 0.0,
        }
    report["by_language"] = by_language

    # Where the error concentrates: gold codes most often missed, and the codes
    # predicted in their place.
    missed = collections.Counter(r["gold_isco_4digit"].strip() for r in failures)
    report["most_missed_gold_codes"] = [
        {"gold": code, "missed": count, "percent_of_all_failures": 100 * count / len(failures)}
        for code, count in missed.most_common(10)
    ]
    confusions = collections.Counter(
        (r["gold_isco_4digit"].strip(), r["pred_isco_4digit"].strip())
        for r in failures if r["pred_isco_4digit"].strip()
    )
    report["most_common_confusions"] = [
        {"gold": g, "predicted": p, "count": c, "same_major_group": g[:1] == p[:1]}
        for (g, p), c in confusions.most_common(10)
    ]

    # Does the confidence score separate right from wrong? If it does not, the
    # score cannot be used to route cases to human review.
    def mean_conf(subset):
        vals = [float(r["pred_confidence"]) for r in subset if r.get("pred_confidence", "").strip()]
        return sum(vals) / len(vals) if vals else None

    correct_rows = [r for r in rows if r["gold_isco_4digit"].strip() == r["pred_isco_4digit"].strip()]
    report["confidence_separation"] = {
        "mean_confidence_correct": mean_conf(correct_rows),
        "mean_confidence_incorrect": mean_conf(failures),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = load(args.predictions)
    report = {"label": args.label or args.predictions.name, "source": str(args.predictions)}
    report.update(analyse(rows))

    lv = report["accuracy_by_level"]
    print(f"{report['label']}  (n={report['n']:,})")
    for level in LEVELS:
        b = lv[level]
        print(f"  {b['name']:<16} {b['correct']:>6,}  {b['accuracy_percent']:>6.2f}%")
    f = report["four_digit_failures"]
    print(f"  unit-group failures: {f['n']:,}")
    print(f"    right major group, wrong unit : {f['right_major_group_wrong_unit']:,}"
          f" ({f['right_major_group_percent_of_failures']:.1f}% of failures)")
    print(f"    right minor group, wrong unit : {f['right_minor_group_wrong_unit']:,}"
          f" ({f['right_minor_group_percent_of_failures']:.1f}% of failures)")
    print(f"    no prediction returned        : {f['no_prediction_returned']:,}")
    c = report["confidence_separation"]
    if c["mean_confidence_correct"] is not None:
        print(f"  mean confidence  correct {c['mean_confidence_correct']:.3f}"
              f" / incorrect {c['mean_confidence_incorrect']:.3f}")
    print("  per language (exact / major-group / near-miss share of failures):")
    for lang, b in report["by_language"].items():
        print(f"    {lang:<8} {b['exact_percent']:>6.2f}%  {b['major_group_percent']:>6.2f}%"
              f"  {b['near_miss_share_of_failures_percent']:>6.1f}%")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
