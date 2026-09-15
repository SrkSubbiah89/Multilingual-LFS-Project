"""
eval/run_query_planning_eval.py

Real evaluation of Item 2 (multi-step agentic retrieval / query planning,
2026-09-12) -- does decomposing an ambiguous description into sub-queries
and reconciling actually improve accuracy over the existing single-shot
retrieval? Reuses the same n=60 synthetic combined benchmark built for
Item 1 (eval/generate_synthetic_coordination_benchmark.py) -- no need for
new data, since query planning's trigger condition (result still
ambiguous after normal reranking) is generic, not specific to visibly
compound text.

Three independent, paired comparisons -- evaluated one at a time, per
this project's own standing discipline of never confounding two
experimental flags in one measurement (see enable_query_planning's own
docstring for why this matters: it's mutually exclusive with
enable_corrective_retry in practice for the same reason):

  ISCO  -- ISCOClassifier(enable_query_planning=True) vs baseline, on job_title
  ISIC  -- ISICClassifier(enable_query_planning=True) vs baseline, on industry_text
  ISCED -- ISCEDClassifier(enable_query_planning=True) vs baseline, on education_text

ISIC/ISCED have no gold labels in this benchmark (see Item 1's own eval
script docstring for why -- WISCO is occupation-only and this benchmark
was built to fill that specific gap, not a general one), so only ISCO's
comparison produces an accuracy number; ISIC/ISCED comparisons report
prediction-agreement rate (how often query planning changed the answer
at all) as a secondary, always-computable signal of how often the
mechanism even has a chance to matter.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.analyze import mcnemar_test, wilson_score_interval

DEFAULT_INPUT = Path(__file__).resolve().parent / "results" / "synthetic_coordination_benchmark" / "benchmark.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "synthetic_query_planning_eval"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--isco-profile", default=None,
        help="isco_catalogue_profile for both ISCO classifiers (default: ISCOClassifier's own "
             "default, LEGACY_PROFILE). Pass official_ilo2021_v1_enriched_e5large for this "
             "project's real best-tested config.",
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} synthetic cases from {args.input}")

    print("Constructing classifiers (this takes a while)...")
    from backend.agents.isco_classifier import ISCOClassifier
    from backend.agents.isic_classifier import ISICClassifier
    from backend.agents.isced_classifier import ISCEDClassifier

    isco_kwargs = {"isco_catalogue_profile": args.isco_profile} if args.isco_profile else {}
    isco_baseline = ISCOClassifier(**isco_kwargs)
    isco_planned = ISCOClassifier(enable_query_planning=True, **isco_kwargs)
    isic_baseline = ISICClassifier()
    isic_planned = ISICClassifier(enable_query_planning=True)
    isced_baseline = ISCEDClassifier()
    isced_planned = ISCEDClassifier(enable_query_planning=True)
    print("Classifiers ready.\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── ISCO: real accuracy comparison ──────────────────────────────────
    print("=" * 70)
    print("ISCO (job_title) -- accuracy comparison")
    print("=" * 70)
    isco_results = []
    b = c = 0
    baseline_correct = planned_correct = 0
    for i, row in enumerate(rows, 1):
        gold = row["gold_isco_4digit"].strip()
        job_title = row["job_title"]
        t0 = time.perf_counter()
        base = isco_baseline.classify(job_title)
        planned = isco_planned.classify(job_title)
        elapsed = time.perf_counter() - t0

        base_ok = base.primary.code == gold
        planned_ok = planned.primary.code == gold
        if base_ok:
            baseline_correct += 1
        if planned_ok:
            planned_correct += 1
        if base_ok and not planned_ok:
            b += 1
        elif not base_ok and planned_ok:
            c += 1

        isco_results.append({
            "gold_isco_4digit": gold, "job_title": job_title,
            "baseline_code": base.primary.code, "baseline_correct": base_ok,
            "planned_code": planned.primary.code, "planned_correct": planned_ok,
            "planned_method": planned.method,
        })
        print(f"[{i}/{len(rows)}] gold={gold} baseline={base.primary.code}({base_ok}) "
              f"planned={planned.primary.code}({planned_ok}) method={planned.method} ({elapsed:.2f}s)")

    n = len(rows)
    baseline_acc = baseline_correct / n
    planned_acc = planned_correct / n
    _, p_value = mcnemar_test(b, c)
    with (args.output_dir / "isco_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(isco_results[0].keys()))
        writer.writeheader()
        writer.writerows(isco_results)

    print(f"\nISCO baseline accuracy: {baseline_acc:.2%} ({baseline_correct}/{n}), "
          f"95% CI {wilson_score_interval(baseline_correct, n)}")
    print(f"ISCO planned  accuracy: {planned_acc:.2%} ({planned_correct}/{n}), "
          f"95% CI {wilson_score_interval(planned_correct, n)}")
    print(f"McNemar: b={b}, c={c}, p={p_value:.4f}")

    # ── ISIC / ISCED: prediction-agreement rate (no gold labels here) ──
    for label, baseline_clf, planned_clf, text_key in [
        ("ISIC", isic_baseline, isic_planned, "industry_text"),
        ("ISCED", isced_baseline, isced_planned, "education_text"),
    ]:
        print("\n" + "=" * 70)
        print(f"{label} ({text_key}) -- no gold label in this benchmark; "
              f"reporting prediction-agreement rate only")
        print("=" * 70)
        results = []
        changed = 0
        for i, row in enumerate(rows, 1):
            text = row[text_key]
            base = baseline_clf.classify(text)
            planned = planned_clf.classify(text)
            base_key = base.section if label == "ISIC" else base.level
            planned_key = planned.section if label == "ISIC" else planned.level
            is_changed = base_key != planned_key
            if is_changed:
                changed += 1
            results.append({
                "text": text, "baseline": base_key, "planned": planned_key,
                "changed": is_changed, "planned_method": planned.method,
            })
            print(f"[{i}/{len(rows)}] baseline={base_key} planned={planned_key} "
                  f"changed={is_changed} method={planned.method}")

        with (args.output_dir / f"{label.lower()}_results.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n{label}: query planning changed the prediction in {changed}/{n} cases "
              f"({changed/n:.1%})")

    print(f"\nAll results written to {args.output_dir}")


if __name__ == "__main__":
    main()
