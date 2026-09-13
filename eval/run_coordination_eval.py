"""
eval/run_coordination_eval.py

Real evaluation of Item 1 (cross-standard coordination, 2026-09-12) --
does ISCO accuracy actually improve when ISIC/ISCED evidence is allowed
to revise or bias an uncertain ISCO guess? Runs against the synthetic
combined benchmark from eval/generate_synthetic_coordination_benchmark.py
(see that script's own docstring for why WISCO and the existing
ISIC/ISCED-F synthetic benchmark can't answer this question -- neither
pairs occupation+industry+education for the same case).

Two arms, same cases, paired comparison:
  BASELINE    -- ISCOClassifier with no coordination (today's production
                 default: ENABLE_COORDINATED_CLASSIFICATION unset).
  COORDINATED -- ISICClassifier(use_cross_classification_hints=True) gets
                 the baseline ISCO code as a hint, then
                 cross_standard_coordinator.maybe_revise_isco_with_cross_signal
                 gets a chance to revise the ISCO primary using the
                 (possibly hint-biased) ISIC result plus the ISCED result
                 -- i.e. the EXACT sequence survey_routes.py's Stage
                 4/4b/4e run when ENABLE_COORDINATED_CLASSIFICATION=true,
                 replicated directly against real classifiers rather than
                 through the HTTP layer.

Reports: exact-match ISCO accuracy for both arms, Wilson 95% CI, McNemar
exact test on the paired outcomes, and secondary counts (how often the
ISIC cross-hint actually changed the ISIC result, how often the backward
coordinator actually revised ISCO) -- an honest "how often did this even
have a chance to matter" number, not just the aggregate accuracy delta.
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
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "synthetic_coordination_benchmark" / "eval_results.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} synthetic cases from {args.input}")

    print("Constructing classifiers (this takes a while -- loading embedding models/Qdrant collections)...")
    from backend.agents.isco_classifier import ISCOClassifier
    from backend.agents.isic_classifier import ISICClassifier
    from backend.agents.isced_classifier import ISCEDClassifier
    from backend.agents.cross_standard_coordinator import maybe_revise_isco_with_cross_signal

    isco_clf = ISCOClassifier()
    isic_baseline_clf = ISICClassifier()
    isic_hinted_clf = ISICClassifier(use_cross_classification_hints=True)
    isced_clf = ISCEDClassifier()
    print("Classifiers ready.\n")

    results = []
    baseline_correct = 0
    coordinated_correct = 0
    isic_hint_changed_count = 0
    isco_revised_count = 0
    b = c = 0  # McNemar discordant pair counts

    for i, row in enumerate(rows, 1):
        gold = row["gold_isco_4digit"].strip()
        job_title = row["job_title"]
        industry_text = row["industry_text"]
        education_text = row["education_text"]

        t0 = time.perf_counter()

        isco_result = isco_clf.classify(job_title)
        baseline_isco_code = isco_result.primary.code
        baseline_ok = baseline_isco_code == gold

        isic_baseline_result = isic_baseline_clf.classify(industry_text)
        isced_result = isced_clf.classify(education_text)

        isic_hinted_result = isic_hinted_clf.classify(
            industry_text, cross_hints={"isco_code": baseline_isco_code}
        )
        hint_changed = isic_hinted_result.section != isic_baseline_result.section

        # Mirror survey_routes.py's ISCOResult shape well enough for the
        # coordinator function (it only reads .primary_code/.hitl_required/
        # .alternatives via getattr, real or SimpleNamespace both work).
        from types import SimpleNamespace
        isco_result_shim = SimpleNamespace(
            primary_code=baseline_isco_code,
            hitl_required=isco_result.hitl_required,
            alternatives=isco_result.alternatives,
        )
        promoted, reason = maybe_revise_isco_with_cross_signal(
            isco_result_shim, isic_hinted_result.section, isced_result.level,
        )
        coordinated_isco_code = promoted.code if promoted is not None else baseline_isco_code
        coordinated_ok = coordinated_isco_code == gold

        elapsed = time.perf_counter() - t0

        if baseline_ok:
            baseline_correct += 1
        if coordinated_ok:
            coordinated_correct += 1
        if hint_changed:
            isic_hint_changed_count += 1
        if promoted is not None:
            isco_revised_count += 1

        if baseline_ok and not coordinated_ok:
            b += 1
        elif not baseline_ok and coordinated_ok:
            c += 1

        results.append({
            "gold_isco_4digit": gold,
            "job_title": job_title,
            "baseline_isco_code": baseline_isco_code,
            "baseline_correct": baseline_ok,
            "coordinated_isco_code": coordinated_isco_code,
            "coordinated_correct": coordinated_ok,
            "isic_hint_changed": hint_changed,
            "isco_revised": promoted is not None,
            "revision_reason": reason,
        })

        flag = "SAME" if baseline_ok == coordinated_ok else ("BASE->COORD" if coordinated_ok else "COORD->BASE")
        print(f"[{i}/{len(rows)}] gold={gold} baseline={baseline_isco_code}({baseline_ok}) "
              f"coordinated={coordinated_isco_code}({coordinated_ok}) [{flag}] ({elapsed:.2f}s)")

    n = len(rows)
    baseline_acc = baseline_correct / n
    coordinated_acc = coordinated_correct / n
    baseline_ci = wilson_score_interval(baseline_correct, n)
    coordinated_ci = wilson_score_interval(coordinated_correct, n)
    _, p_value = mcnemar_test(b, c)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 70)
    print(f"n = {n}")
    print(f"Baseline    accuracy: {baseline_acc:.2%} ({baseline_correct}/{n}), "
          f"95% CI [{baseline_ci[0]:.2%}, {baseline_ci[1]:.2%}]")
    print(f"Coordinated accuracy: {coordinated_acc:.2%} ({coordinated_correct}/{n}), "
          f"95% CI [{coordinated_ci[0]:.2%}, {coordinated_ci[1]:.2%}]")
    print(f"McNemar: b(base right/coord wrong)={b}, c(base wrong/coord right)={c}, p={p_value:.4f}")
    print(f"ISIC cross-hint changed the ISIC result in {isic_hint_changed_count}/{n} cases")
    print(f"Backward coordinator revised ISCO in {isco_revised_count}/{n} cases")
    print(f"Results written to {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
