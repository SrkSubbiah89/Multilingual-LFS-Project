"""
eval/run_synthetic_isic_iscedf_eval.py

Scores ISICClassifier / ISCEDClassifier's existing legacy keyword/LLM
pipeline against the new flat-retrieval method (ISIC_FLAT_RETRIEVAL /
ISCEDF_FLAT_RETRIEVAL -- enriched real official text + e5-large, ISCO-08's
own best-tested recipe) on the synthetic benchmark produced by
eval/generate_synthetic_isic_iscedf_benchmark.py.

**Read this before citing any number from this script's output.** The
benchmark is SYNTHETIC (LLM-generated, grounded in real official
definitions, never real respondent data -- see that script's own
docstring for the full disclosure). A result here is a genuine,
real-computed accuracy number over real generated text, but it answers
"how well does each method recover the class its own generating prompt
was built from" -- not "how well would this classify real survey
respondents." It is the best available signal in the current absence of
real ISIC/ISCED-F evaluation data (WISCO is occupation-only; IPUMS's
correspondence closed 2026-08-27 with a definitive negative answer --
see Documentation/Phase_2/Week_1/ipums_correspondence_log.md; Module E
pilot hasn't started), genuinely useful for
regression-testing and directional comparison between methods, but never
to be reported as WISCO-equivalent or pilot-validated accuracy.

Usage
-----
    python -m eval.run_synthetic_isic_iscedf_eval --input <path_to_csv>
    # or, to use the most recently generated file automatically:
    python -m eval.run_synthetic_isic_iscedf_eval
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

from eval.analyze import mcnemar_test, wilson_score_interval


def _latest_csv(out_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(out_dir, "synthetic_isic_iscedf_benchmark_*.csv")))
    if not candidates:
        raise FileNotFoundError(f"No synthetic benchmark CSV found in {out_dir}")
    return candidates[-1]


def _load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _classify_row(row: dict, isic_clf, isced_clf, isic_flat_const, iscedf_flat_const) -> dict:
    text = row["input_text"]
    standard = row["standard"]
    gold = row["gold_code"]

    if standard == "isic":
        legacy = isic_clf.classify(text)
        flat = isic_clf.classify(text, method=isic_flat_const)
        pred_legacy = legacy.class_code
        pred_flat = flat.class_code
    else:
        legacy = isced_clf.classify(text)
        flat = isced_clf.classify(text, method=iscedf_flat_const)
        pred_legacy = legacy.detailed_code
        pred_flat = flat.detailed_code

    return {
        "case_id": row["case_id"],
        "standard": standard,
        "language": row["input_language"],
        "gold_code": gold,
        "pred_legacy": pred_legacy,
        "correct_legacy": int(pred_legacy == gold),
        "legacy_method": legacy.method,
        "pred_flat": pred_flat,
        "correct_flat": int(pred_flat == gold),
        "flat_method": flat.method,
        "flat_fallback_used": getattr(flat, "fallback_used", False),
    }


def run(input_path: str, out_dir: str) -> None:
    from backend.agents.classifier_methods import ISCEDF_FLAT_RETRIEVAL, ISIC_FLAT_RETRIEVAL
    from backend.agents.isced_classifier import ISCEDClassifier
    from backend.agents.isic_classifier import ISICClassifier

    rows = _load_rows(input_path)
    print(f"Loaded {len(rows)} synthetic cases from {input_path}")

    isic_clf = ISICClassifier()
    isced_clf = ISCEDClassifier()

    results = []
    for i, row in enumerate(rows, 1):
        r = _classify_row(row, isic_clf, isced_clf, ISIC_FLAT_RETRIEVAL, ISCEDF_FLAT_RETRIEVAL)
        results.append(r)
        if i % 50 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}]")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = Path(out_dir) / f"synthetic_eval_results_{ts}.csv"
    with open(results_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote per-case results to {results_path}")

    _report(results)


def _report(results: list[dict]) -> None:
    def _acc_block(rows: list[dict], label: str) -> None:
        n = len(rows)
        if n == 0:
            return
        legacy_correct = sum(r["correct_legacy"] for r in rows)
        flat_correct = sum(r["correct_flat"] for r in rows)
        legacy_acc = legacy_correct / n
        flat_acc = flat_correct / n
        l_lo, l_hi = wilson_score_interval(legacy_correct, n)
        f_lo, f_hi = wilson_score_interval(flat_correct, n)

        # McNemar: b = legacy right, flat wrong; c = legacy wrong, flat right
        b = sum(1 for r in rows if r["correct_legacy"] and not r["correct_flat"])
        c = sum(1 for r in rows if not r["correct_legacy"] and r["correct_flat"])
        stat, p = mcnemar_test(b, c)

        fallback_rate = sum(1 for r in rows if r["flat_fallback_used"]) / n

        print(f"\n=== {label} (n={n}) ===")
        print(f"  legacy keyword/LLM : {legacy_correct}/{n} = {legacy_acc:.4f}  95% CI [{l_lo:.4f}, {l_hi:.4f}]")
        print(f"  flat_retrieval     : {flat_correct}/{n} = {flat_acc:.4f}  95% CI [{f_lo:.4f}, {f_hi:.4f}]")
        print(f"  McNemar: b={b} c={c} p={p:.4g}")
        print(f"  flat_retrieval fallback rate: {fallback_rate:.4f}")

    print("\n" + "=" * 70)
    print("SYNTHETIC BENCHMARK RESULT -- see this script's own module docstring")
    print("before citing any number below. Not WISCO-equivalent, not pilot data.")
    print("=" * 70)

    _acc_block(results, "OVERALL")
    for standard in ("isic", "iscedf"):
        _acc_block([r for r in results if r["standard"] == standard], f"standard={standard}")

    by_lang = defaultdict(list)
    for r in results:
        by_lang[r["language"]].append(r)
    for lang in sorted(by_lang):
        _acc_block(by_lang[lang], f"language={lang}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None, help="Path to a synthetic benchmark CSV. Defaults to the most recent file in --generation-dir.")
    parser.add_argument("--generation-dir", default="eval/results/synthetic_isic_iscedf_benchmark")
    parser.add_argument("--out", default="eval/results/synthetic_isic_iscedf_benchmark")
    args = parser.parse_args()

    input_path = args.input or _latest_csv(args.generation_dir)
    run(input_path, args.out)


if __name__ == "__main__":
    main()
