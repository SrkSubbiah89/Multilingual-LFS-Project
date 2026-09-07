"""
eval/gulf_arabic_dialect_normalization_ab_test.py

Closes a real, previously-blocked gap (Module G, "multilingual
validation" -- see CLAUDE.md): the planned Gulf Arabic dialect-
normalization A/B test could not run because WISCO's Arabic data was
confirmed to have zero dialectal content -- there was no real Gulf-
dialect text to test `LanguageProcessor._normalise_gulf_arabic()`
(backend/agents/language_processor.py's 79-rule dialect→MSA dictionary)
against. This was blocked purely on data availability, not on missing
code -- the normalization function has existed the whole time.

**A different data source, as Module G's own note said was needed**: the
`ar-gulf` rows of eval/generate_synthetic_isic_iscedf_benchmark.py's
synthetic benchmark are genuine Gulf-dialect text (LLM-instructed to use
colloquial Khaleeji Arabic specifically, e.g. observed using "إحنا" the
Gulf/colloquial first-person plural, not MSA "نحن") -- not WISCO, and
still synthetic (same disclosure as that script), but a real, usable
minimal-pair test: the SAME underlying job/field concept exists in both
`ar` (MSA) and `ar-gulf` rows, generated from the identical official
definition, so any accuracy difference between them isolates the
dialect's effect specifically.

**What this measures, precisely**:
1. Marker detection rate: how often `LanguageProcessor._apply_gulf_detection()`
   actually upgrades "ar" to "ar-gulf" on these real Gulf-generated rows
   (tests whether the 79-term marker dictionary has practical coverage of
   how an LLM naturally produces Gulf dialect, not just the terms the
   dictionary's author anticipated).
2. Classification accuracy A/B: ISIC_FLAT_RETRIEVAL accuracy on the raw
   ar-gulf text vs. the SAME text run through `_normalise_gulf_arabic()`
   first -- does dialect normalization help, hurt, or make no measurable
   difference to downstream classification.

Usage
-----
    python -m eval.gulf_arabic_dialect_normalization_ab_test --input <synthetic_benchmark_csv>
"""

from __future__ import annotations

import argparse
import csv
import sys

from eval.analyze import mcnemar_test, wilson_score_interval
from eval.run_synthetic_isic_iscedf_eval import _latest_csv, _load_rows


def run(input_path: str) -> None:
    from backend.agents.classifier_methods import ISCEDF_FLAT_RETRIEVAL, ISIC_FLAT_RETRIEVAL
    from backend.agents.isced_classifier import ISCEDClassifier
    from backend.agents.isic_classifier import ISICClassifier
    from backend.agents.language_processor import LanguageProcessor

    rows = _load_rows(input_path)
    gulf_rows = [r for r in rows if r["input_language"] == "ar-gulf"]
    print(f"Loaded {len(rows)} total rows, {len(gulf_rows)} are ar-gulf.")
    if not gulf_rows:
        print("No ar-gulf rows found in this file -- nothing to test.")
        return

    lp = LanguageProcessor()
    isic_clf = ISICClassifier()
    isced_clf = ISCEDClassifier()

    detected_gulf = 0
    normalization_changed_text = 0
    raw_correct = 0
    normalized_correct = 0
    b = c = 0  # McNemar: b = raw right/normalized wrong, c = raw wrong/normalized right

    for row in gulf_rows:
        text = row["input_text"]
        gold = row["gold_code"]
        standard = row["standard"]

        detected_lang = lp._apply_gulf_detection("ar", text)
        if detected_lang == "ar-gulf":
            detected_gulf += 1

        normalized = lp._normalise_gulf_arabic(text)
        if normalized != text:
            normalization_changed_text += 1

        if standard == "isic":
            pred_raw = isic_clf.classify(text, method=ISIC_FLAT_RETRIEVAL).class_code
            pred_norm = isic_clf.classify(normalized, method=ISIC_FLAT_RETRIEVAL).class_code
        else:
            pred_raw = isced_clf.classify(text, method=ISCEDF_FLAT_RETRIEVAL).detailed_code
            pred_norm = isced_clf.classify(normalized, method=ISCEDF_FLAT_RETRIEVAL).detailed_code

        raw_ok = pred_raw == gold
        norm_ok = pred_norm == gold
        raw_correct += raw_ok
        normalized_correct += norm_ok
        if raw_ok and not norm_ok:
            b += 1
        elif not raw_ok and norm_ok:
            c += 1

    n = len(gulf_rows)
    raw_acc = raw_correct / n
    norm_acc = normalized_correct / n
    raw_lo, raw_hi = wilson_score_interval(raw_correct, n)
    norm_lo, norm_hi = wilson_score_interval(normalized_correct, n)
    stat, p = mcnemar_test(b, c)

    print("\n" + "=" * 70)
    print("GULF ARABIC DIALECT-NORMALIZATION A/B TEST -- synthetic data,")
    print("see this script's own docstring before citing any number below.")
    print("=" * 70)
    print(f"\nn = {n} ar-gulf synthetic rows")
    print(f"Marker-based ar-gulf detection rate: {detected_gulf}/{n} = {detected_gulf/n:.4f}")
    print(f"Normalization actually changed the text: {normalization_changed_text}/{n} = {normalization_changed_text/n:.4f}")
    print(f"\nISIC_FLAT_RETRIEVAL / ISCEDF_FLAT_RETRIEVAL accuracy:")
    print(f"  raw ar-gulf text        : {raw_correct}/{n} = {raw_acc:.4f}  95% CI [{raw_lo:.4f}, {raw_hi:.4f}]")
    print(f"  normalized (MSA-ified)  : {normalized_correct}/{n} = {norm_acc:.4f}  95% CI [{norm_lo:.4f}, {norm_hi:.4f}]")
    print(f"  McNemar: b={b} (raw won) c={c} (normalized won) p={p:.4g}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None)
    parser.add_argument("--generation-dir", default="eval/results/synthetic_isic_iscedf_benchmark")
    args = parser.parse_args()
    input_path = args.input or _latest_csv(args.generation_dir)
    run(input_path)


if __name__ == "__main__":
    main()
