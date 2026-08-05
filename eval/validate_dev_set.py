"""
eval/validate_dev_set.py

Leakage-safety and coverage validator for eval/dev_set_v1.csv -- the B2
development set used only to select --reranker-candidates (K) before the
single, pre-specified confirmation run on eval/test_set_full130.csv. See
eval/dev_set_schema.md for the full field-by-field schema and the rationale
for why this file must be independent of both eval/test_set_smoke20.csv
(reused for prior model/beam/routing experiments) and
eval/test_set_full130.csv (the frozen confirmation set).

This script does not run any classification -- it only checks the CSV
itself. Run it after you've filled in dev_set_v1.csv and before it is ever
passed to dev_sweep.py.

Usage
-----
    python eval/validate_dev_set.py --dev-set eval/dev_set_v1.csv

Exit codes
----------
0   No hard errors. May still print warnings (e.g. below the *preferred*
    50-case / 15-Arabic target but at/above the 30-case absolute minimum).
1   At least one hard error (leakage against smoke20/full130, missing
    required column, below the 30-case absolute minimum, no cases in one
    of the two languages, malformed gold code, etc.) -- do not use this
    file for K selection until every hard error is fixed.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED_COLUMNS = [
    "case_id", "language", "respondent_text", "gold_isco_code",
    "gold_label_source", "coder_or_adjudicator", "major_group",
    "difficulty_level", "notes",
]

VALID_LANGUAGES = {"en", "ar", "mixed"}
VALID_DIFFICULTY = {"easy", "medium", "hard"}

PREFERRED_MIN_TOTAL = 50
PREFERRED_MIN_ARABIC = 15
ABSOLUTE_MIN_TOTAL = 30

_DEFAULT_DIR = Path(__file__).resolve().parent
_DEFAULT_SMOKE20 = _DEFAULT_DIR / "test_set_smoke20.csv"
_DEFAULT_FULL130 = _DEFAULT_DIR / "test_set_full130.csv"


def normalize_text(s: str) -> str:
    """Lowercase + collapse internal whitespace. Intentionally mechanical,
    not semantic -- see dev_set_schema.md's "Independence requirements"
    note: this catches exact/whitespace/case duplicates only; a human
    reviewer should still check for paraphrases that survive this check."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_case_ids(rows: list[dict], id_column: str = "case_id") -> set[str]:
    return {r[id_column] for r in rows if r.get(id_column)}


def load_normalized_texts(rows: list[dict], text_column: str) -> set[str]:
    return {normalize_text(r[text_column]) for r in rows if r.get(text_column)}


@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_dev_set(
    dev_rows: list[dict],
    other_case_ids: set,
    other_normalized_texts: set,
    other_major_groups: set = None,
) -> ValidationReport:
    """Pure validation logic, no filesystem/CLI -- takes already-loaded rows
    plus the case_ids/normalized input texts pooled from smoke20+full130
    (callers combine both sets before calling this; the check doesn't care
    which of the two a collision came from, both are equally disqualifying).
    other_major_groups (optional): major groups present in full130, used
    only to flag zero-coverage groups as a warning, never a hard error."""
    report = ValidationReport()
    other_major_groups = other_major_groups or set()

    if not dev_rows:
        report.errors.append("Dev set is empty (no data rows).")
        return report

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in dev_rows[0]]
    if missing_cols:
        report.errors.append(f"Missing required column(s): {', '.join(missing_cols)}")
        return report  # remaining checks assume all columns exist

    seen_ids: set = set()
    seen_texts: dict = {}  # normalized text -> case_id that first used it
    lang_counts = {"en": 0, "ar": 0, "mixed": 0}
    major_group_counts: dict = {}
    unknown_lang_rows = []

    for row in dev_rows:
        cid = (row.get("case_id") or "").strip()
        text = row.get("respondent_text") or ""
        norm_text = normalize_text(text)
        lang = (row.get("language") or "").strip().lower()
        gold = (row.get("gold_isco_code") or "").strip()
        major = (row.get("major_group") or "").strip()
        label_source = (row.get("gold_label_source") or "").strip()
        difficulty = (row.get("difficulty_level") or "").strip().lower()

        if not cid:
            report.errors.append("Row with blank case_id.")
            continue
        if cid in seen_ids:
            report.errors.append(f"Duplicate case_id within dev set: {cid!r}")
        seen_ids.add(cid)

        if cid in other_case_ids:
            report.errors.append(
                f"case_id {cid!r} collides with an existing smoke20/full130 case_id."
            )

        if not text.strip():
            report.errors.append(f"case_id={cid!r}: respondent_text is blank.")
        elif norm_text in other_normalized_texts:
            report.errors.append(
                f"case_id={cid!r}: respondent_text duplicates (or near-duplicates, "
                f"after whitespace/case normalisation) an existing smoke20/full130 case."
            )
        elif norm_text in seen_texts:
            report.errors.append(
                f"case_id={cid!r}: respondent_text duplicates case_id="
                f"{seen_texts[norm_text]!r} within the dev set itself."
            )
        seen_texts.setdefault(norm_text, cid)

        if lang not in VALID_LANGUAGES:
            unknown_lang_rows.append(cid)
        else:
            lang_counts[lang] += 1

        if not re.fullmatch(r"\d{4}", gold):
            report.errors.append(
                f"case_id={cid!r}: gold_isco_code {gold!r} is not a 4-digit code."
            )

        if not major:
            report.errors.append(f"case_id={cid!r}: major_group is blank.")
        elif gold[:1] and major != gold[:1]:
            report.errors.append(
                f"case_id={cid!r}: major_group={major!r} does not match "
                f"gold_isco_code[0]={gold[:1]!r}."
            )
        major_group_counts[major] = major_group_counts.get(major, 0) + 1

        if not label_source:
            report.errors.append(f"case_id={cid!r}: gold_label_source is blank.")

        if difficulty and difficulty not in VALID_DIFFICULTY:
            report.warnings.append(
                f"case_id={cid!r}: difficulty_level {difficulty!r} is not one of "
                f"{sorted(VALID_DIFFICULTY)} (informational only, not scored)."
            )

    if unknown_lang_rows:
        report.errors.append(
            f"{len(unknown_lang_rows)} row(s) have a language outside "
            f"{sorted(VALID_LANGUAGES)}: case_ids={unknown_lang_rows}"
        )

    total = len(dev_rows)
    report.stats.update({
        "total_cases": total,
        "language_counts": lang_counts,
        "major_group_counts": major_group_counts,
    })

    if total < ABSOLUTE_MIN_TOTAL:
        report.errors.append(
            f"Only {total} case(s); absolute minimum is {ABSOLUTE_MIN_TOTAL}."
        )
    elif total < PREFERRED_MIN_TOTAL:
        report.warnings.append(
            f"Only {total} case(s); preferred target is {PREFERRED_MIN_TOTAL}+ "
            f"(above the {ABSOLUTE_MIN_TOTAL}-case absolute minimum, so this is "
            f"usable, but a larger dev set would give a more reliable K choice)."
        )

    if lang_counts["ar"] < PREFERRED_MIN_ARABIC:
        report.warnings.append(
            f"Only {lang_counts['ar']} Arabic case(s); preferred target is "
            f"{PREFERRED_MIN_ARABIC}+."
        )
    if lang_counts["en"] == 0:
        report.errors.append("No English cases at all.")
    if lang_counts["ar"] == 0:
        report.errors.append("No Arabic cases at all.")

    missing_major_groups = other_major_groups - set(major_group_counts.keys())
    if missing_major_groups:
        report.warnings.append(
            f"Major group(s) with zero dev-set coverage but present in "
            f"test_set_full130.csv: {sorted(missing_major_groups)}"
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-set", required=True, type=Path)
    parser.add_argument("--smoke20", type=Path, default=_DEFAULT_SMOKE20)
    parser.add_argument("--full130", type=Path, default=_DEFAULT_FULL130)
    args = parser.parse_args()

    if not args.dev_set.exists():
        parser.error(f"Dev set not found: {args.dev_set}")

    dev_rows = load_csv_rows(args.dev_set)
    smoke20_rows = load_csv_rows(args.smoke20)
    full130_rows = load_csv_rows(args.full130)

    if not smoke20_rows:
        print(f"WARNING: could not load {args.smoke20} (0 rows) -- case_id/text "
              f"collision checks against it will be skipped.", file=sys.stderr)
    if not full130_rows:
        print(f"WARNING: could not load {args.full130} (0 rows) -- case_id/text "
              f"collision checks against it will be skipped.", file=sys.stderr)

    other_ids = load_case_ids(smoke20_rows) | load_case_ids(full130_rows)
    other_texts = (
        load_normalized_texts(smoke20_rows, "input_text")
        | load_normalized_texts(full130_rows, "input_text")
    )
    other_major_groups = {
        (r.get("gold_isco_4digit") or "")[:1]
        for r in full130_rows
        if (r.get("gold_isco_4digit") or "")[:1]
    }

    report = validate_dev_set(dev_rows, other_ids, other_texts, other_major_groups)

    print(f"Loaded {len(dev_rows)} row(s) from {args.dev_set}")
    if report.stats:
        print(f"stats: {report.stats}")
    if report.warnings:
        print(f"\n{len(report.warnings)} warning(s):")
        for w in report.warnings:
            print(f"  WARNING: {w}")
    if report.errors:
        print(f"\n{len(report.errors)} error(s):")
        for e in report.errors:
            print(f"  ERROR: {e}")
        print("\nFAIL: fix the error(s) above before using this file for K selection.")
        sys.exit(1)

    print("\nPASS: no hard errors." + (" (see warnings above)" if report.warnings else ""))


if __name__ == "__main__":
    main()
