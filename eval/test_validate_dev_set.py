"""
Tests for eval/validate_dev_set.py -- pure CSV-validation logic, no
classification/live infra involved. See eval/dev_set_schema.md for the
schema these checks enforce.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_dev_set as vds  # noqa: E402


def make_row(case_id="dev001", language="en", respondent_text="baker",
             gold_isco_code="7512", gold_label_source="single_coder",
             coder_or_adjudicator="AB", major_group="7", difficulty_level="easy",
             notes=""):
    return {
        "case_id": case_id, "language": language, "respondent_text": respondent_text,
        "gold_isco_code": gold_isco_code, "gold_label_source": gold_label_source,
        "coder_or_adjudicator": coder_or_adjudicator, "major_group": major_group,
        "difficulty_level": difficulty_level, "notes": notes,
    }


def make_valid_set(n_en=16, n_ar=15):
    rows = []
    for i in range(n_en):
        rows.append(make_row(case_id=f"dev_en_{i}", language="en",
                              respondent_text=f"english job title number {i}"))
    for i in range(n_ar):
        rows.append(make_row(case_id=f"dev_ar_{i}", language="ar",
                              respondent_text=f"arabic job title number {i}"))
    return rows  # 31 total, clears the 30-case absolute minimum


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

def test_normalize_text_collapses_whitespace_and_lowercases():
    assert vds.normalize_text("  Software   Developer\n") == "software developer"


def test_normalize_text_empty_and_none():
    assert vds.normalize_text("") == ""
    assert vds.normalize_text(None) == ""


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def test_empty_dev_set_is_an_error():
    report = vds.validate_dev_set([], set(), set())
    assert not report.ok
    assert any("empty" in e.lower() for e in report.errors)


def test_missing_required_column_is_an_error():
    rows = [{"case_id": "x", "language": "en"}]  # missing most required columns
    report = vds.validate_dev_set(rows, set(), set())
    assert not report.ok
    assert any("missing required column" in e.lower() for e in report.errors)


# ---------------------------------------------------------------------------
# Leakage: case_id collisions
# ---------------------------------------------------------------------------

def test_case_id_colliding_with_existing_set_is_an_error():
    rows = make_valid_set()
    rows[0]["case_id"] = "17"  # pretend this collides with an existing full130 id
    report = vds.validate_dev_set(rows, other_case_ids={"17"}, other_normalized_texts=set())
    assert not report.ok
    assert any("collides" in e for e in report.errors)


def test_duplicate_case_id_within_dev_set_is_an_error():
    rows = make_valid_set()
    rows[1]["case_id"] = rows[0]["case_id"]
    report = vds.validate_dev_set(rows, set(), set())
    assert not report.ok
    assert any("duplicate case_id" in e.lower() for e in report.errors)


# ---------------------------------------------------------------------------
# Leakage: near-duplicate respondent_text
# ---------------------------------------------------------------------------

def test_respondent_text_duplicating_existing_set_is_an_error():
    rows = make_valid_set()
    rows[0]["respondent_text"] = "Software Developer"
    other_texts = {vds.normalize_text("software   developer")}
    report = vds.validate_dev_set(rows, set(), other_texts)
    assert not report.ok
    assert any("near-duplicate" in e or "duplicates" in e for e in report.errors)


def test_respondent_text_duplicating_within_dev_set_is_an_error():
    rows = make_valid_set()
    rows[1]["respondent_text"] = rows[0]["respondent_text"]
    report = vds.validate_dev_set(rows, set(), set())
    assert not report.ok
    assert any("within the dev set itself" in e for e in report.errors)


def test_blank_respondent_text_is_an_error():
    rows = make_valid_set()
    rows[0]["respondent_text"] = ""
    report = vds.validate_dev_set(rows, set(), set())
    assert not report.ok
    assert any("blank" in e for e in report.errors)


# ---------------------------------------------------------------------------
# Field-level validation
# ---------------------------------------------------------------------------

def test_non_4digit_gold_code_is_an_error():
    rows = make_valid_set()
    rows[0]["gold_isco_code"] = "751"
    report = vds.validate_dev_set(rows, set(), set())
    assert not report.ok
    assert any("not a 4-digit code" in e for e in report.errors)


def test_major_group_mismatch_with_gold_code_is_an_error():
    rows = make_valid_set()
    rows[0]["gold_isco_code"] = "7512"
    rows[0]["major_group"] = "2"
    report = vds.validate_dev_set(rows, set(), set())
    assert not report.ok
    assert any("does not match" in e for e in report.errors)


def test_blank_gold_label_source_is_an_error():
    rows = make_valid_set()
    rows[0]["gold_label_source"] = ""
    report = vds.validate_dev_set(rows, set(), set())
    assert not report.ok
    assert any("gold_label_source is blank" in e for e in report.errors)


def test_unknown_language_is_an_error():
    rows = make_valid_set()
    rows[0]["language"] = "fr"
    report = vds.validate_dev_set(rows, set(), set())
    assert not report.ok


def test_unrecognised_difficulty_is_only_a_warning():
    rows = make_valid_set()
    rows[0]["difficulty_level"] = "extreme"
    report = vds.validate_dev_set(rows, set(), set())
    assert report.ok  # not a hard error
    assert any("difficulty_level" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Coverage thresholds
# ---------------------------------------------------------------------------

def test_below_absolute_minimum_total_is_an_error():
    rows = make_valid_set(n_en=10, n_ar=10)  # 20, below the 30 floor
    report = vds.validate_dev_set(rows, set(), set())
    assert not report.ok
    assert any("absolute minimum" in e for e in report.errors)


def test_between_absolute_and_preferred_is_a_warning_not_error():
    rows = make_valid_set(n_en=20, n_ar=15)  # 35 total: clears 30, below 50
    report = vds.validate_dev_set(rows, set(), set())
    assert report.ok
    assert any("preferred target" in w for w in report.warnings)


def test_meets_preferred_targets_no_coverage_warnings():
    rows = make_valid_set(n_en=35, n_ar=15)  # 50 total, 15 Arabic
    report = vds.validate_dev_set(rows, set(), set())
    assert report.ok
    assert not any("preferred target" in w for w in report.warnings)


def test_zero_english_cases_is_an_error():
    rows = make_valid_set(n_en=0, n_ar=31)
    report = vds.validate_dev_set(rows, set(), set())
    assert not report.ok
    assert any("no english" in e.lower() for e in report.errors)


def test_zero_arabic_cases_is_an_error():
    rows = make_valid_set(n_en=31, n_ar=0)
    report = vds.validate_dev_set(rows, set(), set())
    assert not report.ok
    assert any("no arabic" in e.lower() for e in report.errors)


def test_missing_major_group_coverage_is_a_warning():
    rows = make_valid_set()  # all major_group "7"
    report = vds.validate_dev_set(rows, set(), set(), other_major_groups={"7", "2"})
    assert report.ok
    assert any("zero dev-set coverage" in w for w in report.warnings)


def test_stats_reports_totals_and_language_counts():
    rows = make_valid_set(n_en=16, n_ar=15)
    report = vds.validate_dev_set(rows, set(), set())
    assert report.stats["total_cases"] == 31
    assert report.stats["language_counts"]["en"] == 16
    assert report.stats["language_counts"]["ar"] == 15


# ---------------------------------------------------------------------------
# A genuinely clean set passes with no errors and no warnings
# ---------------------------------------------------------------------------

def test_fully_clean_set_passes_with_no_warnings():
    rows = make_valid_set(n_en=35, n_ar=15)
    report = vds.validate_dev_set(rows, other_case_ids=set(), other_normalized_texts=set())
    assert report.ok
    assert report.warnings == []
