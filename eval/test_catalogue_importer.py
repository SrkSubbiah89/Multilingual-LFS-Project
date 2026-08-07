"""
Tests for eval/catalogue_importer.py -- the official classification-source
and coverage-denominator follow-up to Conference I Reviewer #2 response
Section C.

Uses ONLY small synthetic catalogues (a 2-level "widget_std" fixture
standard: top-level "category" [A-Z], child "item" \\d{3}) -- never the
real, much larger, potentially-copyrighted ISCO/ISIC/ISCED catalogues,
consistent with "do not automatically commit official source PDFs,
restricted data, or copyrighted material into Git."
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import catalogue_importer as ci  # noqa: E402

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_CLEAN_CSV = _FIXTURES_DIR / "synthetic_catalogue_clean.csv"
_ISSUES_CSV = _FIXTURES_DIR / "synthetic_catalogue_with_issues.csv"

WIDGET_STANDARDS_REF = {
    "standards": {
        "widget_std": {
            "standard_name": "Widget Standard",
            "version": "v1",
            "hierarchy_levels": [
                {"name": "category", "code_pattern": "[A-Z]", "parent_level": None, "description": "top-level category"},
                {"name": "item", "code_pattern": r"\d{3}", "parent_level": "category", "description": "child item"},
            ],
        }
    }
}

WIDGET_LEVELS = ci.load_level_specs(WIDGET_STANDARDS_REF, "widget_std")


def load_rows(path: Path) -> list[dict]:
    return ci.load_catalogue_csv(path)


# ---------------------------------------------------------------------------
# load_level_specs
# ---------------------------------------------------------------------------

def test_load_level_specs_returns_ordered_levels():
    levels = ci.load_level_specs(WIDGET_STANDARDS_REF, "widget_std")
    assert [lv.name for lv in levels] == ["category", "item"]
    assert levels[0].parent_level is None
    assert levels[1].parent_level == "category"


def test_load_level_specs_unknown_standard_raises():
    with pytest.raises(KeyError):
        ci.load_level_specs(WIDGET_STANDARDS_REF, "nonexistent_std")


def test_load_level_specs_no_hierarchy_levels_raises():
    ref = {"standards": {"empty_std": {"standard_name": "Empty"}}}
    with pytest.raises(ValueError):
        ci.load_level_specs(ref, "empty_std")


# ---------------------------------------------------------------------------
# validate_catalogue: clean synthetic catalogue
# ---------------------------------------------------------------------------

def test_clean_catalogue_validates_ok():
    rows = load_rows(_CLEAN_CSV)
    result = ci.validate_catalogue("widget_std", WIDGET_LEVELS, rows)
    assert result.ok is True
    assert result.issues == []
    assert result.counts_by_level == {"category": 2, "item": 3}
    assert result.row_count == 5


# ---------------------------------------------------------------------------
# validate_catalogue: each validation rule, in isolation
# ---------------------------------------------------------------------------

def test_malformed_code_detected():
    rows = [{"level": "category", "code": "a", "parent_code": "", "label": "lowercase"}]
    result = ci.validate_catalogue("widget_std", WIDGET_LEVELS, rows)
    assert result.ok is False
    assert result.issues[0].issue_type == "malformed_code"
    assert result.issues[0].code == "a"


def test_duplicate_code_detected():
    rows = [
        {"level": "category", "code": "A", "parent_code": "", "label": "first"},
        {"level": "category", "code": "A", "parent_code": "", "label": "dup"},
    ]
    result = ci.validate_catalogue("widget_std", WIDGET_LEVELS, rows)
    assert result.ok is False
    dup_issues = [i for i in result.issues if i.issue_type == "duplicate_code"]
    assert len(dup_issues) == 1
    assert dup_issues[0].code == "A"


def test_orphan_parent_detected():
    rows = [
        {"level": "category", "code": "A", "parent_code": "", "label": "cat"},
        {"level": "item", "code": "001", "parent_code": "Z", "label": "orphan"},
    ]
    result = ci.validate_catalogue("widget_std", WIDGET_LEVELS, rows)
    assert result.ok is False
    assert result.issues[0].issue_type == "orphan_parent"
    assert result.issues[0].code == "001"


def test_unknown_level_detected():
    rows = [{"level": "widget", "code": "999", "parent_code": "A", "label": "bad level"}]
    result = ci.validate_catalogue("widget_std", WIDGET_LEVELS, rows)
    assert result.ok is False
    assert result.issues[0].issue_type == "unknown_level"


def test_top_level_row_with_parent_code_rejected():
    rows = [{"level": "category", "code": "A", "parent_code": "X", "label": "should not have a parent"}]
    result = ci.validate_catalogue("widget_std", WIDGET_LEVELS, rows)
    assert result.ok is False
    assert result.issues[0].issue_type == "unexpected_parent_for_top_level"


def test_child_row_missing_parent_code_rejected():
    rows = [
        {"level": "category", "code": "A", "parent_code": "", "label": "cat"},
        {"level": "item", "code": "001", "parent_code": "", "label": "no parent given"},
    ]
    result = ci.validate_catalogue("widget_std", WIDGET_LEVELS, rows)
    assert result.ok is False
    assert result.issues[0].issue_type == "missing_parent_code"


def test_synthetic_catalogue_with_issues_flags_all_of_them():
    rows = load_rows(_ISSUES_CSV)
    result = ci.validate_catalogue("widget_std", WIDGET_LEVELS, rows)
    assert result.ok is False
    issue_types = {i.issue_type for i in result.issues}
    assert "malformed_code" in issue_types
    assert "duplicate_code" in issue_types
    assert "orphan_parent" in issue_types
    assert "unknown_level" in issue_types


def test_partial_counts_still_reported_on_failed_import():
    """Diagnostic value: counts_by_level reflects rows that DID pass, even
    though ok=False -- but see write_verified_counts() tests below, this
    partial data must never reach verified_catalogue_counts.yaml."""
    rows = load_rows(_ISSUES_CSV)
    result = ci.validate_catalogue("widget_std", WIDGET_LEVELS, rows)
    assert result.counts_by_level["category"] >= 1  # the one clean "A" row


# ---------------------------------------------------------------------------
# Fail-closed: write_verified_counts is never called with a failed result
# in the normal import_catalogue_file() flow
# ---------------------------------------------------------------------------

def test_import_catalogue_file_clean_writes_verified_counts(tmp_path):
    ref_path = tmp_path / "standards_reference.yaml"
    ref_path.write_text(yaml.safe_dump(WIDGET_STANDARDS_REF), encoding="utf-8")
    out_path = tmp_path / "verified_catalogue_counts.yaml"

    result = ci.import_catalogue_file(
        "widget_std", _CLEAN_CSV, standards_ref_path=ref_path, verified_counts_path=out_path,
    )
    assert result.ok is True
    assert out_path.exists()

    written = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert written["widget_std"]["category"]["count"] == 2
    assert written["widget_std"]["item"]["count"] == 3
    assert written["widget_std"]["category"]["source_file"] == str(_CLEAN_CSV)
    assert len(written["widget_std"]["category"]["source_file_sha256"]) == 64


def test_import_catalogue_file_with_issues_writes_nothing(tmp_path):
    ref_path = tmp_path / "standards_reference.yaml"
    ref_path.write_text(yaml.safe_dump(WIDGET_STANDARDS_REF), encoding="utf-8")
    out_path = tmp_path / "verified_catalogue_counts.yaml"

    result = ci.import_catalogue_file(
        "widget_std", _ISSUES_CSV, standards_ref_path=ref_path, verified_counts_path=out_path,
    )
    assert result.ok is False
    assert not out_path.exists()


def test_import_catalogue_file_dry_run_never_writes_even_when_clean(tmp_path):
    ref_path = tmp_path / "standards_reference.yaml"
    ref_path.write_text(yaml.safe_dump(WIDGET_STANDARDS_REF), encoding="utf-8")
    out_path = tmp_path / "verified_catalogue_counts.yaml"

    result = ci.import_catalogue_file(
        "widget_std", _CLEAN_CSV, standards_ref_path=ref_path, verified_counts_path=out_path, dry_run=True,
    )
    assert result.ok is True
    assert not out_path.exists()


def test_write_verified_counts_preserves_other_standards(tmp_path):
    out_path = tmp_path / "verified_catalogue_counts.yaml"
    out_path.write_text(yaml.safe_dump({"other_std": {"level1": {"count": 99, "source_file": "x", "source_file_sha256": "y", "imported_at_utc": "t", "row_count": 1}}}), encoding="utf-8")

    result = ci.ImportResult(
        standard_key="widget_std", ok=True, counts_by_level={"category": 2},
        issues=[], row_count=2, source_file=str(_CLEAN_CSV),
        source_file_sha256="abc", imported_at_utc="2026-01-01T00:00:00+00:00",
    )
    ci.write_verified_counts(result, out_path)

    merged = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert merged["other_std"]["level1"]["count"] == 99  # untouched
    assert merged["widget_std"]["category"]["count"] == 2


def test_load_verified_counts_missing_file_returns_empty_dict(tmp_path):
    assert ci.load_verified_counts(tmp_path / "does_not_exist.yaml") == {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_clean_catalogue_exits_zero_and_writes_file(tmp_path, monkeypatch, capsys):
    ref_path = tmp_path / "standards_reference.yaml"
    ref_path.write_text(yaml.safe_dump(WIDGET_STANDARDS_REF), encoding="utf-8")
    out_path = tmp_path / "verified_catalogue_counts.yaml"

    monkeypatch.setattr(sys, "argv", [
        "catalogue_importer.py", "--standard", "widget_std", "--catalogue", str(_CLEAN_CSV),
        "--standards-ref", str(ref_path), "--verified-counts-out", str(out_path),
    ])
    ci.main()  # must not raise / must not sys.exit
    assert out_path.exists()
    out = capsys.readouterr().out
    assert "ok=True" in out


def test_cli_catalogue_with_issues_exits_nonzero(tmp_path, monkeypatch):
    ref_path = tmp_path / "standards_reference.yaml"
    ref_path.write_text(yaml.safe_dump(WIDGET_STANDARDS_REF), encoding="utf-8")
    out_path = tmp_path / "verified_catalogue_counts.yaml"

    monkeypatch.setattr(sys, "argv", [
        "catalogue_importer.py", "--standard", "widget_std", "--catalogue", str(_ISSUES_CSV),
        "--standards-ref", str(ref_path), "--verified-counts-out", str(out_path),
    ])
    with pytest.raises(SystemExit) as exc_info:
        ci.main()
    assert exc_info.value.code == 1
    assert not out_path.exists()


def test_cli_missing_catalogue_file_errors(tmp_path, monkeypatch):
    ref_path = tmp_path / "standards_reference.yaml"
    ref_path.write_text(yaml.safe_dump(WIDGET_STANDARDS_REF), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "catalogue_importer.py", "--standard", "widget_std",
        "--catalogue", str(tmp_path / "does_not_exist.csv"),
        "--standards-ref", str(ref_path),
    ])
    with pytest.raises(SystemExit):
        ci.main()
