"""
Tests for eval/coverage_audit.py -- Section C of the Conference I Reviewer #2
response (coverage-audit tooling, reviewer comment 5: ISIC coverage
disclosure), including the official classification-source and
coverage-denominator follow-up (verified vs. unverified counts,
coverage_percentage).

Uses small fixture code lists for the duplicate/malformed-detection unit
tests (no dependency on the real, much larger embedded tables), plus a
handful of integration tests against the REAL imported tables that assert
the report is well-formed -- never asserting a specific count as ground
truth, since discovering those counts is exactly what this tool is for.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import coverage_audit as ca  # noqa: E402


# ---------------------------------------------------------------------------
# find_duplicates / find_malformed
# ---------------------------------------------------------------------------

def test_find_duplicates_basic():
    assert ca.find_duplicates(["1", "2", "1", "3", "1", "2"]) == ["1", "2"]


def test_find_duplicates_empty():
    assert ca.find_duplicates([]) == []


def test_find_duplicates_no_dupes():
    assert ca.find_duplicates(["1", "2", "3"]) == []


def test_find_malformed_numeric_pattern():
    codes = ["1234", "12", "abcd", "1234"]
    assert ca.find_malformed(codes, r"\d{4}") == ["12", "abcd"]


def test_find_malformed_all_valid():
    assert ca.find_malformed(["A", "B", "C"], r"[A-Z]") == []


# ---------------------------------------------------------------------------
# _level_info honesty: unconfigured/unimported -> null + explicit note,
# never a guess. Unverified counts are sourced but never used for %.
# ---------------------------------------------------------------------------

def test_level_info_missing_standard_returns_nulls():
    name, version, unverified, unverified_note, verified, verified_note = ca._level_info(
        {"standards": {}}, {}, "nonexistent", "level1",
    )
    assert unverified is None
    assert verified is None
    assert "not configured" in unverified_note
    assert "no verified official catalogue" in verified_note
    assert name == "nonexistent"


def test_level_info_reads_unverified_count_and_note():
    ref = {"standards": {"foo": {
        "standard_name": "Foo Standard", "version": "v1",
        "levels": {"bar": {"official_count_unverified": 42, "unverified_source_note": "FooOrg 2020, https://example.org"}},
    }}}
    name, version, unverified, unverified_note, verified, verified_note = ca._level_info(ref, {}, "foo", "bar")
    assert unverified == 42
    assert unverified_note == "FooOrg 2020, https://example.org"
    assert verified is None
    assert name == "Foo Standard"
    assert version == "v1"


def test_level_info_reads_verified_count_from_verified_counts_file():
    ref = {"standards": {"foo": {"standard_name": "Foo", "version": "v1", "levels": {}}}}
    verified_counts = {"foo": {"bar": {
        "count": 7, "source_file": "/tmp/cat.csv", "source_file_sha256": "abc123def456", "imported_at_utc": "2026-01-01T00:00:00+00:00",
    }}}
    _, _, unverified, _, verified, verified_note = ca._level_info(ref, verified_counts, "foo", "bar")
    assert unverified is None
    assert verified == 7
    assert "cat.csv" in verified_note
    assert "abc123def456"[:12] in verified_note


# ---------------------------------------------------------------------------
# _compute_coverage_percentage: ONLY from verified counts, never unverified
# ---------------------------------------------------------------------------

def test_coverage_percentage_null_when_no_verified_count():
    pct, status = ca._compute_coverage_percentage(implemented_count=134, official_count_verified=None)
    assert pct is None
    assert "unavailable" in status


def test_coverage_percentage_computed_when_verified_count_present():
    pct, status = ca._compute_coverage_percentage(implemented_count=134, official_count_verified=419)
    assert pct == pytest.approx(31.98, abs=0.01)
    assert "verified" in status


def test_coverage_percentage_null_when_verified_count_is_zero():
    pct, status = ca._compute_coverage_percentage(implemented_count=5, official_count_verified=0)
    assert pct is None
    assert "not positive" in status


def test_coverage_percentage_never_uses_unverified_count():
    """Even a huge unverified count must never leak into the percentage
    computation -- the function signature itself doesn't accept one, but
    this test pins that contract explicitly."""
    import inspect
    sig = inspect.signature(ca._compute_coverage_percentage)
    assert "official_count_unverified" not in sig.parameters


# ---------------------------------------------------------------------------
# _build_level_report against small fixture catalogues (deliberate duplicate
# + malformed code, per the plan's requirement)
# ---------------------------------------------------------------------------

FIXTURE_REF = {"standards": {"fixture_std": {
    "standard_name": "Fixture Standard", "version": "v0",
    "levels": {"widget": {"official_count_unverified": None, "unverified_source_note": "not sourced yet (fixture)"}},
}}}


def test_build_level_report_counts_unique_implemented(tmp_path):
    codes = ["001", "002", "003", "002"]  # one duplicate
    src = tmp_path / "fixture_source.py"
    src.write_text("# fixture", encoding="utf-8")
    report = ca._build_level_report(codes, r"\d{3}", FIXTURE_REF, {}, "fixture_std", "widget", src)
    assert report.implemented_count == 3
    assert report.duplicate_codes == ["002"]
    assert report.malformed_codes == []
    assert report.official_count_unverified is None
    assert report.official_count_unverified_note == "not sourced yet (fixture)"
    assert report.official_count_verified is None
    assert report.coverage_percentage is None


def test_build_level_report_flags_malformed_codes(tmp_path):
    codes = ["001", "02", "abc", "003"]  # "02" and "abc" don't match \d{3}
    src = tmp_path / "fixture_source.py"
    src.write_text("# fixture", encoding="utf-8")
    report = ca._build_level_report(codes, r"\d{3}", FIXTURE_REF, {}, "fixture_std", "widget", src)
    assert set(report.malformed_codes) == {"02", "abc"}


def test_build_level_report_missing_codes_always_empty(tmp_path):
    """See module docstring: without the full official code list (not just
    a count), missing-code detection would have to guess -- so this is
    always [] regardless of verified/unverified counts."""
    src = tmp_path / "fixture_source.py"
    src.write_text("# fixture", encoding="utf-8")
    ref = {"standards": {"fixture_std": {
        "standard_name": "Fixture", "version": "v0",
        "levels": {"widget": {"official_count_unverified": 10, "unverified_source_note": "sourced"}},
    }}}
    report = ca._build_level_report(["001", "002"], r"\d{3}", ref, {}, "fixture_std", "widget", src)
    assert report.missing_codes == []


def test_build_level_report_captures_source_hash(tmp_path):
    src = tmp_path / "fixture_source.py"
    src.write_text("hello world", encoding="utf-8")
    report = ca._build_level_report(["001"], r"\d{3}", FIXTURE_REF, {}, "fixture_std", "widget", src)
    assert report.source_file == str(src)
    assert len(report.source_file_sha256) == 64  # sha256 hex digest length


def test_build_level_report_missing_source_file_has_empty_hash(tmp_path):
    missing = tmp_path / "does_not_exist.py"
    report = ca._build_level_report(["001"], r"\d{3}", FIXTURE_REF, {}, "fixture_std", "widget", missing)
    assert report.source_file_sha256 == ""


def test_build_level_report_computes_percentage_when_verified_present(tmp_path):
    src = tmp_path / "fixture_source.py"
    src.write_text("# fixture", encoding="utf-8")
    verified_counts = {"fixture_std": {"widget": {
        "count": 4, "source_file": "cat.csv", "source_file_sha256": "deadbeef", "imported_at_utc": "2026-01-01T00:00:00+00:00",
    }}}
    report = ca._build_level_report(["001", "002"], r"\d{3}", FIXTURE_REF, verified_counts, "fixture_std", "widget", src)
    assert report.official_count_verified == 4
    assert report.coverage_percentage == pytest.approx(50.0)
    assert "verified" in report.coverage_percentage_status


# ---------------------------------------------------------------------------
# Integration: real embedded tables produce well-formed reports
# (never asserting a specific count as ground truth)
# ---------------------------------------------------------------------------

def test_audit_isco_returns_four_levels():
    reports = ca.audit_isco()
    level_names = {r.level_name for r in reports}
    assert level_names == {"major", "submajor", "minor", "unit"}
    for r in reports:
        assert r.implemented_count > 0
        assert r.standard_name == "ISCO-08"


def test_audit_isic_dedupes_flat_leaf_table():
    reports = ca.audit_isic()
    level_names = {r.level_name for r in reports}
    assert level_names == {"section", "division", "group", "class"}
    by_level = {r.level_name: r for r in reports}
    # class-level unique count must be <= the flat table's row count (each
    # row IS one class, but division/group/section counts must be smaller
    # since many classes share the same section/division/group)
    assert by_level["section"].implemented_count <= by_level["division"].implemented_count
    assert by_level["division"].implemented_count <= by_level["class"].implemented_count


def test_audit_isced_covers_level_and_field_dimensions():
    reports = ca.audit_isced()
    level_names = {r.level_name for r in reports}
    assert level_names == {"level", "broad", "narrow", "detailed"}
    by_level = {r.level_name: r for r in reports}
    assert by_level["level"].implemented_count == 9  # ISCED 2011 levels 0-8, from the codebase's own table
    assert by_level["broad"].implemented_count <= by_level["narrow"].implemented_count
    assert by_level["narrow"].implemented_count <= by_level["detailed"].implemented_count


def test_audit_all_covers_three_standard_groups():
    result = ca.audit_all()
    assert set(result.keys()) == {"isco08", "isic_rev4", "isced2011_and_iscedf2013"}
    for reports in result.values():
        assert len(reports) > 0


def test_real_standards_reference_yaml_has_unverified_counts_but_no_verified_ones_by_default():
    """As of this session, real sourced-but-unverified counts exist in
    eval/standards_reference.yaml (see STANDARDS_SOURCE_PROVENANCE.md), but
    NO catalogue has ever been imported in this repo, so every real report's
    coverage_percentage must still be null -- this is the central invariant
    the whole verified/unverified split exists to enforce."""
    reports = ca.audit_isic()  # real data, real standards_reference.yaml, but load_verified_counts() finds no file
    by_level = {r.level_name: r for r in reports}
    assert by_level["class"].official_count_unverified == 419
    assert by_level["class"].official_count_verified is None
    assert by_level["class"].coverage_percentage is None


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _sample_reports():
    return [
        ca.CoverageReport(
            standard_name="Fixture", version="v0", level_name="widget",
            official_count_verified=None, official_count_verified_note="no verified official catalogue has been imported for this level",
            official_count_unverified=None, official_count_unverified_note="not sourced yet",
            implemented_count=3, coverage_percentage=None, coverage_percentage_status="unavailable",
            duplicate_codes=["002"], malformed_codes=[],
            missing_codes=[], source_file="fixture.py", source_file_sha256="abc123",
        )
    ]


def test_write_json_round_trips(tmp_path):
    out = tmp_path / "report.json"
    ca.write_json(_sample_reports(), out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["reports"]) == 1
    assert payload["reports"][0]["standard_name"] == "Fixture"
    assert payload["reports"][0]["official_count_verified"] is None
    assert payload["reports"][0]["coverage_percentage"] is None


def test_write_csv_has_expected_columns(tmp_path):
    out = tmp_path / "report.csv"
    ca.write_csv(_sample_reports(), out)
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["standard_name"] == "Fixture"
    assert rows[0]["duplicate_count"] == "1"
    assert rows[0]["coverage_percentage"] == ""


def test_write_markdown_never_shows_a_percentage_without_verified_count(tmp_path):
    out = tmp_path / "report.md"
    ca.write_markdown(_sample_reports(), out)
    text = out.read_text(encoding="utf-8")
    assert "Fixture" in text
    assert "null" in text  # verified/unverified counts and percentage all None -> rendered "null", not fabricated
    assert "%" not in text.split("|")[0]  # header row only; no stray fabricated percentage digit anywhere before a real one exists


def test_write_markdown_shows_percentage_when_present(tmp_path):
    reports = [
        ca.CoverageReport(
            standard_name="Fixture", version="v0", level_name="widget",
            official_count_verified=10, official_count_verified_note="verified via importer",
            official_count_unverified=12, official_count_unverified_note="sourced",
            implemented_count=5, coverage_percentage=50.0, coverage_percentage_status="verified",
            duplicate_codes=[], malformed_codes=[], missing_codes=[],
            source_file="fixture.py", source_file_sha256="abc123",
        )
    ]
    out = tmp_path / "report.md"
    ca.write_markdown(reports, out)
    text = out.read_text(encoding="utf-8")
    assert "50.00%" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_writes_files_for_all_three_standard_groups(tmp_path, monkeypatch):
    out_dir = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", ["coverage_audit.py", "--out", str(out_dir)])
    ca.main()
    files = list(out_dir.iterdir())
    assert any("isco08" in f.name for f in files)
    assert any("isic_rev4" in f.name for f in files)
    assert any("isced2011_and_iscedf2013" in f.name for f in files)
    # each standard group gets json+csv+md
    assert len(files) == 9


def test_cli_real_run_produces_no_fabricated_percentage(tmp_path, monkeypatch):
    """End-to-end: with no verified_catalogue_counts.yaml present, every
    generated report's coverage_percentage must be null."""
    out_dir = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", [
        "coverage_audit.py", "--out", str(out_dir),
        "--verified-counts", str(tmp_path / "does_not_exist.yaml"),
    ])
    ca.main()
    isic_json = next(f for f in out_dir.iterdir() if "isic_rev4" in f.name and f.suffix == ".json")
    payload = json.loads(isic_json.read_text(encoding="utf-8"))
    for report in payload["reports"]:
        assert report["coverage_percentage"] is None
