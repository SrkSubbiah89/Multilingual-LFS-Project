"""
Tests for eval/normalize_ilo_isco08_catalogue.py (Task 20).

Fully hermetic: every workbook fixture is a small, hand-constructed
temporary XLSX built in-process with openpyxl.Workbook(). The real
official ILO workbook and WISCO are never read anywhere in this file.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import openpyxl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import normalize_ilo_isco08_catalogue as norm  # noqa: E402


HEADER = ["Level", "ISCO 08 Code", "Title EN", "Definition", "Tasks include",
          "Included occupations", "Excluded occupations", "Notes"]


def _write_xlsx(path: Path, sheet_name: str, rows: list[list]) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(HEADER)
    for r in rows:
        ws.append(r)
    wb.save(path)
    return path


def _valid_rows() -> list[list]:
    """A small, hand-checkable, depth-first-ordered synthetic hierarchy:
    1 major -> 2 submajor -> 2 minor -> 3 unit (one minor has 2 units,
    the other has 1)."""
    return [
        ["1", "1", "Managers", "", "", "", "", ""],
        ["2", "11", "Chief Executives", "", "", "", "", ""],
        ["3", "111", "Legislators and Senior Officials", "", "", "", "", ""],
        ["4", "1111", "Legislators", "", "", "", "", ""],
        ["4", "1112", "Senior Government Officials", "", "", "", "", ""],
        ["3", "112", "Managing Directors", "", "", "", "", ""],
        ["4", "1120", "Managing Directors and Chief Executives", "", "", "", "", ""],
        ["2", "12", "Administrative Managers", "", "", "", "", ""],
        ["3", "121", "Business Services Managers", "", "", "", "", ""],
        # deliberately no unit under 121, to prove a minor with zero
        # units doesn't break normalization
    ]


_VALID_EXPECTED_COUNTS = {"major": 1, "submajor": 2, "minor": 3, "unit": 3}


def test_successful_parsing_and_hand_checkable_rows(tmp_path):
    xlsx = _write_xlsx(tmp_path / "iso.xlsx", norm.DEFAULT_SHEET_NAME, _valid_rows())
    out_csv = tmp_path / "out.csv"

    report, report_dict = norm.normalize(xlsx, out_csv, expected_counts=_VALID_EXPECTED_COUNTS)

    assert report.n_rows_by_level == _VALID_EXPECTED_COUNTS
    assert report.total_rows == 9
    assert report.source_sha256 == norm._sha256_file(xlsx)
    assert report_dict["normalized_csv_sha256"] == norm._sha256_file(out_csv)

    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 9
    by_code = {r["code"]: r for r in rows}
    assert by_code["1"]["level"] == "major" and by_code["1"]["parent_code"] == ""
    assert by_code["11"]["level"] == "submajor" and by_code["11"]["parent_code"] == "1"
    assert by_code["111"]["level"] == "minor" and by_code["111"]["parent_code"] == "11"
    assert by_code["1111"]["level"] == "unit" and by_code["1111"]["parent_code"] == "111"
    assert by_code["1111"]["label"] == "Legislators"


def test_missing_workbook_rejected(tmp_path):
    with pytest.raises(norm.NormalizationError, match="not found"):
        norm.normalize(tmp_path / "does_not_exist.xlsx", tmp_path / "out.csv", expected_counts=None)


def test_wrong_sheet_name_rejected(tmp_path):
    xlsx = _write_xlsx(tmp_path / "iso.xlsx", "Some Other Sheet", _valid_rows())
    with pytest.raises(norm.NormalizationError, match="not found"):
        norm.normalize(xlsx, tmp_path / "out.csv", expected_counts=None)


def test_missing_required_column_rejected(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = norm.DEFAULT_SHEET_NAME
    ws.append(["Level", "Not The Code Column", "Title EN"])
    ws.append(["1", "1", "Managers"])
    xlsx = tmp_path / "iso.xlsx"
    wb.save(xlsx)
    with pytest.raises(norm.NormalizationError, match="missing required column"):
        norm.normalize(xlsx, tmp_path / "out.csv", expected_counts=None)


def test_malformed_code_rejected(tmp_path):
    rows = _valid_rows()
    rows[0] = ["1", "X", "Managers", "", "", "", "", ""]
    xlsx = _write_xlsx(tmp_path / "iso.xlsx", norm.DEFAULT_SHEET_NAME, rows)
    with pytest.raises(norm.NormalizationError, match="non-numeric code"):
        norm.normalize(xlsx, tmp_path / "out.csv", expected_counts=None)


def test_code_length_mismatch_rejected(tmp_path):
    rows = _valid_rows()
    rows[0] = ["1", "12", "Managers", "", "", "", "", ""]  # 2 digits at major level
    xlsx = _write_xlsx(tmp_path / "iso.xlsx", norm.DEFAULT_SHEET_NAME, rows)
    with pytest.raises(norm.NormalizationError, match="expected 1"):
        norm.normalize(xlsx, tmp_path / "out.csv", expected_counts=None)


def test_blank_code_rejected(tmp_path):
    rows = _valid_rows()
    rows[0] = ["1", "", "Managers", "", "", "", "", ""]
    xlsx = _write_xlsx(tmp_path / "iso.xlsx", norm.DEFAULT_SHEET_NAME, rows)
    with pytest.raises(norm.NormalizationError, match="blank or non-numeric code"):
        norm.normalize(xlsx, tmp_path / "out.csv", expected_counts=None)


def test_malformed_level_rejected(tmp_path):
    rows = _valid_rows()
    rows[0] = ["9", "1", "Managers", "", "", "", "", ""]
    xlsx = _write_xlsx(tmp_path / "iso.xlsx", norm.DEFAULT_SHEET_NAME, rows)
    with pytest.raises(norm.NormalizationError, match="malformed hierarchy level"):
        norm.normalize(xlsx, tmp_path / "out.csv", expected_counts=None)


def test_duplicate_code_rejected(tmp_path):
    rows = _valid_rows()
    rows.append(["4", "1111", "Legislators Duplicate", "", "", "", "", ""])
    xlsx = _write_xlsx(tmp_path / "iso.xlsx", norm.DEFAULT_SHEET_NAME, rows)
    with pytest.raises(norm.NormalizationError, match="duplicate code"):
        norm.normalize(xlsx, tmp_path / "out.csv", expected_counts=None)


def test_orphan_parent_rejected(tmp_path):
    # A unit code whose parent minor group never appeared earlier.
    rows = [
        ["1", "1", "Managers", "", "", "", "", ""],
        ["2", "11", "Chief Executives", "", "", "", "", ""],
        ["4", "1199", "Orphan Unit", "", "", "", "", ""],  # parent "119" never seen
    ]
    xlsx = _write_xlsx(tmp_path / "iso.xlsx", norm.DEFAULT_SHEET_NAME, rows)
    with pytest.raises(norm.NormalizationError, match="has not appeared yet"):
        norm.normalize(xlsx, tmp_path / "out.csv", expected_counts=None)


def test_blank_title_rejected(tmp_path):
    rows = _valid_rows()
    rows[0] = ["1", "1", "", "", "", "", "", ""]
    xlsx = _write_xlsx(tmp_path / "iso.xlsx", norm.DEFAULT_SHEET_NAME, rows)
    with pytest.raises(norm.NormalizationError, match="blank title"):
        norm.normalize(xlsx, tmp_path / "out.csv", expected_counts=None)


def test_expected_count_mismatch_rejected(tmp_path):
    xlsx = _write_xlsx(tmp_path / "iso.xlsx", norm.DEFAULT_SHEET_NAME, _valid_rows())
    wrong_expected = {"major": 1, "submajor": 2, "minor": 3, "unit": 999}
    with pytest.raises(norm.NormalizationError, match="unexpected record count"):
        norm.normalize(xlsx, tmp_path / "out.csv", expected_counts=wrong_expected)


def test_expected_count_match_passes(tmp_path):
    xlsx = _write_xlsx(tmp_path / "iso.xlsx", norm.DEFAULT_SHEET_NAME, _valid_rows())
    out_csv = tmp_path / "out.csv"
    report, _ = norm.normalize(xlsx, out_csv, expected_counts=_VALID_EXPECTED_COUNTS)
    assert report.n_rows_by_level == _VALID_EXPECTED_COUNTS
    assert out_csv.exists()


def test_no_partial_output_written_on_failure(tmp_path):
    rows = _valid_rows()
    rows.append(["4", "1111", "Duplicate", "", "", "", "", ""])
    xlsx = _write_xlsx(tmp_path / "iso.xlsx", norm.DEFAULT_SHEET_NAME, rows)
    out_csv = tmp_path / "out.csv"
    with pytest.raises(norm.NormalizationError):
        norm.normalize(xlsx, out_csv, expected_counts=None)
    assert not out_csv.exists()


def test_blank_rows_skipped_without_error(tmp_path):
    rows = _valid_rows()
    rows.insert(2, [None, None, None, None, None, None, None, None])
    xlsx = _write_xlsx(tmp_path / "iso.xlsx", norm.DEFAULT_SHEET_NAME, rows)
    out_csv = tmp_path / "out.csv"
    report, _ = norm.normalize(xlsx, out_csv, expected_counts=_VALID_EXPECTED_COUNTS)
    assert report.total_rows == 9


def test_output_csv_matches_catalogue_importer_shape(tmp_path):
    """The normalized CSV must be directly loadable by
    eval.catalogue_importer.load_catalogue_csv() and pass its
    validate_catalogue() against the project's real isco08 level specs."""
    import catalogue_importer as ci

    xlsx = _write_xlsx(tmp_path / "iso.xlsx", norm.DEFAULT_SHEET_NAME, _valid_rows())
    out_csv = tmp_path / "out.csv"
    norm.normalize(xlsx, out_csv, expected_counts=_VALID_EXPECTED_COUNTS)

    standards_ref = ci.load_standards_reference()
    levels = ci.load_level_specs(standards_ref, "isco08")
    rows = ci.load_catalogue_csv(out_csv)
    result = ci.validate_catalogue("isco08", levels, rows)

    assert result.ok, result.issues
    assert result.counts_by_level == _VALID_EXPECTED_COUNTS
