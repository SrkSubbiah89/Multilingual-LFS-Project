"""
Tests for eval/legacy824/dataset_gate.py (Task 39, scenarios 9 and 10).

Fully hermetic: every fixture is a small, hand-constructed temporary
CSV. No real WISCO data (dev or heldout) is read anywhere in this file.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.legacy824.dataset_gate import (  # noqa: E402
    EXPECTED_DEV_ROW_COUNT,
    EXPECTED_HELDOUT_ROW_COUNT,
    HeldoutAccessRefusedError,
    MalformedDevRowError,
    assert_not_heldout_path,
    load_and_validate_dev_rows,
)

FIELDS = ["case_id", "input_text", "input_language", "gold_isco_4digit", "gold_isic", "gold_isced"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _make_row(i: int, gold: str = "1234") -> dict:
    return {
        "case_id": f"WISCO-DEV-{i}", "input_text": f"job {i}", "input_language": "en",
        "gold_isco_4digit": gold, "gold_isic": "", "gold_isced": "",
    }


def test_refuses_path_containing_heldout_by_name(tmp_path):
    p = tmp_path / "wisco_heldout_export.csv"
    _write_csv(p, [_make_row(i) for i in range(EXPECTED_DEV_ROW_COUNT)])
    with pytest.raises(HeldoutAccessRefusedError):
        assert_not_heldout_path(p)
    with pytest.raises(HeldoutAccessRefusedError):
        load_and_validate_dev_rows(p)


def test_refuses_row_count_matching_heldout_size(tmp_path):
    p = tmp_path / "wisco_dev_but_actually_wrong_count.csv"
    _write_csv(p, [_make_row(i) for i in range(EXPECTED_HELDOUT_ROW_COUNT)])
    with pytest.raises(HeldoutAccessRefusedError):
        load_and_validate_dev_rows(p)


def test_refuses_wrong_dev_row_count(tmp_path):
    p = tmp_path / "wisco_dev_input.csv"
    _write_csv(p, [_make_row(i) for i in range(10)])  # far from 2013
    with pytest.raises(ValueError):
        load_and_validate_dev_rows(p)


def test_accepts_exact_dev_row_count_with_valid_codes(tmp_path):
    p = tmp_path / "wisco_dev_input.csv"
    _write_csv(p, [_make_row(i) for i in range(EXPECTED_DEV_ROW_COUNT)])
    rows = load_and_validate_dev_rows(p)
    assert len(rows) == EXPECTED_DEV_ROW_COUNT


def test_rejects_malformed_gold_code_rather_than_modifying_it(tmp_path):
    p = tmp_path / "wisco_dev_input.csv"
    rows = [_make_row(i) for i in range(EXPECTED_DEV_ROW_COUNT)]
    rows[5]["gold_isco_4digit"] = "12"  # coarse/malformed
    _write_csv(p, rows)
    with pytest.raises(MalformedDevRowError) as exc_info:
        load_and_validate_dev_rows(p)
    assert "WISCO-DEV-5" in str(exc_info.value)


def test_rejects_blank_gold_code(tmp_path):
    p = tmp_path / "wisco_dev_input.csv"
    rows = [_make_row(i) for i in range(EXPECTED_DEV_ROW_COUNT)]
    rows[0]["gold_isco_4digit"] = ""
    _write_csv(p, rows)
    with pytest.raises(MalformedDevRowError):
        load_and_validate_dev_rows(p)


def test_rejects_duplicate_case_ids(tmp_path):
    p = tmp_path / "wisco_dev_input.csv"
    rows = [_make_row(i) for i in range(EXPECTED_DEV_ROW_COUNT)]
    rows[1]["case_id"] = rows[0]["case_id"]
    _write_csv(p, rows)
    with pytest.raises(ValueError):
        load_and_validate_dev_rows(p)
