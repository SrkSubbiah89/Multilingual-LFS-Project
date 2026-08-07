"""
Tests for eval/analyze.py -- Section D of the Conference I Reviewer #2
response (accuracy analysis: ISCO/ISIC/ISCED, Wilson CI, McNemar's test).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze as an  # noqa: E402


# ---------------------------------------------------------------------------
# wilson_score_interval -- textbook values
# ---------------------------------------------------------------------------

def test_wilson_interval_50_50_split():
    lo, hi = an.wilson_score_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)


def test_wilson_interval_all_success():
    lo, hi = an.wilson_score_interval(10, 10)
    assert 0.6 < lo < 1.0
    assert hi == pytest.approx(1.0, abs=1e-6)


def test_wilson_interval_all_failure():
    lo, hi = an.wilson_score_interval(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-6)
    assert 0.0 < hi < 0.4


def test_wilson_interval_zero_n_raises():
    with pytest.raises(ValueError):
        an.wilson_score_interval(0, 0)


def test_wilson_interval_successes_exceeds_n_raises():
    with pytest.raises(ValueError):
        an.wilson_score_interval(11, 10)


def test_wilson_interval_narrows_with_larger_n():
    lo1, hi1 = an.wilson_score_interval(50, 100)
    lo2, hi2 = an.wilson_score_interval(500, 1000)
    assert (hi2 - lo2) < (hi1 - lo1)


# ---------------------------------------------------------------------------
# mcnemar_test -- hand-computed 2x2 table
# ---------------------------------------------------------------------------

def test_mcnemar_symmetric_split_p_is_one():
    stat, p = an.mcnemar_test(5, 5)
    assert stat == 5.0
    assert p == pytest.approx(1.0)


def test_mcnemar_highly_asymmetric_split_is_significant():
    stat, p = an.mcnemar_test(1, 15)
    assert stat == 1.0
    assert p < 0.05


def test_mcnemar_zero_discordant_pairs():
    stat, p = an.mcnemar_test(0, 0)
    assert stat == 0.0
    assert p == 1.0


def test_mcnemar_negative_raises():
    with pytest.raises(ValueError):
        an.mcnemar_test(-1, 5)


def test_mcnemar_known_small_case_b2_10_c4():
    # b=10, c=4 -> n=14, k=4; verify against a hand-computed exact binomial
    # two-sided p-value for comparison purposes (regression pin, not an
    # external ground truth citation).
    stat, p = an.mcnemar_test(10, 4)
    assert stat == 4.0
    assert 0.0 < p < 1.0


# ---------------------------------------------------------------------------
# _exact_match_metric / isco_accuracy / isic_accuracy / isced_accuracy
# ---------------------------------------------------------------------------

def make_row(**overrides):
    row = {
        "gold_isco_1digit": "2", "pred_isco_1digit": "2",
        "gold_isco_2digit": "25", "pred_isco_2digit": "25",
        "gold_isco_3digit": "251", "pred_isco_3digit": "251",
        "gold_isco_4digit": "2512", "pred_isco_4digit": "2512",
        "gold_rank_in_pool": "1",
        "gold_isic": "J", "pred_isic_section": "J",
        "gold_isced": "6", "pred_isced_level": "6",
    }
    row.update(overrides)
    return row


def test_isco_top1_all_correct():
    rows = [make_row() for _ in range(5)]
    metrics = {m.metric_name: m for m in an.isco_accuracy(rows)}
    assert metrics["isco_top1_4digit"].accuracy == 1.0
    assert metrics["isco_top1_4digit"].n == 5
    assert metrics["isco_top1_4digit"].status == "measured"


def test_isco_top1_partial_correct():
    rows = [make_row(pred_isco_4digit="2512"), make_row(pred_isco_4digit="9999")]
    metrics = {m.metric_name: m for m in an.isco_accuracy(rows)}
    assert metrics["isco_top1_4digit"].accuracy == 0.5
    assert metrics["isco_top1_4digit"].successes == 1


def test_isco_top3_prererank_pool_uses_gold_rank_column():
    rows = [make_row(gold_rank_in_pool="1"), make_row(gold_rank_in_pool="5")]
    metrics = {m.metric_name: m for m in an.isco_accuracy(rows)}
    m = metrics["isco_top3_prererank_pool"]
    assert m.successes == 1
    assert m.n == 2


def test_isco_top3_pool_not_measured_when_column_missing():
    rows = [{k: v for k, v in make_row().items() if k != "gold_rank_in_pool"}]
    metrics = {m.metric_name: m for m in an.isco_accuracy(rows)}
    m = metrics["isco_top3_prererank_pool"]
    assert m.status == "not_measured"
    assert "gold_rank_in_pool" in m.reason


def test_isic_section_measured_but_deeper_levels_not_measured():
    """Real test-set CSVs today only carry gold_isic at the section level --
    division/group/class gold columns don't exist yet, so those three
    metrics must honestly report not_measured, not 0% or a crash."""
    rows = [make_row()]
    metrics = {m.metric_name: m for m in an.isic_accuracy(rows)}
    assert metrics["isic_section"].status == "measured"
    assert metrics["isic_division"].status == "not_measured"
    assert metrics["isic_group"].status == "not_measured"
    assert metrics["isic_class"].status == "not_measured"


def test_isced_level_measured_but_field_dimensions_not_measured():
    rows = [make_row()]
    metrics = {m.metric_name: m for m in an.isced_accuracy(rows)}
    assert metrics["isced_level"].status == "measured"
    assert metrics["iscedf_broad"].status == "not_measured"


def test_exact_match_metric_empty_gold_column_not_measured():
    rows = [make_row(gold_isic="")]
    m = an._exact_match_metric(rows, "gold_isic", "pred_isic_section", "isic_section")
    assert m.status == "not_measured"
    assert "non-empty" in m.reason


def test_exact_match_metric_missing_column_not_measured():
    m = an._exact_match_metric([{"a": "1"}], "gold_x", "pred_x", "x_metric")
    assert m.status == "not_measured"


def test_exact_match_metric_includes_wilson_ci_when_measured():
    rows = [make_row() for _ in range(10)]
    m = an._exact_match_metric(rows, "gold_isco_4digit", "pred_isco_4digit", "test")
    assert m.ci_lo is not None
    assert m.ci_hi is not None
    assert 0.0 <= m.ci_lo <= m.accuracy <= m.ci_hi <= 1.0


# ---------------------------------------------------------------------------
# load_case_rows / analyze_csv (file I/O)
# ---------------------------------------------------------------------------

def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_load_case_rows(tmp_path):
    csv_path = tmp_path / "cases.csv"
    _write_csv(csv_path, [make_row()])
    rows = an.load_case_rows(csv_path)
    assert len(rows) == 1
    assert rows[0]["gold_isco_4digit"] == "2512"


def test_analyze_csv_returns_all_three_groups(tmp_path):
    csv_path = tmp_path / "cases.csv"
    _write_csv(csv_path, [make_row(), make_row(pred_isco_4digit="0000")])
    results = an.analyze_csv(csv_path)
    assert set(results.keys()) == {"isco", "isic", "isced"}


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def test_write_json_and_markdown(tmp_path):
    rows = [make_row()]
    results = {"isco": an.isco_accuracy(rows)}
    json_path = tmp_path / "out.json"
    md_path = tmp_path / "out.md"
    an.write_json(results, json_path)
    an.write_markdown(results, md_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "isco" in payload
    md_text = md_path.read_text(encoding="utf-8")
    assert "isco_top1_4digit" in md_text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_writes_output_files(tmp_path, monkeypatch):
    csv_path = tmp_path / "cases.csv"
    _write_csv(csv_path, [make_row()])
    out_dir = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", ["analyze.py", "--case-csv", str(csv_path), "--out", str(out_dir)])
    an.main()
    assert (out_dir / "cases_accuracy.json").exists()
    assert (out_dir / "cases_accuracy.md").exists()
