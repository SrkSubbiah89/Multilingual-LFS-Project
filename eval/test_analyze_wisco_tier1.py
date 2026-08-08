"""
Tests for eval/analyze_wisco_tier1.py (Task 18).

Fully hermetic: every fixture is a small, hand-constructed temporary CSV.
No 18,747-row local WISCO data, network, Qdrant, model, or LLM resource
is read anywhere in this file. The real KNOWN_RISK_IDS list (24 real
WISCO IDs) is monkeypatched down to a tiny subset for the tests that
need to exercise the known-risk gate, since fixtures cannot plausibly
contain the real production ID list.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_wisco_tier1 as awt  # noqa: E402


REFERENCE_FIELDS = ["case_id", "input_text", "input_language", "gold_isco_4digit", "gold_isic", "gold_isced"]

RESULT_FIELDS = [
    "case_id", "gold_isco_4digit", "pred_isco_4digit", "pred_method", "error",
    "reranker_fired", "reranker_model", "estimated_cost_usd", "prompt_tokens",
    "completion_tokens", "evaluation_status", "sre_status",
    "stage1_candidates", "stage2_candidates", "stage3_candidates", "stage4_candidates",
    "stage1_latency_ms", "stage2_latency_ms", "stage3_latency_ms", "stage4_latency_ms",
    "keyword_anchor_retry_used", "stage1_source", "end_to_end_latency_ms",
]

_STAGE_JSON = '[{"code": "1", "label_en": "x", "score": 0.9}]'

# Four hand-picked cases, hand-checkable at every digit level and useful
# for a non-trivial paired McNemar comparison (a=1 both-correct, b=1
# flat-correct/hier-wrong, c=1 flat-wrong/hier-correct, d=1 both-wrong).
_CASES = [
    # case_id, language, gold4, flat_pred4, hier_pred4
    ("c1", "en", "1234", "1234", "1234"),  # both correct
    ("c2", "ar", "1234", "1299", "5678"),  # flat: 1/2 digit only; hier: fully wrong
    ("c3", "hi", "5678", "2234", "5678"),  # flat: fully wrong; hier: fully correct
    ("c4", "tl", "5678", "5678", "2234"),  # flat: fully correct; hier: fully wrong
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _reference_rows():
    return [
        {"case_id": cid, "input_text": f"job {cid}", "input_language": lang,
         "gold_isco_4digit": gold, "gold_isic": "", "gold_isced": ""}
        for cid, lang, gold, _f, _h in _CASES
    ]


def _base_result_row(case_id, gold4, pred4, method) -> dict:
    return {
        "case_id": case_id, "gold_isco_4digit": gold4, "pred_isco_4digit": pred4,
        "pred_method": method, "error": "", "reranker_fired": "False",
        "reranker_model": "", "estimated_cost_usd": "0.0", "prompt_tokens": "0",
        "completion_tokens": "0", "evaluation_status": "measured", "sre_status": "not_applicable",
        "stage1_candidates": _STAGE_JSON, "stage2_candidates": _STAGE_JSON,
        "stage3_candidates": _STAGE_JSON, "stage4_candidates": _STAGE_JSON,
        "stage1_latency_ms": "5.0", "stage2_latency_ms": "5.0",
        "stage3_latency_ms": "5.0", "stage4_latency_ms": "5.0",
        "keyword_anchor_retry_used": "False", "stage1_source": "semantic_retrieval",
        "end_to_end_latency_ms": "20.0",
    }


def _flat_rows():
    rows = []
    for cid, _lang, gold, fpred, _hpred in _CASES:
        r = _base_result_row(cid, gold, fpred, "flat_semantic")
        for c in ("stage1_candidates", "stage2_candidates", "stage3_candidates"):
            r[c] = "null"
        for c in ("stage1_latency_ms", "stage2_latency_ms", "stage3_latency_ms"):
            r[c] = ""
        r["keyword_anchor_retry_used"] = "False"
        r["stage1_source"] = "not_applicable (flat has no stage1)"
        rows.append(r)
    return rows


def _hier_rows():
    rows = []
    for cid, _lang, gold, _fpred, hpred in _CASES:
        rows.append(_base_result_row(cid, gold, hpred, "hierarchical_semantic"))
    return rows


def _write_valid_fixture(tmp_path: Path):
    ref = tmp_path / "reference.csv"
    flat = tmp_path / "flat.csv"
    hier = tmp_path / "hierarchical.csv"
    _write_csv(ref, REFERENCE_FIELDS, _reference_rows())
    _write_csv(flat, RESULT_FIELDS, _flat_rows())
    _write_csv(hier, RESULT_FIELDS, _hier_rows())
    return ref, flat, hier


@pytest.fixture(autouse=True)
def _small_known_risk_list(monkeypatch):
    """All 4 fixture cases are treated as known-risk by default so the
    known-risk gate is exercised on every happy-path test; individual
    tests override this further where they need a different subset."""
    monkeypatch.setattr(awt, "KNOWN_RISK_IDS", ["c1", "c2", "c3", "c4"])
    yield


# ---------------------------------------------------------------------------
# 1. Hand-checkable digit accuracy
# ---------------------------------------------------------------------------

def test_overall_accuracy_hand_checkable(tmp_path):
    ref, flat, hier = _write_valid_fixture(tmp_path)
    ref_rows, _ = awt.validate_reference(ref, awt.sha256_file(ref), 4)
    flat_rows, _ = awt.validate_result_csv(flat, [r["case_id"] for r in ref_rows], 4, "flat", "hierarchical_", 30000)
    hier_rows, _ = awt.validate_result_csv(hier, [r["case_id"] for r in ref_rows], 4, "hierarchical", "hierarchical_", 30000)

    flat_acc = {m.metric_name: m for m in awt.compute_overall_accuracy(flat_rows)}
    hier_acc = {m.metric_name: m for m in awt.compute_overall_accuracy(hier_rows)}

    assert flat_acc["isco_top1_4digit"].correct == 2 and flat_acc["isco_top1_4digit"].n == 4
    assert flat_acc["isco_top1_3digit"].correct == 2
    assert flat_acc["isco_top1_2digit"].correct == 3
    assert flat_acc["isco_top1_1digit"].correct == 3

    assert hier_acc["isco_top1_4digit"].correct == 2 and hier_acc["isco_top1_4digit"].n == 4
    assert hier_acc["isco_top1_3digit"].correct == 2
    assert hier_acc["isco_top1_2digit"].correct == 2
    assert hier_acc["isco_top1_1digit"].correct == 2


# ---------------------------------------------------------------------------
# 2. Wilson interval + exact McNemar wiring
# ---------------------------------------------------------------------------

def test_wilson_and_mcnemar_wiring(tmp_path):
    ref, flat, hier = _write_valid_fixture(tmp_path)
    ref_rows, _ = awt.validate_reference(ref, awt.sha256_file(ref), 4)
    flat_rows, _ = awt.validate_result_csv(flat, [r["case_id"] for r in ref_rows], 4, "flat", "hierarchical_", 30000)
    hier_rows, _ = awt.validate_result_csv(hier, [r["case_id"] for r in ref_rows], 4, "hierarchical", "hierarchical_", 30000)

    m = awt.compute_digit_accuracy(hier_rows, 4, "x")
    expected_lo, expected_hi = awt.wilson_score_interval(2, 4)
    assert m.ci_lo == round(expected_lo, 4)
    assert m.ci_hi == round(expected_hi, 4)

    paired = awt.paired_comparison_4digit(flat_rows, hier_rows)
    assert paired["both_correct"] == 1
    assert paired["flat_correct_hierarchical_wrong_b"] == 1
    assert paired["flat_wrong_hierarchical_correct_c"] == 1
    assert paired["both_wrong"] == 1
    assert paired["n_discordant_pairs"] == 2
    assert paired["accuracy_diff_hierarchical_minus_flat_pp"] == 0.0
    expected_stat, expected_p = awt.mcnemar_test(1, 1)
    assert paired["mcnemar_statistic_min_b_c"] == expected_stat
    assert paired["mcnemar_p_value_two_sided_exact"] == expected_p
    assert expected_p == 1.0


# ---------------------------------------------------------------------------
# 3. Per-language grouping from the reference field
# ---------------------------------------------------------------------------

def test_language_grouping_and_macro_average(tmp_path):
    ref, flat, hier = _write_valid_fixture(tmp_path)
    ref_rows, _ = awt.validate_reference(ref, awt.sha256_file(ref), 4)
    hier_rows, _ = awt.validate_result_csv(hier, [r["case_id"] for r in ref_rows], 4, "hierarchical", "hierarchical_", 30000)
    lang_map = {r["case_id"]: r["input_language"] for r in ref_rows}

    by_lang, macro = awt.compute_language_accuracy(hier_rows, lang_map)
    assert set(by_lang.keys()) == {"en", "ar", "hi", "tl"}
    assert by_lang["en"].accuracy == 1.0   # c1 correct
    assert by_lang["ar"].accuracy == 0.0   # c2 wrong
    assert by_lang["hi"].accuracy == 1.0   # c3 correct
    assert by_lang["tl"].accuracy == 0.0   # c4 wrong
    assert macro == 0.5


# ---------------------------------------------------------------------------
# 4. Key/order mismatch rejection
# ---------------------------------------------------------------------------

def test_order_mismatch_rejected(tmp_path):
    ref, flat, hier = _write_valid_fixture(tmp_path)
    ref_rows, _ = awt.validate_reference(ref, awt.sha256_file(ref), 4)
    reordered = list(reversed(_flat_rows()))
    bad_flat = tmp_path / "flat_reordered.csv"
    _write_csv(bad_flat, RESULT_FIELDS, reordered)
    with pytest.raises(awt.GateFailure, match="order"):
        awt.validate_result_csv(bad_flat, [r["case_id"] for r in ref_rows], 4, "flat", "hierarchical_", 30000)


def test_id_set_mismatch_rejected(tmp_path):
    ref, flat, hier = _write_valid_fixture(tmp_path)
    ref_rows, _ = awt.validate_reference(ref, awt.sha256_file(ref), 4)
    rows = _flat_rows()
    rows[0]["case_id"] = "not-a-reference-id"
    bad_flat = tmp_path / "flat_badid.csv"
    _write_csv(bad_flat, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="missing case_id"):
        awt.validate_result_csv(bad_flat, [r["case_id"] for r in ref_rows], 4, "flat", "hierarchical_", 30000)


# ---------------------------------------------------------------------------
# 5. Duplicate ID rejection
# ---------------------------------------------------------------------------

def test_duplicate_case_id_rejected(tmp_path):
    rows = _flat_rows()
    rows[1]["case_id"] = rows[0]["case_id"]
    bad_flat = tmp_path / "flat_dupe.csv"
    _write_csv(bad_flat, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="duplicate case_id"):
        awt.validate_result_csv(bad_flat, ["c1", "c2", "c3", "c4"], 4, "flat", "hierarchical_", 30000)


def test_reference_duplicate_case_id_rejected(tmp_path):
    rows = _reference_rows()
    rows[1]["case_id"] = rows[0]["case_id"]
    bad_ref = tmp_path / "reference_dupe.csv"
    _write_csv(bad_ref, REFERENCE_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="duplicate case_id"):
        awt.validate_reference(bad_ref, awt.sha256_file(bad_ref), 4)


# ---------------------------------------------------------------------------
# 6. Invalid/missing four-digit code rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", ["", "123", "12345", "abcd"])
def test_invalid_gold_code_rejected(tmp_path, bad_value):
    rows = _flat_rows()
    rows[0]["gold_isco_4digit"] = bad_value
    bad_flat = tmp_path / "flat_badgold.csv"
    _write_csv(bad_flat, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="gold_isco_4digit"):
        awt.validate_result_csv(bad_flat, ["c1", "c2", "c3", "c4"], 4, "flat", "hierarchical_", 30000)


def test_invalid_pred_code_rejected(tmp_path):
    rows = _flat_rows()
    rows[0]["pred_isco_4digit"] = "xx"
    bad_flat = tmp_path / "flat_badpred.csv"
    _write_csv(bad_flat, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="pred_isco_4digit"):
        awt.validate_result_csv(bad_flat, ["c1", "c2", "c3", "c4"], 4, "flat", "hierarchical_", 30000)


# ---------------------------------------------------------------------------
# 7. Nonblank row error rejection
# ---------------------------------------------------------------------------

def test_nonblank_error_rejected(tmp_path):
    rows = _flat_rows()
    rows[0]["error"] = "boom"
    bad_flat = tmp_path / "flat_err.csv"
    _write_csv(bad_flat, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="error field"):
        awt.validate_result_csv(bad_flat, ["c1", "c2", "c3", "c4"], 4, "flat", "hierarchical_", 30000)


# ---------------------------------------------------------------------------
# 8. Flat wrong-method / reranker / cost / token rejection
# ---------------------------------------------------------------------------

def test_flat_wrong_method_rejected(tmp_path):
    rows = _flat_rows()
    rows[0]["pred_method"] = "hierarchical_semantic"
    bad_flat = tmp_path / "flat_wrongmethod.csv"
    _write_csv(bad_flat, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="flat_semantic"):
        awt.validate_result_csv(bad_flat, ["c1", "c2", "c3", "c4"], 4, "flat", "hierarchical_", 30000)


def test_flat_reranker_fired_rejected(tmp_path):
    rows = _flat_rows()
    rows[0]["reranker_fired"] = "True"
    bad_flat = tmp_path / "flat_rerank.csv"
    _write_csv(bad_flat, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="reranker_fired"):
        awt.validate_result_csv(bad_flat, ["c1", "c2", "c3", "c4"], 4, "flat", "hierarchical_", 30000)


def test_flat_nonzero_cost_rejected(tmp_path):
    rows = _flat_rows()
    rows[0]["estimated_cost_usd"] = "0.02"
    bad_flat = tmp_path / "flat_cost.csv"
    _write_csv(bad_flat, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="estimated_cost_usd"):
        awt.validate_result_csv(bad_flat, ["c1", "c2", "c3", "c4"], 4, "flat", "hierarchical_", 30000)


def test_flat_nonzero_tokens_rejected(tmp_path):
    rows = _flat_rows()
    rows[0]["prompt_tokens"] = "12"
    bad_flat = tmp_path / "flat_tokens.csv"
    _write_csv(bad_flat, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="tokens"):
        awt.validate_result_csv(bad_flat, ["c1", "c2", "c3", "c4"], 4, "flat", "hierarchical_", 30000)


# ---------------------------------------------------------------------------
# 9. Hierarchical fallback/wrong-method rejection
# ---------------------------------------------------------------------------

def test_hierarchical_fallback_method_rejected(tmp_path):
    rows = _hier_rows()
    rows[0]["pred_method"] = "flat_semantic"
    bad_hier = tmp_path / "hier_fallback.csv"
    _write_csv(bad_hier, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="hierarchical_"):
        awt.validate_result_csv(bad_hier, ["c1", "c2", "c3", "c4"], 4, "hierarchical", "hierarchical_", 30000)


# ---------------------------------------------------------------------------
# 10. Hierarchical malformed/empty stage-evidence rejection
# ---------------------------------------------------------------------------

def test_hierarchical_empty_stage_evidence_rejected(tmp_path):
    rows = _hier_rows()
    rows[0]["stage2_candidates"] = "[]"
    bad_hier = tmp_path / "hier_emptystage.csv"
    _write_csv(bad_hier, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="stage2_candidates"):
        awt.validate_result_csv(bad_hier, ["c1", "c2", "c3", "c4"], 4, "hierarchical", "hierarchical_", 30000)


def test_hierarchical_malformed_json_stage_evidence_rejected(tmp_path):
    rows = _hier_rows()
    rows[0]["stage3_candidates"] = "{not valid json"
    bad_hier = tmp_path / "hier_badjson.csv"
    _write_csv(bad_hier, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="stage3_candidates"):
        awt.validate_result_csv(bad_hier, ["c1", "c2", "c3", "c4"], 4, "hierarchical", "hierarchical_", 30000)


# ---------------------------------------------------------------------------
# 11. Hierarchical over-cap / non-numeric stage-latency rejection
# ---------------------------------------------------------------------------

def test_hierarchical_latency_over_cap_rejected(tmp_path):
    rows = _hier_rows()
    rows[0]["stage1_latency_ms"] = "30000.01"
    bad_hier = tmp_path / "hier_overcap.csv"
    _write_csv(bad_hier, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="stage1_latency_ms"):
        awt.validate_result_csv(bad_hier, ["c1", "c2", "c3", "c4"], 4, "hierarchical", "hierarchical_", 30000)


def test_hierarchical_latency_non_numeric_rejected(tmp_path):
    rows = _hier_rows()
    rows[0]["stage4_latency_ms"] = "not-a-number"
    bad_hier = tmp_path / "hier_nan.csv"
    _write_csv(bad_hier, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="stage4_latency_ms"):
        awt.validate_result_csv(bad_hier, ["c1", "c2", "c3", "c4"], 4, "hierarchical", "hierarchical_", 30000)


# ---------------------------------------------------------------------------
# 12. Known-risk missing or failed-row rejection
# ---------------------------------------------------------------------------

def test_known_risk_missing_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(awt, "KNOWN_RISK_IDS", ["c1", "does-not-exist"])
    rows = _hier_rows()
    bad_hier = tmp_path / "hier_missingrisk.csv"
    _write_csv(bad_hier, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="missing known-risk"):
        awt.validate_result_csv(bad_hier, ["c1", "c2", "c3", "c4"], 4, "hierarchical", "hierarchical_", 30000)


def test_known_risk_failing_row_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(awt, "KNOWN_RISK_IDS", ["c1"])
    rows = _hier_rows()
    rows[0]["stage1_candidates"] = "null"  # c1 now fails stage-evidence
    bad_hier = tmp_path / "hier_failingrisk.csv"
    _write_csv(bad_hier, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="stage1_candidates"):
        awt.validate_result_csv(bad_hier, ["c1", "c2", "c3", "c4"], 4, "hierarchical", "hierarchical_", 30000)


# ---------------------------------------------------------------------------
# 13. Dry-run/non-measured evaluation_status rejection
# ---------------------------------------------------------------------------

def test_dry_run_evaluation_status_rejected(tmp_path):
    rows = _flat_rows()
    rows[0]["evaluation_status"] = "dry_run_not_measured"
    bad_flat = tmp_path / "flat_dryrun.csv"
    _write_csv(bad_flat, RESULT_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="evaluation_status"):
        awt.validate_result_csv(bad_flat, ["c1", "c2", "c3", "c4"], 4, "flat", "hierarchical_", 30000)


# ---------------------------------------------------------------------------
# reference-only gates: nonblank gold_isic / gold_isced
# ---------------------------------------------------------------------------

def test_reference_nonblank_gold_isic_rejected(tmp_path):
    rows = _reference_rows()
    rows[0]["gold_isic"] = "Q"
    bad_ref = tmp_path / "reference_isic.csv"
    _write_csv(bad_ref, REFERENCE_FIELDS, rows)
    with pytest.raises(awt.GateFailure, match="gold_isic"):
        awt.validate_reference(bad_ref, awt.sha256_file(bad_ref), 4)


def test_reference_sha256_mismatch_rejected(tmp_path):
    ref, _flat, _hier = _write_valid_fixture(tmp_path)
    with pytest.raises(awt.GateFailure, match="sha256 mismatch"):
        awt.validate_reference(ref, "0" * 64, 4)


def test_reference_row_count_mismatch_rejected(tmp_path):
    ref, _flat, _hier = _write_valid_fixture(tmp_path)
    with pytest.raises(awt.GateFailure, match="row count mismatch"):
        awt.validate_reference(ref, awt.sha256_file(ref), 5)


# ---------------------------------------------------------------------------
# 14. Output JSON/Markdown schema presence and deterministic values (full CLI)
# ---------------------------------------------------------------------------

def test_main_writes_expected_outputs(tmp_path, monkeypatch, capsys):
    ref, flat, hier = _write_valid_fixture(tmp_path)
    out_dir = tmp_path / "out"
    argv = [
        "analyze_wisco_tier1.py",
        "--flat-csv", str(flat),
        "--hierarchical-csv", str(hier),
        "--reference-csv", str(ref),
        "--out", str(out_dir),
        "--expected-reference-sha256", awt.sha256_file(ref),
        "--expected-n", "4",
        "--max-stage-latency-ms", "30000",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    awt.main()

    provenance = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    metrics = json.loads((out_dir / "wisco_tier1_metrics.json").read_text(encoding="utf-8"))
    md = (out_dir / "wisco_tier1_metrics.md").read_text(encoding="utf-8")

    assert provenance["inputs"]["reference_csv"]["status"] == "pass"
    assert provenance["inputs"]["flat_csv"]["status"] == "pass"
    assert provenance["inputs"]["hierarchical_csv"]["status"] == "pass"
    assert "WISCO" in provenance["evidence_boundary"]

    assert metrics["n_cases"] == 4
    flat_4digit = next(m for m in metrics["overall_accuracy"]["flat"] if m["metric_name"] == "isco_top1_4digit")
    assert flat_4digit["correct"] == 2
    assert metrics["paired_4digit_comparison"]["n_pairs"] == 4
    assert "not real LFS respondent data" in " ".join(metrics["limitations"])

    assert "# WISCO Tier 1 Controlled Evaluation" in md
    assert "McNemar" in md


def test_main_exits_nonzero_and_writes_nothing_on_gate_failure(tmp_path, monkeypatch, capsys):
    ref, flat, hier = _write_valid_fixture(tmp_path)
    # Corrupt the flat CSV after the valid fixture was written.
    rows = _flat_rows()
    rows[0]["error"] = "boom"
    _write_csv(flat, RESULT_FIELDS, rows)
    out_dir = tmp_path / "out_fail"
    argv = [
        "analyze_wisco_tier1.py",
        "--flat-csv", str(flat),
        "--hierarchical-csv", str(hier),
        "--reference-csv", str(ref),
        "--out", str(out_dir),
        "--expected-reference-sha256", awt.sha256_file(ref),
        "--expected-n", "4",
        "--max-stage-latency-ms", "30000",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        awt.main()
    assert exc_info.value.code != 0
    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# 15. Input hashes recorded and source CSVs unchanged after analysis
# ---------------------------------------------------------------------------

def test_source_csvs_unchanged_after_analysis(tmp_path, monkeypatch):
    ref, flat, hier = _write_valid_fixture(tmp_path)
    ref_sha_before = awt.sha256_file(ref)
    flat_sha_before = awt.sha256_file(flat)
    hier_sha_before = awt.sha256_file(hier)

    out_dir = tmp_path / "out2"
    argv = [
        "analyze_wisco_tier1.py",
        "--flat-csv", str(flat),
        "--hierarchical-csv", str(hier),
        "--reference-csv", str(ref),
        "--out", str(out_dir),
        "--expected-reference-sha256", ref_sha_before,
        "--expected-n", "4",
        "--max-stage-latency-ms", "30000",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    awt.main()

    assert awt.sha256_file(ref) == ref_sha_before
    assert awt.sha256_file(flat) == flat_sha_before
    assert awt.sha256_file(hier) == hier_sha_before

    provenance = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["inputs"]["reference_csv"]["sha256"] == ref_sha_before
    assert provenance["inputs"]["flat_csv"]["sha256"] == flat_sha_before
    assert provenance["inputs"]["hierarchical_csv"]["sha256"] == hier_sha_before
