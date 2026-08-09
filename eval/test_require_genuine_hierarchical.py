"""
Tests for eval/run_eval.py's Task 13 --require-genuine-hierarchical
strict evaluation-only guard and check_strict_hierarchical().

Fully hermetic: ISCOClassifier/ISICClassifier/ISCEDClassifier/
SemanticRelationEngine are always monkeypatched to fakes/mocks that fail
loudly if constructed when they shouldn't be. No Qdrant, SentenceTransformer
download, Ollama, paid LLM/API call, benchmark run, or real dataset access
occurs anywhere in this file.
"""

from __future__ import annotations

import csv as csv_module
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_eval  # noqa: E402


def _write_isco_only_test_set(tmp_path: Path) -> Path:
    test_set = tmp_path / "test.csv"
    test_set.write_text(
        "case_id,input_text,input_language,gold_isco_4digit\n"
        "c1,software developer,en,2512\n",
        encoding="utf-8",
    )
    return test_set


def _fake_isco_instance(pred_method="hierarchical_semantic"):
    m = MagicMock()
    m.reranker_model_resolved = "none (reranking disabled)"
    stage = '[{"code": "2512", "label_en": "Software Developers", "score": 0.9}]'

    def _classify(job_title, language, top_k, use_llm, trace, max_stage_latency_ms=None):
        if trace is not None:
            trace["stage1"] = [{"code": "2", "label_en": "x", "score": 0.9}]
            trace["stage2"] = [{"code": "25", "label_en": "x", "score": 0.9}]
            trace["stage3"] = [{"code": "251", "label_en": "x", "score": 0.9}]
            trace["stage4"] = [{"code": "2512", "label_en": "x", "score": 0.9}]
            trace["stage1_latency_ms"] = 5.0
            trace["stage2_latency_ms"] = 5.0
            trace["stage3_latency_ms"] = 5.0
            trace["stage4_latency_ms"] = 5.0
        return SimpleNamespace(
            primary=SimpleNamespace(code="2512", title_en="Software Developers", title_ar="", confidence=0.9),
            method=pred_method, hitl_required=False, reasoning="test",
        )

    m.classify.side_effect = _classify
    return m


# ---------------------------------------------------------------------------
# 7. CLI validation
# ---------------------------------------------------------------------------

def test_cli_rejects_require_genuine_hierarchical_with_flat_system(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "flat",
        "--use-llm-reranker", "off", "--require-genuine-hierarchical",
    ])
    with pytest.raises(SystemExit):
        run_eval.main()


def test_cli_rejects_require_genuine_hierarchical_with_bm25_system(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "bm25",
        "--require-genuine-hierarchical",
    ])
    with pytest.raises(SystemExit):
        run_eval.main()


def test_cli_accepts_require_genuine_hierarchical_with_hierarchical_system(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=lambda **kw: _fake_isco_instance()))
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError("must not construct")))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError("must not construct")))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError("must not construct")))
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--require-genuine-hierarchical",
        "--output-dir", str(out_dir),
    ])
    run_eval.main()  # must not raise -- valid combination, genuine result


def test_cli_rejects_non_positive_max_stage_latency_ms(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--require-genuine-hierarchical",
        "--max-stage-latency-ms", "0",
    ])
    with pytest.raises(SystemExit):
        run_eval.main()


def test_cli_rejects_negative_max_stage_latency_ms(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--require-genuine-hierarchical",
        "--max-stage-latency-ms", "-5",
    ])
    with pytest.raises(SystemExit):
        run_eval.main()


def test_max_stage_latency_ms_has_no_effect_without_strict_flag(tmp_path, monkeypatch):
    """Opt-in: --max-stage-latency-ms alone (no --require-genuine-hierarchical)
    must not change ordinary run behaviour at all."""
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=lambda **kw: _fake_isco_instance()))
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError))
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--max-stage-latency-ms", "1",
        "--output-dir", str(out_dir),
    ])
    run_eval.main()  # must not raise -- strict flag was never passed
    assert list(out_dir.glob("*.csv"))


# ---------------------------------------------------------------------------
# check_strict_hierarchical() unit behaviour
# ---------------------------------------------------------------------------

def _make_result(pred_method="hierarchical_semantic", stage_candidates=None, stage_latencies=None):
    stage_candidates = stage_candidates if stage_candidates is not None else {
        1: '[{"code": "2", "label_en": "x", "score": 0.8}]',
        2: '[{"code": "25", "label_en": "x", "score": 0.8}]',
        3: '[{"code": "251", "label_en": "x", "score": 0.8}]',
        4: '[{"code": "2512", "label_en": "x", "score": 0.8}]',
    }
    stage_latencies = stage_latencies if stage_latencies is not None else {1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0}
    r = run_eval.CaseResult(case_id="c1", input_text="x", input_language="en")
    r.pred_method = pred_method
    for i in (1, 2, 3, 4):
        setattr(r, f"stage{i}_candidates", stage_candidates.get(i, "null"))
        setattr(r, f"stage{i}_latency_ms", stage_latencies.get(i))
    return r


# ---------------------------------------------------------------------------
# 8. Strict guard fails on fallback / missing stage evidence / exceeded latency
# ---------------------------------------------------------------------------

def test_check_strict_hierarchical_fails_on_flat_method():
    r = _make_result(pred_method="flat_semantic")
    violation = run_eval.check_strict_hierarchical(r, max_stage_latency_ms=None)
    assert violation is not None
    assert "not a genuine hierarchical method" in violation


def test_check_strict_hierarchical_fails_on_missing_stage_evidence():
    r = _make_result(stage_candidates={
        1: "null",
        2: '[{"code": "25", "label_en": "x", "score": 0.8}]',
        3: '[{"code": "251", "label_en": "x", "score": 0.8}]',
        4: '[{"code": "2512", "label_en": "x", "score": 0.8}]',
    })
    violation = run_eval.check_strict_hierarchical(r, max_stage_latency_ms=None)
    assert violation is not None
    assert "stage1_candidates" in violation


def test_check_strict_hierarchical_fails_on_empty_list_stage_evidence():
    r = _make_result(stage_candidates={
        1: '[{"code": "2", "label_en": "x", "score": 0.8}]',
        2: "[]",
        3: '[{"code": "251", "label_en": "x", "score": 0.8}]',
        4: '[{"code": "2512", "label_en": "x", "score": 0.8}]',
    })
    violation = run_eval.check_strict_hierarchical(r, max_stage_latency_ms=None)
    assert violation is not None
    assert "stage2_candidates" in violation


def test_check_strict_hierarchical_fails_on_exceeded_latency_threshold():
    r = _make_result(stage_latencies={1: 10.0, 2: 10.0, 3: 5_000_000.0, 4: 10.0})
    violation = run_eval.check_strict_hierarchical(r, max_stage_latency_ms=1000.0)
    assert violation is not None
    assert "stage3_latency_ms" in violation
    assert "5000000" in violation.replace(".0", "")


def test_check_strict_hierarchical_ignores_latency_when_threshold_not_supplied():
    r = _make_result(stage_latencies={1: 10.0, 2: 10.0, 3: 5_000_000.0, 4: 10.0})
    violation = run_eval.check_strict_hierarchical(r, max_stage_latency_ms=None)
    assert violation is None  # genuine, complete hierarchical evidence -- no threshold configured


# ---------------------------------------------------------------------------
# 9. A valid genuine four-stage result passes the strict guard
# ---------------------------------------------------------------------------

def test_check_strict_hierarchical_passes_genuine_complete_result():
    r = _make_result()
    assert run_eval.check_strict_hierarchical(r, max_stage_latency_ms=None) is None


def test_check_strict_hierarchical_passes_with_latency_under_threshold():
    r = _make_result(stage_latencies={1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0})
    assert run_eval.check_strict_hierarchical(r, max_stage_latency_ms=1000.0) is None


def test_check_strict_hierarchical_passes_hierarchical_llm_method():
    r = _make_result(pred_method="hierarchical_llm")
    assert run_eval.check_strict_hierarchical(r, max_stage_latency_ms=None) is None


# ---------------------------------------------------------------------------
# 8 (integration). Strict guard aborts the run and writes no CSV
# ---------------------------------------------------------------------------

def test_strict_guard_aborts_run_and_writes_no_csv_on_fallback(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        run_eval, "ISCOClassifier",
        MagicMock(side_effect=lambda **kw: _fake_isco_instance(pred_method="flat_semantic")),
    )
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError))

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--require-genuine-hierarchical",
        "--output-dir", str(out_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        run_eval.main()
    assert exc_info.value.code != 0
    assert not list(out_dir.glob("*.csv"))


def test_strict_guard_aborts_run_on_exceeded_stage_latency(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"

    slow_clf = MagicMock()
    slow_clf.reranker_model_resolved = "none (reranking disabled)"

    def _classify(job_title, language, top_k, use_llm, trace, max_stage_latency_ms=None):
        if trace is not None:
            trace["stage1"] = [{"code": "2", "label_en": "x", "score": 0.9}]
            trace["stage2"] = [{"code": "25", "label_en": "x", "score": 0.9}]
            trace["stage3"] = [{"code": "251", "label_en": "x", "score": 0.9}]
            trace["stage4"] = [{"code": "2512", "label_en": "x", "score": 0.9}]
            trace["stage1_latency_ms"] = 5.0
            trace["stage2_latency_ms"] = 5.0
            trace["stage3_latency_ms"] = 999_999.0  # simulated stall
            trace["stage4_latency_ms"] = 5.0
        return SimpleNamespace(
            primary=SimpleNamespace(code="2512", title_en="x", title_ar="", confidence=0.9),
            method="hierarchical_semantic", hitl_required=False, reasoning="test",
        )

    slow_clf.classify.side_effect = _classify
    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=lambda **kw: slow_clf))
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError))

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--require-genuine-hierarchical",
        "--max-stage-latency-ms", "1000",
        "--output-dir", str(out_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        run_eval.main()
    assert exc_info.value.code != 0
    assert not list(out_dir.glob("*.csv"))


def test_strict_guard_passes_genuine_four_stage_result(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=lambda **kw: _fake_isco_instance()))
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError))

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--require-genuine-hierarchical",
        "--max-stage-latency-ms", "1000",
        "--output-dir", str(out_dir),
    ])
    run_eval.main()  # must not raise
    csvs = list(out_dir.glob("*.csv"))
    assert len(csvs) == 1
    with open(csvs[0], newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))
    assert rows[0]["pred_method"] == "hierarchical_semantic"


# ---------------------------------------------------------------------------
# 10. Ordinary runs (no strict flags) preserve current fallback labelling
# and --dry-run behaviour
# ---------------------------------------------------------------------------

def test_ordinary_run_without_strict_flag_preserves_fallback_labelling(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        run_eval, "ISCOClassifier",
        MagicMock(side_effect=lambda **kw: _fake_isco_instance(pred_method="flat_semantic")),
    )
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError))

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--output-dir", str(out_dir),
    ])
    run_eval.main()  # must NOT raise -- no strict flag was passed

    csvs = list(out_dir.glob("*.csv"))
    assert len(csvs) == 1
    with open(csvs[0], newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))
    assert rows[0]["pred_method"] == "flat_semantic"  # explicit fallback label, never hidden


def test_dry_run_unaffected_by_require_genuine_hierarchical(tmp_path, monkeypatch):
    """--dry-run's documented semantics (no classifier constructed at all)
    must be completely unchanged even when --require-genuine-hierarchical
    is also passed."""
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=AssertionError("must not construct in dry-run")))
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--dry-run", "--require-genuine-hierarchical", "--output-dir", str(out_dir),
    ])
    run_eval.main()  # must not raise -- dry-run's documented semantics unchanged
    csvs = list(out_dir.glob("*.csv"))
    assert len(csvs) == 1
    with open(csvs[0], newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))
    assert rows[0]["evaluation_status"] == "dry_run_not_measured"
