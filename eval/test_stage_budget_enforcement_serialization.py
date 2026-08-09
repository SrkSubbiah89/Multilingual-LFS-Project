"""
Tests for Task 31's stage-budget threading through eval/run_eval.py and
its interaction with the strict --require-genuine-hierarchical guard.

Fully hermetic: ISCOClassifier is always monkeypatched to a fake -- same
pattern as eval/test_qdrant_retry_resilience_serialization.py. No Qdrant,
embedding model, LLM, or WISCO artifact is used anywhere in this file.
"""

from __future__ import annotations

import csv as csv_module
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_eval  # noqa: E402

_OFFICIAL_PROFILE = "official_ilo2021_v1"


def _write_isco_only_test_set(tmp_path: Path) -> Path:
    test_set = tmp_path / "test.csv"
    test_set.write_text(
        "case_id,input_text,input_language,gold_isco_4digit\n"
        "c1,unclassifiable job title,ar,8131\n",
        encoding="utf-8",
    )
    return test_set


def _install_fake_classifier(monkeypatch, classify_fn):
    m = MagicMock()
    m.reranker_model_resolved = "none (reranking disabled)"
    m.classify.side_effect = classify_fn
    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=lambda **kw: m))
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError("must not construct ISIC")))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError("must not construct ISCED")))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError("must not construct SRE")))
    return m


# ---------------------------------------------------------------------------
# Budget threading: only active with --system hierarchical AND
# --require-genuine-hierarchical together (matching --max-stage-latency-ms's
# pre-existing documented relationship)
# ---------------------------------------------------------------------------

def test_strict_hierarchical_run_threads_max_stage_latency_ms_into_classify(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"
    captured = {}

    def _classify(job_title, language, top_k, use_llm, trace, max_stage_latency_ms=None):
        captured["max_stage_latency_ms"] = max_stage_latency_ms
        if trace is not None:
            for i in range(1, 5):
                trace[f"stage{i}"] = [{"code": "8131", "label_en": "x", "score": 0.9}]
                trace[f"stage{i}_latency_ms"] = 5.0
        return SimpleNamespace(
            primary=SimpleNamespace(code="8131", title_en="x", title_ar="", confidence=0.9),
            method=f"hierarchical_isco08_{_OFFICIAL_PROFILE}", hitl_required=False, reasoning="test",
        )

    _install_fake_classifier(monkeypatch, _classify)
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--isco-catalogue-profile", _OFFICIAL_PROFILE,
        "--require-genuine-hierarchical", "--max-stage-latency-ms", "30000",
        "--output-dir", str(out_dir),
    ])
    run_eval.main()
    assert captured["max_stage_latency_ms"] == 30000.0


def test_plain_hierarchical_run_without_strict_flag_passes_no_budget(tmp_path, monkeypatch):
    """--max-stage-latency-ms without --require-genuine-hierarchical must
    never impose a live budget -- matching its pre-existing documented
    relationship (the post-hoc guard also never runs in this case)."""
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"
    captured = {}

    def _classify(job_title, language, top_k, use_llm, trace, max_stage_latency_ms=None):
        captured["max_stage_latency_ms"] = max_stage_latency_ms
        return SimpleNamespace(
            primary=SimpleNamespace(code="8131", title_en="x", title_ar="", confidence=0.9),
            method=f"hierarchical_isco08_{_OFFICIAL_PROFILE}", hitl_required=False, reasoning="test",
        )

    _install_fake_classifier(monkeypatch, _classify)
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--isco-catalogue-profile", _OFFICIAL_PROFILE,
        "--output-dir", str(out_dir),
    ])
    run_eval.main()
    assert captured["max_stage_latency_ms"] is None


def test_flat_run_never_receives_a_stage_budget(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"
    captured = {}

    def _classify(job_title, language, top_k, use_llm, trace, max_stage_latency_ms=None):
        captured["max_stage_latency_ms"] = max_stage_latency_ms
        if trace is not None:
            trace["flat_query_outcome"] = "success"
            trace["flat_query_duration_ms"] = 12.3
            trace["flat_query_attempts"] = 1
            trace["flat_query_attempt_durations_ms"] = [12.3]
        return SimpleNamespace(
            primary=SimpleNamespace(code="2512", title_en="x", title_ar="", confidence=0.9),
            method=f"flat_isco08_{_OFFICIAL_PROFILE}", hitl_required=False, reasoning="test",
        )

    _install_fake_classifier(monkeypatch, _classify)
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "flat",
        "--use-llm-reranker", "off", "--isco-catalogue-profile", _OFFICIAL_PROFILE,
        "--output-dir", str(out_dir),
    ])
    run_eval.main()
    assert captured["max_stage_latency_ms"] is None


# ---------------------------------------------------------------------------
# 8. --require-genuine-hierarchical rejects budget exhaustion; no CSV written
# ---------------------------------------------------------------------------

def test_strict_guard_rejects_stage_budget_exhaustion_no_csv_written(tmp_path, monkeypatch, capsys):
    """Simulates exactly what a genuine stage-budget exhaustion produces
    downstream: the existing (unmodified) flat-fallback path fires, so
    pred_method becomes the flat/fallback prefix -- the strict guard
    rejects it exactly as it already does for any other fallback cause,
    and main() aborts with no output CSV."""
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"

    def _classify(job_title, language, top_k, use_llm, trace, max_stage_latency_ms=None):
        if trace is not None:
            trace["stage1_query_telemetry"] = {
                "queries": 1, "any_retry": False, "any_exception": False,
                "max_attempts_used": 0, "exception_types": [],
                "stage_budget_exhausted": True,
                "configured_query_timeout_seconds": 8.0,
                "initial_stage_budget_ms": 30000.0,
                "queries_detail": [{"outcome": "stage_budget_exhausted", "attempts": 0,
                                     "attempt_durations_ms": [], "remaining_stage_budget_ms_at_entry": 0.0}],
            }
        return SimpleNamespace(
            primary=SimpleNamespace(code="", title_en="Unknown", title_ar="", confidence=0.0),
            method=f"unavailable_isco08_{_OFFICIAL_PROFILE}", hitl_required=True,
            reasoning="No candidates returned by the vector store.",
        )

    _install_fake_classifier(monkeypatch, _classify)
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--isco-catalogue-profile", _OFFICIAL_PROFILE,
        "--require-genuine-hierarchical", "--max-stage-latency-ms", "30000",
        "--output-dir", str(out_dir),
    ])
    with pytest.raises(SystemExit) as exc_info:
        run_eval.main()
    assert exc_info.value.code != 0
    assert not any(out_dir.glob("*.csv"))
    captured = capsys.readouterr()
    assert "STRICT GUARD FAILURE" in captured.err
    assert "not a genuine hierarchical method" in captured.err


# ---------------------------------------------------------------------------
# 13. hier_stage_query_telemetry serializes the new Task 31 fields cleanly
# ---------------------------------------------------------------------------

def test_hier_stage_query_telemetry_serializes_task31_fields(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"

    def _classify(job_title, language, top_k, use_llm, trace, max_stage_latency_ms=None):
        if trace is not None:
            for i in range(1, 5):
                trace[f"stage{i}"] = [{"code": "8131", "label_en": "x", "score": 0.9}]
                trace[f"stage{i}_latency_ms"] = 5.0
            trace["stage4_query_telemetry"] = {
                "queries": 3, "any_retry": True, "any_exception": False,
                "max_attempts_used": 2, "exception_types": [],
                "stage_budget_exhausted": False,
                "configured_query_timeout_seconds": 8.0,
                "initial_stage_budget_ms": 30000.0,
                "queries_detail": [
                    {"outcome": "success", "attempts": 1, "attempt_durations_ms": [10.0],
                     "remaining_stage_budget_ms_at_entry": 29990.0},
                    {"outcome": "success_after_retry", "attempts": 2, "attempt_durations_ms": [8000.0, 12.0],
                     "remaining_stage_budget_ms_at_entry": 29970.0},
                    {"outcome": "success", "attempts": 1, "attempt_durations_ms": [9.0],
                     "remaining_stage_budget_ms_at_entry": 21950.0},
                ],
            }
        return SimpleNamespace(
            primary=SimpleNamespace(code="8131", title_en="x", title_ar="", confidence=0.9),
            method=f"hierarchical_isco08_{_OFFICIAL_PROFILE}", hitl_required=False, reasoning="test",
        )

    _install_fake_classifier(monkeypatch, _classify)
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--isco-catalogue-profile", _OFFICIAL_PROFILE,
        "--require-genuine-hierarchical", "--max-stage-latency-ms", "30000",
        "--output-dir", str(out_dir),
    ])
    run_eval.main()

    out_csv = next(out_dir.glob("*.csv"))
    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv_module.DictReader(f))
    row = rows[0]
    parsed = json.loads(row["hier_stage_query_telemetry"])
    assert parsed["stage4"]["any_retry"] is True
    assert parsed["stage4"]["stage_budget_exhausted"] is False
    assert parsed["stage4"]["configured_query_timeout_seconds"] == 8.0
    assert len(parsed["stage4"]["queries_detail"]) == 3
    # Never conflated with flat telemetry.
    assert row["flat_query_outcome"] == ""
    assert row["flat_query_attempts"] == ""
