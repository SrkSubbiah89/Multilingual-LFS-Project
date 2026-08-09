"""
Tests for Task 25's additive flat-query telemetry as it flows through
eval/run_eval.py::run_one_case()/CaseResult into the written CSV, and its
interaction with check_strict_hierarchical() (Task 13's strict guard).

Fully hermetic: ISCOClassifier is always monkeypatched to a fake that
fails loudly if constructed with unexpected kwargs -- same pattern as
eval/test_official_isco08_profile_evaluator.py. No Qdrant, embedding
model, LLM, or WISCO artifact is used anywhere in this file.
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

_OFFICIAL_PROFILE = "official_ilo2021_v1"


def _write_isco_only_test_set(tmp_path: Path) -> Path:
    test_set = tmp_path / "test.csv"
    test_set.write_text(
        "case_id,input_text,input_language,gold_isco_4digit\n"
        "c1,unclassifiable job title,ar,8131\n",
        encoding="utf-8",
    )
    return test_set


def _fake_isco_instance_with_flat_exception():
    """Simulates exactly Task 24's observed row: an unavailable result
    whose trace carries Task 25's new flat-query exception telemetry."""
    m = MagicMock()
    m.reranker_model_resolved = "none (reranking disabled)"

    def _classify(job_title, language, top_k, use_llm, trace, max_stage_latency_ms=None):
        if trace is not None:
            trace["flat_query_outcome"] = "exception"
            trace["flat_query_duration_ms"] = 30125.417
            trace["flat_query_exception_type"] = "ResponseHandlingException"
            trace["flat_query_exception_message"] = "Timed out"
        return SimpleNamespace(
            primary=SimpleNamespace(code="", title_en="Unknown", title_ar="", confidence=0.0),
            method=f"unavailable_isco08_{_OFFICIAL_PROFILE}", hitl_required=True,
            reasoning="No candidates returned by the vector store.",
        )

    m.classify.side_effect = _classify
    return m


def _fake_isco_instance_success():
    m = MagicMock()
    m.reranker_model_resolved = "none (reranking disabled)"

    def _classify(job_title, language, top_k, use_llm, trace, max_stage_latency_ms=None):
        if trace is not None:
            trace["flat_query_outcome"] = "success"
            trace["flat_query_duration_ms"] = 12.3
        return SimpleNamespace(
            primary=SimpleNamespace(code="2512", title_en="Software Developers", title_ar="", confidence=0.9),
            method=f"flat_isco08_{_OFFICIAL_PROFILE}", hitl_required=False, reasoning="test",
        )

    m.classify.side_effect = _classify
    return m


# ---------------------------------------------------------------------------
# 5. Serialization of the additive telemetry to evaluation output
# ---------------------------------------------------------------------------

def test_flat_query_exception_telemetry_serialized_to_csv(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        run_eval, "ISCOClassifier",
        MagicMock(side_effect=lambda **kw: _fake_isco_instance_with_flat_exception()),
    )
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError("must not construct ISIC")))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError("must not construct ISCED")))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError("must not construct SRE")))

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "flat",
        "--use-llm-reranker", "off", "--isco-catalogue-profile", _OFFICIAL_PROFILE,
        "--output-dir", str(out_dir),
    ])
    run_eval.main()

    out_csv = next(out_dir.glob("*.csv"))
    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv_module.DictReader(f))
    row = rows[0]
    assert row["pred_method"] == f"unavailable_isco08_{_OFFICIAL_PROFILE}"
    assert row["flat_query_outcome"] == "exception"
    assert row["flat_query_duration_ms"] == "30125.417"
    assert row["flat_query_exception_type"] == "ResponseHandlingException"
    assert row["flat_query_exception_message"] == "Timed out"


def test_flat_query_success_telemetry_serialized_and_exception_fields_blank(tmp_path, monkeypatch):
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        run_eval, "ISCOClassifier",
        MagicMock(side_effect=lambda **kw: _fake_isco_instance_success()),
    )
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError("must not construct ISIC")))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError("must not construct ISCED")))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError("must not construct SRE")))

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "flat",
        "--use-llm-reranker", "off", "--isco-catalogue-profile", _OFFICIAL_PROFILE,
        "--output-dir", str(out_dir),
    ])
    run_eval.main()

    out_csv = next(out_dir.glob("*.csv"))
    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv_module.DictReader(f))
    row = rows[0]
    assert row["pred_isco_4digit"] == "2512"  # existing classification unaffected
    assert row["flat_query_outcome"] == "success"
    assert row["flat_query_exception_type"] == ""
    assert row["flat_query_exception_message"] == ""


def test_hierarchical_run_leaves_flat_query_telemetry_blank(tmp_path, monkeypatch):
    """A hierarchical (non-flat) run's trace never touches flat_query_*
    keys at all -- CaseResult defaults (blank/None) must survive to CSV."""
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"

    def _ctor(**kw):
        m = MagicMock()
        m.reranker_model_resolved = "none (reranking disabled)"

        def _classify(job_title, language, top_k, use_llm, trace, max_stage_latency_ms=None):
            if trace is not None:
                for i in range(1, 5):
                    trace[f"stage{i}"] = [{"code": "8131", "label_en": "x", "score": 0.9}]
                    trace[f"stage{i}_latency_ms"] = 5.0
            return SimpleNamespace(
                primary=SimpleNamespace(code="8131", title_en="x", title_ar="", confidence=0.9),
                method=f"hierarchical_isco08_{_OFFICIAL_PROFILE}", hitl_required=False, reasoning="test",
            )

        m.classify.side_effect = _classify
        return m

    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=_ctor))
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError("must not construct ISIC")))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError("must not construct ISCED")))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError("must not construct SRE")))

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--test-set", str(test_set), "--system", "hierarchical",
        "--use-llm-reranker", "off", "--isco-catalogue-profile", _OFFICIAL_PROFILE,
        "--require-genuine-hierarchical", "--output-dir", str(out_dir),
    ])
    run_eval.main()

    out_csv = next(out_dir.glob("*.csv"))
    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv_module.DictReader(f))
    row = rows[0]
    assert row["flat_query_outcome"] == ""
    assert row["flat_query_duration_ms"] == ""
    assert row["flat_query_exception_type"] == ""
    assert row["flat_query_exception_message"] == ""


# ---------------------------------------------------------------------------
# 7. Strict-hierarchical guard behaviour is unchanged by the new fields
# ---------------------------------------------------------------------------

def test_strict_guard_ignores_new_telemetry_fields_still_rejects_fallback():
    """check_strict_hierarchical() must keep rejecting a flat/unavailable
    pred_method exactly as before -- it never inspects flat_query_* at
    all, so the new fields cannot be mistaken for stage evidence."""
    result = run_eval.CaseResult(
        case_id="c1", input_text="x", input_language="ar",
        pred_method=f"unavailable_isco08_{_OFFICIAL_PROFILE}",
        flat_query_outcome="exception",
        flat_query_exception_type="TimeoutError",
        flat_query_exception_message="timed out",
    )
    violation = run_eval.check_strict_hierarchical(result, max_stage_latency_ms=None)
    assert violation is not None
    assert "not a genuine hierarchical method" in violation


def test_strict_guard_still_passes_genuine_hierarchical_case_with_blank_telemetry():
    result = run_eval.CaseResult(
        case_id="c1", input_text="x", input_language="en",
        pred_method=f"hierarchical_isco08_{_OFFICIAL_PROFILE}",
        stage1_candidates='[{"code": "2"}]', stage2_candidates='[{"code": "25"}]',
        stage3_candidates='[{"code": "251"}]', stage4_candidates='[{"code": "2512"}]',
        stage1_latency_ms=1.0, stage2_latency_ms=1.0, stage3_latency_ms=1.0, stage4_latency_ms=1.0,
    )
    violation = run_eval.check_strict_hierarchical(result, max_stage_latency_ms=30000)
    assert violation is None
