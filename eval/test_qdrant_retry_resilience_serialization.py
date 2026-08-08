"""
Tests for Task 27's retry telemetry as it flows through
eval/run_eval.py::run_one_case()/CaseResult into the written CSV, and its
interaction with check_strict_hierarchical() (Task 13's strict guard).

Fully hermetic: ISCOClassifier is always monkeypatched to a fake -- same
pattern as eval/test_flat_query_telemetry_serialization.py. No Qdrant,
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


# ---------------------------------------------------------------------------
# 8. Flat telemetry remains backward-compatible under default (no-retry)
#    config, and serializes the new fields correctly when retry occurs
# ---------------------------------------------------------------------------

def _run_with_fake_classifier(tmp_path, monkeypatch, classify_fn, system="flat", extra_args=None):
    test_set = _write_isco_only_test_set(tmp_path)
    out_dir = tmp_path / "out"
    m = MagicMock()
    m.reranker_model_resolved = "none (reranking disabled)"
    m.classify.side_effect = classify_fn
    monkeypatch.setattr(run_eval, "ISCOClassifier", MagicMock(side_effect=lambda **kw: m))
    monkeypatch.setattr(run_eval, "ISICClassifier", MagicMock(side_effect=AssertionError("must not construct ISIC")))
    monkeypatch.setattr(run_eval, "ISCEDClassifier", MagicMock(side_effect=AssertionError("must not construct ISCED")))
    monkeypatch.setattr(run_eval, "SemanticRelationEngine", MagicMock(side_effect=AssertionError("must not construct SRE")))
    argv = [
        "run_eval.py", "--test-set", str(test_set), "--system", system,
        "--use-llm-reranker", "off", "--isco-catalogue-profile", _OFFICIAL_PROFILE,
        "--output-dir", str(out_dir),
    ]
    if extra_args:
        argv += extra_args
    monkeypatch.setattr(sys, "argv", argv)
    run_eval.main()
    out_csv = next(out_dir.glob("*.csv"))
    with out_csv.open(encoding="utf-8", newline="") as f:
        return list(csv_module.DictReader(f))[0]


def test_flat_default_no_retry_attempts_field_is_one(tmp_path, monkeypatch):
    def _classify(job_title, language, top_k, use_llm, trace):
        if trace is not None:
            trace["flat_query_outcome"] = "success"
            trace["flat_query_duration_ms"] = 12.3
            trace["flat_query_attempts"] = 1
            trace["flat_query_attempt_durations_ms"] = [12.3]
        return SimpleNamespace(
            primary=SimpleNamespace(code="2512", title_en="x", title_ar="", confidence=0.9),
            method=f"flat_isco08_{_OFFICIAL_PROFILE}", hitl_required=False, reasoning="test",
        )

    row = _run_with_fake_classifier(tmp_path, monkeypatch, _classify)
    assert row["flat_query_outcome"] == "success"
    assert row["flat_query_attempts"] == "1"
    assert json.loads(row["flat_query_attempt_durations_ms"]) == [12.3]


def test_flat_success_after_retry_serializes_new_outcome_value(tmp_path, monkeypatch):
    def _classify(job_title, language, top_k, use_llm, trace):
        if trace is not None:
            trace["flat_query_outcome"] = "success_after_retry"
            trace["flat_query_duration_ms"] = 145.9
            trace["flat_query_attempts"] = 2
            trace["flat_query_attempt_durations_ms"] = [95.0, 50.9]
        return SimpleNamespace(
            primary=SimpleNamespace(code="2512", title_en="x", title_ar="", confidence=0.9),
            method=f"flat_isco08_{_OFFICIAL_PROFILE}", hitl_required=False, reasoning="test",
        )

    row = _run_with_fake_classifier(tmp_path, monkeypatch, _classify)
    assert row["flat_query_outcome"] == "success_after_retry"
    assert row["flat_query_attempts"] == "2"
    assert json.loads(row["flat_query_attempt_durations_ms"]) == [95.0, 50.9]


def test_flat_retry_exhausted_serializes_correctly_and_no_fabricated_code(tmp_path, monkeypatch):
    def _classify(job_title, language, top_k, use_llm, trace):
        if trace is not None:
            trace["flat_query_outcome"] = "retry_exhausted"
            trace["flat_query_duration_ms"] = 300.0
            trace["flat_query_attempts"] = 3
            trace["flat_query_attempt_durations_ms"] = [100.0, 100.0, 100.0]
            trace["flat_query_exception_type"] = "ResponseHandlingException"
            trace["flat_query_exception_message"] = "timed out"
        return SimpleNamespace(
            primary=SimpleNamespace(code="", title_en="Unknown", title_ar="", confidence=0.0),
            method=f"unavailable_isco08_{_OFFICIAL_PROFILE}", hitl_required=True,
            reasoning="No candidates returned by the vector store.",
        )

    row = _run_with_fake_classifier(tmp_path, monkeypatch, _classify)
    assert row["pred_isco_4digit"] == ""  # never fabricated
    assert row["flat_query_outcome"] == "retry_exhausted"
    assert row["flat_query_attempts"] == "3"
    assert row["flat_query_exception_type"] == "ResponseHandlingException"


def test_hierarchical_row_leaves_new_flat_fields_blank(tmp_path, monkeypatch):
    def _classify(job_title, language, top_k, use_llm, trace):
        if trace is not None:
            for i in range(1, 5):
                trace[f"stage{i}"] = [{"code": "8131", "label_en": "x", "score": 0.9}]
                trace[f"stage{i}_latency_ms"] = 5.0
        return SimpleNamespace(
            primary=SimpleNamespace(code="8131", title_en="x", title_ar="", confidence=0.9),
            method=f"hierarchical_isco08_{_OFFICIAL_PROFILE}", hitl_required=False, reasoning="test",
        )

    row = _run_with_fake_classifier(tmp_path, monkeypatch, _classify, system="hierarchical",
                                     extra_args=["--require-genuine-hierarchical"])
    assert row["flat_query_outcome"] == ""
    assert row["flat_query_attempts"] == ""
    assert row["flat_query_attempt_durations_ms"] == "[]"


# ---------------------------------------------------------------------------
# 9. Hierarchical-stage telemetry is serialized distinctly from flat
# ---------------------------------------------------------------------------

def test_hier_stage_query_telemetry_serializes_and_is_distinct_from_flat(tmp_path, monkeypatch):
    def _classify(job_title, language, top_k, use_llm, trace):
        if trace is not None:
            for i in range(1, 5):
                trace[f"stage{i}"] = [{"code": "8131", "label_en": "x", "score": 0.9}]
                trace[f"stage{i}_latency_ms"] = 5.0
            trace["stage2_query_telemetry"] = {
                "queries": 3, "any_retry": True, "any_exception": False,
                "max_attempts_used": 2, "exception_types": [],
            }
            trace["stage3_query_telemetry"] = {
                "queries": 1, "any_retry": False, "any_exception": False,
                "max_attempts_used": 1, "exception_types": [],
            }
        return SimpleNamespace(
            primary=SimpleNamespace(code="8131", title_en="x", title_ar="", confidence=0.9),
            method=f"hierarchical_isco08_{_OFFICIAL_PROFILE}", hitl_required=False, reasoning="test",
        )

    row = _run_with_fake_classifier(tmp_path, monkeypatch, _classify, system="hierarchical",
                                     extra_args=["--require-genuine-hierarchical"])
    parsed = json.loads(row["hier_stage_query_telemetry"])
    assert parsed["stage2"]["any_retry"] is True
    assert parsed["stage3"]["any_retry"] is False
    assert "stage1" not in parsed  # only stages that actually had telemetry are present
    assert row["flat_query_outcome"] == ""  # never conflated with flat telemetry


# ---------------------------------------------------------------------------
# 10. Strict genuine-hierarchical guard still rejects retry-exhaustion
#     fallout, missing evidence, and excessive total stage latency
# ---------------------------------------------------------------------------

def test_strict_guard_rejects_flat_fallback_caused_by_retry_exhaustion():
    """A retry-exhausted stage-1 query causes the existing fallback path
    to fire (pred_method becomes flat_...) -- the strict guard must
    reject this exactly as it did before Task 27 existed."""
    result = run_eval.CaseResult(
        case_id="c1", input_text="x", input_language="ar",
        pred_method=f"flat_isco08_{_OFFICIAL_PROFILE}",
    )
    violation = run_eval.check_strict_hierarchical(result, max_stage_latency_ms=None)
    assert violation is not None
    assert "not a genuine hierarchical method" in violation


def test_strict_guard_ignores_hier_stage_query_telemetry_field_entirely():
    """The new hier_stage_query_telemetry field must never be mistaken
    for stage evidence -- a genuine hierarchical case with retry
    telemetry attached still passes on its own existing merits."""
    result = run_eval.CaseResult(
        case_id="c1", input_text="x", input_language="en",
        pred_method=f"hierarchical_isco08_{_OFFICIAL_PROFILE}",
        stage1_candidates='[{"code": "2"}]', stage2_candidates='[{"code": "25"}]',
        stage3_candidates='[{"code": "251"}]', stage4_candidates='[{"code": "2512"}]',
        stage1_latency_ms=1.0, stage2_latency_ms=1.0, stage3_latency_ms=1.0, stage4_latency_ms=1.0,
        hier_stage_query_telemetry=json.dumps({"stage2": {"any_retry": True, "queries": 2}}),
    )
    violation = run_eval.check_strict_hierarchical(result, max_stage_latency_ms=30000)
    assert violation is None


def test_strict_guard_rejects_excessive_total_stage_latency_including_retry_time():
    """max_stage_latency_ms must see the FULL time including any retry
    and backoff -- a stage whose accumulated latency (retry-inclusive)
    exceeds the bound is still rejected."""
    result = run_eval.CaseResult(
        case_id="c1", input_text="x", input_language="en",
        pred_method=f"hierarchical_isco08_{_OFFICIAL_PROFILE}",
        stage1_candidates='[{"code": "2"}]', stage2_candidates='[{"code": "25"}]',
        stage3_candidates='[{"code": "251"}]', stage4_candidates='[{"code": "2512"}]',
        stage1_latency_ms=1.0, stage2_latency_ms=35000.0,  # retry+backoff pushed this stage over the bound
        stage3_latency_ms=1.0, stage4_latency_ms=1.0,
    )
    violation = run_eval.check_strict_hierarchical(result, max_stage_latency_ms=30000)
    assert violation is not None
    assert "stage2_latency_ms" in violation


def test_strict_guard_rejects_missing_stage_evidence_unchanged():
    result = run_eval.CaseResult(
        case_id="c1", input_text="x", input_language="en",
        pred_method=f"hierarchical_isco08_{_OFFICIAL_PROFILE}",
        stage1_candidates='[]', stage2_candidates='[{"code": "25"}]',
        stage3_candidates='[{"code": "251"}]', stage4_candidates='[{"code": "2512"}]',
    )
    violation = run_eval.check_strict_hierarchical(result, max_stage_latency_ms=None)
    assert violation is not None
    assert "stage1_candidates" in violation
