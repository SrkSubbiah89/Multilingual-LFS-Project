"""
Tests for the B2 additions to eval/run_eval.py: capture_pool_metadata
threading, the new CaseResult bookkeeping fields, and the
invalid_output_flag/timed_out_flag derivation in run_one_case().

Scope is deliberately B2-only -- this file does not re-test B0/B1 behaviour
(stage1-4 candidate columns, reranker_output, escalation, SRE, etc.), which
predates this change and is unaffected by it (see run_eval.py's module
docstring and the capture_pool_metadata docstrings in hierarchical_store.py
/ isco_classifier.py for why). What IS tested here is that every new field
defaults to a harmless, B0/B1-identical value when the new params are
omitted, and is populated correctly when they're supplied.

run_eval.py is not a package (no eval/__init__.py) -- imported the same way
run_eval.py itself reaches into backend/, via an explicit sys.path insert,
so this file can live next to the module it tests without a package init.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_eval  # noqa: E402


def make_clf_result(code="2512", title_en="Software Developers", confidence=0.9,
                     method="hierarchical", hitl_required=False, reasoning="Selected by LLM classifier."):
    return SimpleNamespace(
        primary=SimpleNamespace(code=code, title_en=title_en, title_ar="", confidence=confidence),
        method=method,
        hitl_required=hitl_required,
        reasoning=reasoning,
    )


def make_fake_clf(clf_result=None, trace_updates=None):
    """A duck-typed stand-in for ISCOClassifier/_BM25Adapter -- run_one_case()
    only ever calls .classify(job_title=, language=, top_k=, use_llm=, trace=)."""
    clf_result = clf_result or make_clf_result()
    trace_updates = trace_updates or {}

    clf = MagicMock()

    def _classify(job_title, language, top_k, use_llm, trace):
        trace.update(trace_updates)
        return clf_result

    clf.classify.side_effect = _classify
    return clf


def base_kwargs():
    return dict(
        sre=MagicMock(), isic_clf=MagicMock(), isced_clf=MagicMock(),
        row_index=0, case_id="c1", input_text="software developer",
        input_language="en", gold_isco_4digit="2512", gold_isic="", gold_isced="",
        config_hash="abc123",
    )


# ---------------------------------------------------------------------------
# CaseResult defaults -- harmless for any existing B0/B1 row
# ---------------------------------------------------------------------------

def test_case_result_b2_fields_default_harmless():
    r = run_eval.CaseResult(case_id="c1", input_text="x", input_language="en")
    assert r.capture_pool_metadata_enabled is False
    assert r.stage4_pool_enriched == "[]"
    assert r.run_id == ""
    assert r.git_commit == ""
    assert r.seed is None
    assert r.input_order_position is None
    assert r.retry_count == 0
    assert r.invalid_output_flag is False
    assert r.timed_out_flag is False
    assert r.peak_memory_mb is None


# ---------------------------------------------------------------------------
# build_system() threads capture_pool_metadata through, defaults False
# ---------------------------------------------------------------------------

def test_build_system_passes_capture_pool_metadata_default_false(monkeypatch):
    captured = {}
    fake_cls = MagicMock(side_effect=lambda **kw: captured.update(kw))
    monkeypatch.setattr(run_eval, "ISCOClassifier", fake_cls)
    run_eval.build_system("hierarchical", reranker_model="ollama/llama3.2:1b")
    assert captured["capture_pool_metadata"] is False


def test_build_system_passes_capture_pool_metadata_true(monkeypatch):
    captured = {}
    fake_cls = MagicMock(side_effect=lambda **kw: captured.update(kw))
    monkeypatch.setattr(run_eval, "ISCOClassifier", fake_cls)
    run_eval.build_system("hierarchical", reranker_model="ollama/llama3.2:1b", capture_pool_metadata=True)
    assert captured["capture_pool_metadata"] is True


# ---------------------------------------------------------------------------
# run_one_case(): new params populate the new fields, default omitted
# ---------------------------------------------------------------------------

def test_run_one_case_b2_params_default_to_harmless_values():
    clf = make_fake_clf()
    r = run_eval.run_one_case(clf, **base_kwargs())
    assert r.capture_pool_metadata_enabled is False
    assert r.run_id == ""
    assert r.git_commit == ""
    assert r.seed is None
    assert r.input_order_position is None
    assert r.stage4_pool_enriched == "[]"


def test_run_one_case_b2_params_populate_when_passed():
    clf = make_fake_clf()
    r = run_eval.run_one_case(
        clf, **base_kwargs(),
        capture_pool_metadata_enabled=True, run_id="run-42", git_commit="abcdef1",
        seed=7, input_order_position=3,
    )
    assert r.capture_pool_metadata_enabled is True
    assert r.run_id == "run-42"
    assert r.git_commit == "abcdef1"
    assert r.seed == 7
    assert r.input_order_position == 3


def test_run_one_case_captures_stage4_pool_enriched_when_present():
    enriched = [{"code": "2512", "branch_id": "2/25/251", "source_rank": 1,
                 "raw_score": 0.8, "normalized_score": 1.0, "path": ["2", "25", "251", "2512"]}]
    clf = make_fake_clf(trace_updates={"stage4_pool_enriched": enriched})
    r = run_eval.run_one_case(clf, **base_kwargs(), capture_pool_metadata_enabled=True)
    assert r.stage4_pool_enriched != "[]"
    import json
    assert json.loads(r.stage4_pool_enriched) == enriched


def test_run_one_case_stage4_pool_untouched_by_enriched_capture():
    """B1's existing stage4_pool / gold_rank_in_pool must come from
    trace['stage4_pool'] only -- never from the new enriched trace key,
    even when both are present in the same trace."""
    pool = [{"code": "2512", "label_en": "", "score": 0.8}]
    enriched = [{"code": "9999", "branch_id": "x", "source_rank": 1,
                 "raw_score": 0.1, "normalized_score": 0.1, "path": []}]
    clf = make_fake_clf(trace_updates={"stage4_pool": pool, "stage4_pool_enriched": enriched})
    r = run_eval.run_one_case(clf, **base_kwargs(), capture_pool_metadata_enabled=True)
    import json
    assert json.loads(r.stage4_pool) == pool
    assert r.gold_rank_in_pool == 1  # gold_isco_4digit="2512" is pool[0], from stage4_pool not enriched


# ---------------------------------------------------------------------------
# invalid_output_flag / timed_out_flag derivation
# ---------------------------------------------------------------------------

def test_invalid_output_flag_true_when_reasoning_matches_parse_failure_string():
    clf_result = make_clf_result(reasoning="Fallback to top semantic match (LLM response could not be parsed).")
    clf = make_fake_clf(clf_result=clf_result)
    r = run_eval.run_one_case(clf, **base_kwargs())
    assert r.invalid_output_flag is True
    assert r.timed_out_flag is False


def test_invalid_output_flag_false_on_normal_selection():
    clf = make_fake_clf(clf_result=make_clf_result(reasoning="Selected by LLM classifier."))
    r = run_eval.run_one_case(clf, **base_kwargs())
    assert r.invalid_output_flag is False


def test_timed_out_flag_true_on_timeout_signature_reranker_error():
    clf = make_fake_clf(trace_updates={
        "reranker_error": "Timeout: Connection timed out after 120.0 seconds",
    })
    r = run_eval.run_one_case(clf, **base_kwargs())
    assert r.timed_out_flag is True
    assert r.degraded is True


def test_timed_out_flag_false_on_non_timeout_reranker_error():
    clf = make_fake_clf(trace_updates={"reranker_error": "ConnectionRefusedError: refused"})
    r = run_eval.run_one_case(clf, **base_kwargs())
    assert r.timed_out_flag is False
    assert r.degraded is True  # still degraded -- just not a timeout specifically


def test_retry_count_always_zero_for_standard_reranker():
    clf = make_fake_clf()
    r = run_eval.run_one_case(clf, **base_kwargs())
    assert r.retry_count == 0
