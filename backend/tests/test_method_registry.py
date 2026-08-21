"""
Tests for backend/agents/method_registry.py -- Section A of the Conference I
Reviewer #2 response (classifier method registry, reviewer comment 6).
"""

from __future__ import annotations

import json

import pytest

from backend.agents import method_registry as mr
from backend.agents.classifier_methods import (
    ISCEDF_HIERARCHICAL_RETRIEVAL,
    ISIC_HIERARCHICAL_RETRIEVAL,
)

_HIERARCHICAL_RETRIEVAL_METHODS = {ISIC_HIERARCHICAL_RETRIEVAL, ISCEDF_HIERARCHICAL_RETRIEVAL}

_NAMED_COMPONENTS = {
    "LanguageProcessor", "ConversationManager", "ISCOClassifier",
    "ISICClassifier", "ISCEDClassifier", "SemanticRelationEngine",
    "ValidationAgent", "HITLQualityManager",
}


# ---------------------------------------------------------------------------
# Coverage: every required component has at least one row
# ---------------------------------------------------------------------------

def test_every_named_component_has_at_least_one_row():
    present = {e.component for e in mr.REGISTRY}
    missing = _NAMED_COMPONENTS - present
    assert not missing, f"Missing registry rows for: {missing}"


def test_no_unexpected_components():
    present = {e.component for e in mr.REGISTRY}
    assert present == _NAMED_COMPONENTS


def test_isco_classifier_has_hierarchical_rag_row():
    isco_methods = {e.method_id for e in mr.REGISTRY if e.component == "ISCOClassifier"}
    assert "isco_hierarchical_rag" in isco_methods


def test_isic_classifier_has_both_current_and_stub_rows():
    isic_methods = {e.method_id for e in mr.REGISTRY if e.component == "ISICClassifier"}
    assert "isic_keyword_llm" in isic_methods
    assert "isic_hierarchical_retrieval" in isic_methods


def test_isced_classifier_has_both_current_and_stub_rows():
    isced_methods = {e.method_id for e in mr.REGISTRY if e.component == "ISCEDClassifier"}
    assert "isced_rule_keyword" in isced_methods
    assert "iscedf_hierarchical_retrieval" in isced_methods


# ---------------------------------------------------------------------------
# Honesty checks: the ISIC/ISCED-F hierarchical-retrieval rows describe a
# REAL, tested code path (Task 05) but must not overclaim -- no accuracy
# measurement exists yet, and the registry must not silently start claiming
# affects_hitl_escalation without the live production path
# (backend/api/survey_routes.py) actually being rewired for it. These
# directly replace the old "stub is never marked evaluated" tests now that
# the stub no longer exists.
# ---------------------------------------------------------------------------

def test_hierarchical_retrieval_rows_are_real_but_unevaluated():
    for e in mr.REGISTRY:
        if e.method_id in _HIERARCHICAL_RETRIEVAL_METHODS:
            assert e.category == "retrieval"
            assert e.evaluated is False, f"{e.component}/{e.method_id}: no accuracy measurement exists yet -- must stay evaluated=False"
            assert e.evaluated_ref is None
            assert e.embedding_model, f"{e.component}/{e.method_id}: real retrieval path must declare its embedding model"


def test_hierarchical_retrieval_rows_never_affect_hitl_escalation():
    for e in mr.REGISTRY:
        if e.method_id in _HIERARCHICAL_RETRIEVAL_METHODS:
            assert e.affects_hitl_escalation is False


def test_hierarchical_retrieval_rows_document_explicit_fallback_labels():
    """The registry text itself must name the *_hierarchical_fallback_*
    labels -- a reader must not be able to conclude from this row alone that
    method=isic_hierarchical_retrieval / iscedf_hierarchical_retrieval is
    always what gets returned."""
    for e in mr.REGISTRY:
        if e.method_id in _HIERARCHICAL_RETRIEVAL_METHODS:
            assert "fallback" in e.fallback_behaviour.lower()
            assert "not yet evaluated" in e.fallback_behaviour.lower() or "NOT YET EVALUATED" in e.fallback_behaviour


def test_isced_hierarchical_retrieval_row_documents_independent_level():
    entry = next(e for e in mr.REGISTRY if e.method_id == ISCEDF_HIERARCHICAL_RETRIEVAL)
    assert "level" in entry.output_schema
    assert "independent" in entry.output_schema["level"].lower()


# ---------------------------------------------------------------------------
# Regression guard: affects_hitl_escalation must match verified wiring
# ---------------------------------------------------------------------------
#
# The LIVE production message-handling path is backend/api/survey_routes.py
# (_send_message_impl) -- confirmed by grep to be the only module the real
# FastAPI app imports for this. An earlier module,
# backend/agents/survey_orchestrator.py, also computed a TurnResult and
# called self._hitl.review_session(session_id) without reading
# semantic_coherence or rule_violations, but that module was confirmed dead
# code (never imported by the live API) with zero effect on production
# traffic -- so its wiring was never what affects_hitl_escalation
# describes -- and it was removed from this codebase after that finding was
# documented. As of 2026-08-16 (Module D Step 5.5), survey_routes.py itself
# queues a real HITLQueue row whenever SemanticRelationEngine returns a
# HIGH-severity violation (Stage 4e); ValidationAgent's rule_violations are
# still only audit-logged there, never read near the escalation decision.
# If a future change to survey_routes.py rewires either of these, this test
# (and the registry entries it checks) must be updated together -- that is
# the entire point of keeping this assertion here rather than just trusting
# the docstring.

def test_semantic_relation_engine_does_affect_hitl_escalation():
    entries = [e for e in mr.REGISTRY if e.component == "SemanticRelationEngine"]
    assert entries
    assert all(e.affects_hitl_escalation is True for e in entries)


def test_validation_agent_does_not_affect_hitl_escalation():
    entries = [e for e in mr.REGISTRY if e.component == "ValidationAgent"]
    assert entries
    assert all(e.affects_hitl_escalation is False for e in entries)


def test_isco_classifier_and_hitl_quality_manager_do_affect_escalation():
    isco = [e for e in mr.REGISTRY if e.component == "ISCOClassifier" and e.method_id == "isco_hierarchical_rag"]
    hitl_scoring = [e for e in mr.REGISTRY if e.component == "HITLQualityManager" and e.method_id == "quality_scoring_deterministic"]
    assert isco and isco[0].affects_hitl_escalation is True
    assert hitl_scoring and hitl_scoring[0].affects_hitl_escalation is True


# ---------------------------------------------------------------------------
# Export round-trip
# ---------------------------------------------------------------------------

def test_export_json_round_trips(tmp_path):
    out = tmp_path / "registry.json"
    mr.export_json(out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["entry_count"] == len(mr.REGISTRY)
    assert len(payload["entries"]) == len(mr.REGISTRY)
    assert payload["entries"][0]["component"] == mr.REGISTRY[0].component


def test_export_markdown_contains_every_component(tmp_path):
    out = tmp_path / "registry.md"
    mr.export_markdown(out)
    text = out.read_text(encoding="utf-8")
    for component in _NAMED_COMPONENTS:
        assert component in text


def test_export_markdown_is_nonempty_and_has_table_header(tmp_path):
    out = tmp_path / "registry.md"
    mr.export_markdown(out)
    text = out.read_text(encoding="utf-8")
    assert "| Component | Method |" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_writes_both_files(tmp_path, monkeypatch):
    import sys
    out_dir = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", ["method_registry.py", "--out", str(out_dir)])
    mr.main()
    assert (out_dir / "classifier_method_registry.json").exists()
    assert (out_dir / "classifier_method_registry.md").exists()


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------

def test_every_entry_has_nonempty_output_schema():
    for e in mr.REGISTRY:
        assert e.output_schema, f"{e.component}/{e.method_id} has empty output_schema"


def test_every_entry_has_nonempty_fallback_behaviour():
    for e in mr.REGISTRY:
        assert e.fallback_behaviour.strip(), f"{e.component}/{e.method_id} has empty fallback_behaviour"


def test_category_is_one_of_four_allowed_values():
    allowed = {"deterministic", "retrieval", "llm", "hybrid"}
    for e in mr.REGISTRY:
        assert e.category in allowed
