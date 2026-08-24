"""
backend/tests/test_orchestration_correctness.py

Module H (Conference I Reviewer #2 response), built 2026-08-24.

CLAUDE.md's own backlog entry for Module H says: "Its 'delegation
correctness' framing needs rescoping first -- this system never uses
CrewAI delegation ... the real evaluable target is 'orchestration
correctness' (does the calling code invoke the right agent at the right
time)." This file is that evaluation, expressed as real, deterministic
tests rather than a statistical/accuracy metric -- "does the code call
the right thing in the right order" is a correctness property, not
something with a meaningful confidence interval.

Ground truth for "the right order" was read directly out of
backend/api/survey_routes.py's _send_message_impl (the only live
per-turn orchestration path -- backend/agents/survey_orchestrator.py,
an earlier module that duplicated some of this logic, was confirmed
dead code and removed; see CLAUDE.md), not assumed or inferred from
documentation. Each test's docstring cites the real line(s) it guards.

Instrumentation approach: every agent this endpoint calls is already
reached through a small, individually-patchable module-level getter
(_get_agents, _get_isco_classifier, _get_isic_classifier, etc. --
survey_routes.py:86-172), the exact pattern test_sre_hitl_enforcement.py
and test_hitl_and_e2e_extended.py already patch per-getter. No production
code changes were needed: this file adds a `side_effect` to each already-
mockable getter's returned method that appends the agent's name to a
shared, ordered `call_order` list before returning a realistic result,
then asserts on that list. Stage 4e (SemanticRelationEngine) is the one
exception -- survey_routes.py imports and calls
`get_semantic_relation_engine()` directly inside the try block rather
than through a survey_routes-local getter, and its result is fed to
`dataclasses.asdict()`, which requires a real dataclass instance, not a
MagicMock (this is the same reason test_sre_hitl_enforcement.py uses the
real engine rather than mocking it). This file does the same: patches
`backend.agents.semantic_relation.get_semantic_relation_engine` to return
a wrapper whose `.analyse()` records into `call_order` and then delegates
to the real engine (`use_llm=False`, fully deterministic, no network).
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture(scope="module")
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    from backend.database.models import Base
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def db(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


def _recording(call_order, name, retval):
    """side_effect callable: records `name` into call_order, then returns retval."""
    def _fn(*args, **kwargs):
        call_order.append(name)
        return retval
    return _fn


def _make_client(
    db,
    *,
    state="VALIDATING",
    job_title="Assembler",
    industry="manufacturing",
    education_level="secondary",
    nationality="Emirati",
    low_isco_confidence=False,
    isco_code="8211",
    isic_section="C",
    isced_level=3,
    message="I assemble electronic components daily",
):
    """Builds a TestClient wired so every agent call is both realistic
    (well-formed return values, matching test_sre_hitl_enforcement.py's
    proven-safe field construction so downstream Pydantic/dataclasses
    code doesn't silently drop the result) and recorded, in call order,
    into the returned `call_order` list.
    """
    from backend.main import app
    from backend.database.connection import get_db
    from backend.agents.conversation_manager import ConversationContext, ConversationState
    from backend.api import survey_routes as _sr_module
    from backend.agents.semantic_relation import SemanticRelationEngine

    app.dependency_overrides[get_db] = lambda: db
    _sr_module._contexts.clear()

    call_order: list[str] = []

    state_enum = getattr(ConversationState, state)

    mock_cm = MagicMock()
    mock_lp = MagicMock()

    def _new_context(sid, lang="en"):
        ctx = ConversationContext(session_id=sid, language=lang)
        if industry is not None:
            ctx.collected_data["industry"] = industry
        if education_level is not None:
            ctx.collected_data["education_level"] = education_level
        if nationality is not None:
            ctx.collected_data["nationality"] = nationality
        ctx.state = state_enum
        return ctx

    def _process_message(ctx, msg):
        call_order.append("ConversationManager")
        if job_title is not None:
            ctx.collected_data["job_title"] = job_title
        return "Thanks, noted."

    mock_cm.new_context.side_effect = _new_context
    mock_cm.process_message.side_effect = _process_message

    mock_lp_result = MagicMock()
    mock_lp_result.detected_language = "en"
    mock_lp_result.is_code_switched = False
    mock_lp_result.entities = []
    mock_lp.process.side_effect = _recording(call_order, "LanguageProcessor", mock_lp_result)

    isco_conf = 0.40 if low_isco_confidence else 0.95
    isco_result_obj = MagicMock(
        primary=MagicMock(code=isco_code, title_en="Test Title", title_ar="test", confidence=isco_conf),
        method="flat_semantic",
        hitl_required=low_isco_confidence,
        reasoning="test reasoning",
        hierarchy_path=None,
        alternatives=[],
        stage_confidences={},
    )
    isic_result_obj = MagicMock(
        section=isic_section, section_title="Test Section",
        division_code="00", division_title="Test Division",
        group_code="", group_title="", class_code="", class_title="",
        confidence=0.90, method="keyword",
    )
    isced_result_obj = MagicMock(
        level=isced_level, level_title="Test Level",
        broad_code="", broad_title="", narrow_code="", narrow_title="",
        detailed_code="", detailed_title="", confidence=0.90, method="keyword",
    )
    val_result_obj = MagicMock(is_valid=True, rule_violations=[], confidence=0.95)

    patchers = [
        patch("backend.auth.email_otp.send_otp_email", return_value=True),
        patch("backend.api.survey_routes._get_agents", return_value=(mock_cm, mock_lp)),
        patch("backend.api.survey_routes._get_context_memory"),
        patch("backend.api.survey_routes._get_isco_classifier"),
        patch("backend.api.survey_routes._get_isic_classifier"),
        patch("backend.api.survey_routes._get_isced_classifier"),
        patch("backend.api.survey_routes._get_nationality_classifier"),
        patch("backend.api.survey_routes._get_hitl_quality_manager"),
        patch("backend.api.survey_routes._get_emotional_intelligence"),
        patch("backend.api.survey_routes._get_validation_agent"),
        patch("backend.api.survey_routes._get_audit_logger"),
    ]
    started = [p.start() for p in patchers]
    (_send_otp, _agents, mock_ctx_mem, mock_isco, mock_isic, mock_isced,
     mock_nat, mock_hitl, mock_ei, mock_val, mock_audit) = started

    mock_ctx_mem.return_value.load_session.return_value = None
    mock_ctx_mem.return_value.save_session.side_effect = _recording(call_order, "ContextMemory", None)
    mock_isco.return_value.classify.side_effect = _recording(call_order, "ISCOClassifier", isco_result_obj)
    mock_isic.return_value.classify.side_effect = _recording(call_order, "ISICClassifier", isic_result_obj)
    mock_isced.return_value.classify.side_effect = _recording(call_order, "ISCEDClassifier", isced_result_obj)
    mock_nat.return_value.classify.side_effect = _recording(call_order, "NationalityClassifier", MagicMock(method="unknown"))
    mock_hitl.return_value.review_session.side_effect = _recording(call_order, "HITLQualityManager", MagicMock(status=MagicMock(value="pass")))
    mock_ei.return_value.analyze.side_effect = _recording(call_order, "EmotionalIntelligence", MagicMock(state="neutral"))
    mock_val.return_value.validate.side_effect = _recording(call_order, "ValidationAgent", val_result_obj)
    mock_audit.return_value.log_interaction.side_effect = _recording(call_order, "AuditLogger", None)
    mock_audit.return_value.log_agent_decision.side_effect = _recording(call_order, "AuditLogger", None)

    # SemanticRelationEngine: not behind a survey_routes-local getter (it's
    # imported and called inline), and its result feeds dataclasses.asdict()
    # downstream -- must stay a real dataclass instance, so wrap-and-delegate
    # to the real engine rather than replacing it with a MagicMock.
    real_engine = SemanticRelationEngine(use_llm=False)

    def _wrapped_analyse(*args, **kwargs):
        call_order.append("SemanticRelationEngine")
        return real_engine.analyse(*args, **kwargs)

    sre_patcher = patch(
        "backend.agents.semantic_relation.get_semantic_relation_engine",
        return_value=MagicMock(analyse=_wrapped_analyse),
    )
    patchers.append(sre_patcher)
    sre_patcher.start()

    # _ensure_isco_classification / _trigger_quality_review: patch with
    # wraps= so the real implementation still runs (early-returns are real
    # behaviour, not something this file needs to fake), only recording
    # that each was *entered*, in order.
    real_ensure = _sr_module._ensure_isco_classification
    real_trigger = _sr_module._trigger_quality_review

    def _wrapped_ensure(*args, **kwargs):
        call_order.append("_ensure_isco_classification")
        return real_ensure(*args, **kwargs)

    def _wrapped_trigger(*args, **kwargs):
        call_order.append("_trigger_quality_review")
        return real_trigger(*args, **kwargs)

    ensure_patcher = patch("backend.api.survey_routes._ensure_isco_classification", side_effect=_wrapped_ensure)
    trigger_patcher = patch("backend.api.survey_routes._trigger_quality_review", side_effect=_wrapped_trigger)
    patchers.append(ensure_patcher)
    patchers.append(trigger_patcher)
    ensure_patcher.start()
    trigger_patcher.start()

    return TestClient(app), patchers, call_order, message


def _create_session_and_send(client, msg):
    from backend.database.models import User, SurveySession
    from backend.database.connection import get_db

    db_gen = client.app.dependency_overrides[get_db]
    db = db_gen()
    user = User(email=f"orch-test-{id(client)}@test.invalid")
    db.add(user)
    db.commit()
    db.refresh(user)

    from backend.auth.jwt_handler import create_access_token
    token = create_access_token(user.id)

    session = SurveySession(user_id=user.id, language="en", status="active")
    db.add(session)
    db.commit()
    db.refresh(session)

    resp = client.post(
        f"/survey/sessions/{session.id}/message",
        json={"message": msg},
        headers={"Authorization": f"Bearer {token}"},
    )
    return resp, session.id, db


class TestFullTurnOrchestrationOrder:
    """The central Module H claim: for a turn where every optional stage's
    trigger condition is satisfied, the real calling code invokes every
    agent in exactly the order survey_routes.py's own Stage comments
    document (Stage 1/3 -> 4 -> 4b -> 4c -> 4d -> 4e -> 4f -> 4g, then the
    unconditional per-turn ContextMemory/AuditLogger writes)."""

    def test_stage_order_matches_documented_pipeline(self, db):
        client, patchers, call_order, message = _make_client(db, state="VALIDATING")
        try:
            resp, session_id, db2 = _create_session_and_send(client, message)
            assert resp.status_code == 200

            expected = [
                # Stage 1+3. Both LanguageProcessor (here) and Stage 4f
                # EmotionalIntelligence below share the same `not skip_ner`
                # gate (survey_routes.py:737,1141); LFS_FAST_MODE=true in
                # this test env forces skip_ner=True unconditionally, so
                # BOTH are correctly absent below -- confirmed by first
                # running this test with EmotionalIntelligence included,
                # which failed and revealed the shared gate this comment
                # now documents. See test_emotional_intelligence_and_language_processor_share_skip_ner_gate
                # below for the explicit, direct test of that gate.
                "ConversationManager",
                "ISCOClassifier",          # Stage 4
                "ISICClassifier",          # Stage 4b
                "ISCEDClassifier",         # Stage 4c
                "NationalityClassifier",   # Stage 4d
                "SemanticRelationEngine",  # Stage 4e
                # Stage 4f (EmotionalIntelligence) skipped -- see comment above
                "ValidationAgent",         # Stage 4g (state == VALIDATING)
                "ContextMemory",           # unconditional, every turn
                "AuditLogger",             # unconditional, every turn
            ]
            assert call_order == expected, (
                f"real orchestration order diverged from survey_routes.py's "
                f"own documented Stage sequence.\nexpected: {expected}\nactual:   {call_order}"
            )
        finally:
            for p in patchers:
                p.stop()


class TestConditionalGating:
    """"Right agent" half of orchestration correctness: agents whose
    trigger condition is NOT met this turn must not be called at all --
    a real bug class distinct from ordering (e.g. calling ISICClassifier
    on every turn regardless of whether `industry` was ever collected
    would waste an LLM call and could misclassify placeholder data)."""

    def test_isic_not_called_without_industry_field(self, db):
        """Guards survey_routes.py's Stage 4b gate: only fires `if
        ctx.collected_data.get("industry")`."""
        client, patchers, call_order, message = _make_client(db, state="VALIDATING", industry=None)
        try:
            resp, _sid, _db2 = _create_session_and_send(client, message)
            assert resp.status_code == 200
            assert "ISICClassifier" not in call_order
        finally:
            for p in patchers:
                p.stop()

    def test_isced_not_called_without_education_field(self, db):
        """Guards Stage 4c's gate on education_level/field_of_study/EDUCATION entity."""
        client, patchers, call_order, message = _make_client(db, state="VALIDATING", education_level=None)
        try:
            resp, _sid, _db2 = _create_session_and_send(client, message)
            assert resp.status_code == 200
            assert "ISCEDClassifier" not in call_order
        finally:
            for p in patchers:
                p.stop()

    def test_nationality_not_called_without_nationality_field(self, db):
        """Guards Stage 4d's gate on nationality/LOCATION entity."""
        client, patchers, call_order, message = _make_client(db, state="VALIDATING", nationality=None)
        try:
            resp, _sid, _db2 = _create_session_and_send(client, message)
            assert resp.status_code == 200
            assert "NationalityClassifier" not in call_order
        finally:
            for p in patchers:
                p.stop()

    def test_validation_agent_only_fires_in_validating_state(self, db):
        """Guards Stage 4g's strict gate: `ctx.state == ConversationState.VALIDATING`
        -- must NOT fire mid-collection."""
        client, patchers, call_order, message = _make_client(db, state="COLLECTING_INFO")
        try:
            resp, _sid, _db2 = _create_session_and_send(client, message)
            assert resp.status_code == 200
            assert "ValidationAgent" not in call_order
        finally:
            for p in patchers:
                p.stop()

    def test_emotional_intelligence_and_language_processor_share_skip_ner_gate(self, db):
        """Guards survey_routes.py:737's `skip_ner = _FAST_MODE or msg.lower()
        in _NER_SKIP_TOKENS or len(msg) <= 3` and Stage 4f's `if not skip_ner
        and len(msg) > 10` (survey_routes.py:1141) -- discovered while
        building this file: EmotionalIntelligence is gated on the *same*
        skip_ner flag as LanguageProcessor, not an independent condition.
        Under this test env's LFS_FAST_MODE=true, skip_ner is always True
        regardless of message content, so both must be absent -- this test
        makes that shared dependency explicit and independently checked,
        rather than only implicit in test_stage_order_matches_documented_pipeline's
        expected-list comment."""
        client, patchers, call_order, message = _make_client(
            db, state="VALIDATING", message="a much longer message than eleven characters"
        )
        try:
            resp, _sid, _db2 = _create_session_and_send(client, message)
            assert resp.status_code == 200
            assert "LanguageProcessor" not in call_order
            assert "EmotionalIntelligence" not in call_order
        finally:
            for p in patchers:
                p.stop()

    def test_semantic_relation_engine_not_called_when_isco_has_no_results(self, db):
        """Guards Stage 4e's gate: `if isco_results and ctx.collected_data`
        -- SRE must never run on a turn where ISCO produced nothing to
        cross-validate against (job_title never collected)."""
        client, patchers, call_order, message = _make_client(db, state="VALIDATING", job_title=None)
        try:
            resp, _sid, _db2 = _create_session_and_send(client, message)
            assert resp.status_code == 200
            assert "SemanticRelationEngine" not in call_order
        finally:
            for p in patchers:
                p.stop()


class TestSessionCompletionOrder:
    """Guards the one explicitly-documented cross-function ordering
    invariant in survey_routes.py: "_ensure_isco_classification must run
    before _trigger_quality_review so the quality metrics include the
    freshly assigned ISCO code" (survey_routes.py:1224). A regression that
    silently swapped these two calls would make HITLQualityManager score
    sessions against stale/missing ISCO codes -- exactly the kind of bug
    an output-only test (asserting the final QualityReview row exists)
    would not catch, but an order-sensitive test does."""

    def test_ensure_isco_classification_runs_before_quality_review(self, db):
        client, patchers, call_order, message = _make_client(db, state="COMPLETING")
        try:
            resp, _sid, _db2 = _create_session_and_send(client, message)
            assert resp.status_code == 200

            assert "_ensure_isco_classification" in call_order
            assert "_trigger_quality_review" in call_order
            ensure_idx = call_order.index("_ensure_isco_classification")
            trigger_idx = call_order.index("_trigger_quality_review")
            assert ensure_idx < trigger_idx, (
                f"_trigger_quality_review ran before _ensure_isco_classification "
                f"(order: {call_order}) -- violates the invariant documented at "
                f"survey_routes.py:1224"
            )
        finally:
            for p in patchers:
                p.stop()
