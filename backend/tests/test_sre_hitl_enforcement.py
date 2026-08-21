"""
Tests for Module D Step 5.5: mandatory HITL escalation on HIGH-severity
SemanticCoherence results.

Confirms the real enforcement wiring in backend/api/survey_routes.py's
_send_message_impl (the only live code path that computes
semantic_coherence -- an earlier module, backend/agents/survey_orchestrator.py,
also computed it but was never imported by the live API, confirmed by grep;
wiring enforcement there would have had zero production effect, and that
module was removed from this codebase after that finding was documented).

Uses the REAL SemanticRelationEngine (use_llm=False, fully deterministic,
no network/DB dependency) rather than mocking it, with isco/isic/isced
combinations already validated in eval/sre_expanded_validation.py to
produce known severities -- this tests real engine behavior end-to-end
through the actual API endpoint, not a mocked stand-in.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
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


def _make_client(db, isco_code, isic_section, isced_level, job_title="Assembler", low_isco_confidence=False):
    """Builds a TestClient wired so a message triggers ISCO + ISIC + ISCED
    classification with the given codes, feeding the REAL
    SemanticRelationEngine with exactly this combination."""
    from backend.main import app
    from backend.database.connection import get_db
    from backend.agents.conversation_manager import ConversationContext
    from backend.api import survey_routes as _sr_module

    app.dependency_overrides[get_db] = lambda: db

    # _contexts is a process-global dict keyed by session_id, and this
    # process's real Redis (ContextMemory) may hold leftover data from actual
    # manual app usage under the same low session_ids our in-memory SQLite
    # autoincrements -- both would silently override the mocked context
    # below and were the actual root cause the first time this test was
    # written. Clearing here + mocking _get_context_memory (below) makes each
    # test deterministic regardless of what else has touched this process.
    _sr_module._contexts.clear()

    mock_cm = MagicMock()
    mock_lp = MagicMock()

    def _new_context(sid, lang="en"):
        ctx = ConversationContext(session_id=sid, language=lang)
        # ISIC/ISCED classifiers only run (Stage 4b/4c) when these fields are
        # already collected -- required so isic_result/isced_result are
        # non-None and Stage 4e's SemanticRelationEngine.analyse() actually
        # receives real isic_section/isced_level values instead of None.
        ctx.collected_data["industry"] = "placeholder industry"
        ctx.collected_data["education_level"] = "placeholder education"
        # job_title is deliberately NOT pre-set here: survey_routes.py runs
        # in _FAST_MODE in this test env, which skips NER entirely (see
        # `skip_ner` in _send_message_impl), so the NER JOB_TITLE-entity path
        # never fires. ISCO classification instead has to go through the
        # "Stage 4 fallback" branch, which only runs when "job_title" is a
        # newly-set field this turn (`"job_title" in _fields_new`). Setting
        # it here (before the turn starts) would make it look pre-existing
        # and skip classification entirely -- so it's set inside
        # process_message's side effect below instead, simulating what the
        # real ConversationManager does when it stores a newly-collected field.
        return ctx

    def _process_message(ctx, msg):
        ctx.collected_data["job_title"] = job_title
        return "Thanks, noted."

    mock_cm.new_context.side_effect = _new_context
    mock_cm.process_message.side_effect = _process_message

    mock_lp_result = MagicMock()
    mock_lp_result.detected_language = "en"
    mock_lp_result.is_code_switched = False
    entity = MagicMock()
    entity.text = job_title
    entity.label = "JOB_TITLE"
    mock_lp_result.entities = [entity]
    mock_lp.process.return_value = mock_lp_result

    isco_conf = 0.40 if low_isco_confidence else 0.95
    isco_result_obj = MagicMock(
        primary=MagicMock(code=isco_code, title_en="Test Title", title_ar="عنوان", confidence=isco_conf),
        method="flat_semantic",
        hitl_required=low_isco_confidence,
        reasoning="test reasoning",
        hierarchy_path=None,
        alternatives=[],
        stage_confidences={},
    )

    # ISICResult/ISCEDResult (survey_routes.py) are Pydantic BaseModels built
    # from these classifier outputs' fields directly -- every field the
    # constructors read must be a real str/int/float, not a MagicMock, or
    # validation raises and the whole result is silently dropped by the
    # surrounding try/except.
    isic_result_obj = MagicMock(
        section=isic_section,
        section_title="Test Section",
        division_code="00",
        division_title="Test Division",
        group_code="",
        group_title="",
        class_code="",
        class_title="",
        confidence=0.90,
        method="keyword",
    )
    isced_result_obj = MagicMock(
        level=isced_level,
        level_title="Test Level",
        broad_code="",
        broad_title="",
        narrow_code="",
        narrow_title="",
        detailed_code="",
        detailed_title="",
        confidence=0.90,
        method="keyword",
    )

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
     mock_nat, mock_hitl, mock_ei, _val, _audit) = started

    mock_ctx_mem.return_value.load_session.return_value = None
    mock_isco.return_value.classify.return_value = isco_result_obj
    mock_isic.return_value.classify.return_value = isic_result_obj
    mock_isced.return_value.classify.return_value = isced_result_obj
    mock_nat.return_value.classify.return_value = MagicMock(method="unknown")
    mock_hitl.return_value.review_session.return_value = MagicMock(status=MagicMock(value="pass"))
    mock_ei.return_value.analyze.return_value = MagicMock(state="neutral")

    return TestClient(app), patchers


def _create_session_and_send(client, msg="I assemble electronic components"):
    from backend.database.models import User, SurveySession
    from backend.database.connection import get_db

    # Directly create a user + session (bypassing full OTP flow for test focus)
    db_gen = client.app.dependency_overrides[get_db]
    db = db_gen()
    user = User(email=f"sre-test-{id(client)}@test.invalid")
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


class TestSREHITLEnforcement:
    def test_high_severity_creates_real_hitl_queue_entry(self, db):
        """82 Assembler + ISIC Q (major-8 doesn't include Q) -- confirmed
        HIGH severity in eval/sre_expanded_validation.py."""
        client, patchers = _make_client(db, isco_code="8211", isic_section="Q", isced_level=3)
        try:
            resp, session_id, db2 = _create_session_and_send(client)
            assert resp.status_code == 200

            from backend.database.models import HITLQueue
            rows = db2.query(HITLQueue).filter(HITLQueue.session_id == session_id).all()
            assert len(rows) >= 1, "expected a HITLQueue row for HIGH-severity SRE violation"
            sre_rows = [r for r in rows if r.ai_reasoning and "SRE HIGH-severity" in r.ai_reasoning]
            assert len(sre_rows) == 1, f"expected exactly 1 SRE-tagged row, got {len(sre_rows)}"
            assert sre_rows[0].status == "pending"
            assert sre_rows[0].priority == "HIGH"
        finally:
            for p in patchers:
                p.stop()

    def test_coherent_case_creates_no_sre_escalation(self, db):
        """82 Assembler + ISIC C + ISCED 3 -- confirmed COHERENT in
        eval/sre_expanded_validation.py."""
        client, patchers = _make_client(db, isco_code="8211", isic_section="C", isced_level=3)
        try:
            resp, session_id, db2 = _create_session_and_send(client)
            assert resp.status_code == 200

            from backend.database.models import HITLQueue
            rows = db2.query(HITLQueue).filter(HITLQueue.session_id == session_id).all()
            sre_rows = [r for r in rows if r.ai_reasoning and "SRE HIGH-severity" in r.ai_reasoning]
            assert len(sre_rows) == 0, "coherent case must not trigger an SRE escalation"
        finally:
            for p in patchers:
                p.stop()

    def test_moderate_severity_creates_no_sre_escalation(self, db):
        """22 Health Professional + ISIC J -- confirmed MODERATE severity
        (not HIGH) in eval/sre_expanded_validation.py."""
        client, patchers = _make_client(db, isco_code="2211", isic_section="J", isced_level=7)
        try:
            resp, session_id, db2 = _create_session_and_send(client, msg="I am a doctor")
            assert resp.status_code == 200

            from backend.database.models import HITLQueue
            rows = db2.query(HITLQueue).filter(HITLQueue.session_id == session_id).all()
            sre_rows = [r for r in rows if r.ai_reasoning and "SRE HIGH-severity" in r.ai_reasoning]
            assert len(sre_rows) == 0, "MODERATE severity must not trigger the HIGH-only SRE escalation path"
        finally:
            for p in patchers:
                p.stop()

    def test_no_duplicate_row_when_isco_confidence_already_escalated(self, db):
        """When ISCO confidence is ALSO low (separate, unrelated escalation
        reason), the SRE finding must fold into the SAME row, not create a
        second HITLQueue entry for the same turn."""
        client, patchers = _make_client(
            db, isco_code="8211", isic_section="Q", isced_level=3, low_isco_confidence=True,
        )
        try:
            resp, session_id, db2 = _create_session_and_send(client)
            assert resp.status_code == 200

            from backend.database.models import HITLQueue
            rows = db2.query(HITLQueue).filter(HITLQueue.session_id == session_id).all()
            assert len(rows) == 1, f"expected exactly 1 combined HITLQueue row, got {len(rows)}"
            assert "SRE HIGH-severity" in (rows[0].ai_reasoning or "")
            assert rows[0].priority == "HIGH"
        finally:
            for p in patchers:
                p.stop()
