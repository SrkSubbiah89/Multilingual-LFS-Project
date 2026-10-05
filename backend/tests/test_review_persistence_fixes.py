"""Regression tests for authorization, revisions, resume and report freshness."""
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from backend.api import survey_routes as routes
from backend.api import auth_routes
from backend.agents.context_memory import ContextMemory
from backend.agents.conversation_manager import ConversationContext, ConversationManager, ConversationState
from backend.agents.hitl_quality_manager import HITLQualityManager
from backend.agents.report_generator import ReportGenerator
from backend.database.models import HITLQueue, SurveySession, SurveyResponse, SurveyReportRecord, QualityReview
from backend.database.response_revisions import active_responses, save_response_revision


class MemoryRedis:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, **kwargs):
        self.values[key] = value

    def lock(self, *args, **kwargs):
        return MagicMock(acquire=MagicMock(return_value=True))


@pytest.fixture
def isolated_routes(monkeypatch):
    manager = object.__new__(ConversationManager)
    manager._agent_available = False
    memory = object.__new__(ContextMemory)
    memory._redis = MemoryRedis()
    memory._ttl = 86400
    routes._contexts.clear()
    monkeypatch.setattr(routes, "_get_agents", lambda: (manager, MagicMock()))
    monkeypatch.setattr(routes, "_get_context_memory", lambda: memory)
    monkeypatch.setattr(routes, "_get_audit_logger", lambda: MagicMock())
    monkeypatch.setattr(routes, "_get_validation_agent", lambda: MagicMock())
    monkeypatch.setattr(routes, "_get_emotional_intelligence", lambda: MagicMock())
    monkeypatch.setattr(routes, "_get_person_register_svc", lambda: MagicMock())
    monkeypatch.setattr(routes, "_FAST_MODE", True)
    yield manager, memory
    routes._contexts.clear()


def make_session(db, user, **kwargs):
    row = SurveySession(user_id=user.id, language="en", **kwargs)
    db.add(row)
    db.commit()
    return row


def make_queue(db, session, response=None):
    item = HITLQueue(session_id=session.id, response_id=response.id if response else None,
        raw_text="Engineer", ai_code="2512", ai_confidence=0.4,
        priority="HIGH", status="pending", created_at=datetime.utcnow())
    db.add(item)
    db.commit()
    return item


def make_report(db, session):
    row = SurveyReportRecord(session_id=session.id, language="en", profile_json='{"isco_code":"2512"}',
        report_en="report", report_ar="report", recommendations_en="review", recommendations_ar="review",
        generated_at=datetime.utcnow(), quality_status="fail", semantic_coherence_json='{"score":0.4}')
    db.add(row)
    db.commit()
    return row


def test_ordinary_users_cannot_access_foreign_reviews(db, user, other_user, monkeypatch):
    monkeypatch.setenv("HITL_REVIEWER_USER_IDS", "")
    session = make_session(db, user)
    item = make_queue(db, session)
    with pytest.raises(HTTPException) as error:
        routes.get_hitl_queue(db=db, current_user=other_user)
    assert error.value.status_code == 403
    with pytest.raises(HTTPException) as error:
        routes.submit_hitl_review(routes.HITLReviewBody(escalation_id=item.id, action="correct", code="1213"), db, other_user)
    assert error.value.status_code == 403
    assert item.status == "pending"


def test_reviewer_scope_excludes_deleted_sessions(db, user, other_user, monkeypatch, isolated_routes):
    monkeypatch.setenv("HITL_REVIEWER_USER_IDS", str(other_user.id))
    session = make_session(db, user, deleted_at=datetime.utcnow())
    item = make_queue(db, session)
    assert routes.get_hitl_queue(db=db, current_user=other_user) == []
    with pytest.raises(HTTPException) as error:
        routes.submit_hitl_review(routes.HITLReviewBody(escalation_id=item.id, action="approve"), db, other_user)
    assert error.value.status_code == 404


@pytest.mark.parametrize("linked", [True, False])
def test_human_correction_saves_revision_invalidates_cache_reassesses_quality(db, user, other_user, monkeypatch, isolated_routes, linked):
    monkeypatch.setenv("HITL_REVIEWER_USER_IDS", str(other_user.id))
    session = make_session(db, user, status="completed")
    original = save_response_revision(db, session.id, "job_title", "Engineer", isco_code="2512", confidence_score=0.4)
    item = make_queue(db, session, original if linked else None)
    report = make_report(db, session)
    result = routes.submit_hitl_review(routes.HITLReviewBody(escalation_id=item.id, action="correct", code="1213"), db, other_user)
    assert result.final_code == "1213"
    current = active_responses(db, session.id)[0]
    assert current.isco_code == "1213"
    assert current.supersedes_id == original.id
    assert original.deleted_at is not None
    assert item.response_id == current.id
    assert item.status == "reviewed" and item.reviewed_by == other_user.id
    assert report.invalidated_at is not None
    generator = object.__new__(ReportGenerator)
    assert generator._load_existing(db, session.id) is None
    assert generator._build_profile(active_responses(db, session.id)).isco_code == "1213"
    quality = db.query(QualityReview).filter_by(session_id=session.id).order_by(QualityReview.id.desc()).first()
    assert quality.passed and quality.flagged_count == 0


def test_deleted_and_superseded_answers_are_not_prefilled(db, user, isolated_routes):
    old = make_session(db, user, status="completed", completed_at=datetime(2026, 1, 1))
    original = save_response_revision(db, old.id, "job_title", "Engineer")
    save_response_revision(db, old.id, "job_title", "Doctor")
    deleted = make_session(db, user, status="completed", completed_at=datetime(2026, 2, 1), deleted_at=datetime.utcnow())
    save_response_revision(db, deleted.id, "job_title", "Deleted title")
    db.commit()
    new = routes.create_session(routes.SessionCreateBody(language="en"), db, user)
    assert routes._contexts[new["id"]].collected_data["job_title"] == "Doctor"
    assert original.deleted_at is not None


def test_completion_preserves_classification_and_versions_changed_answers(db, user):
    session = make_session(db, user)
    original = save_response_revision(db, session.id, "job_title", "Engineer", isco_code="2512", confidence_score=0.95)
    routes._persist_collected_data(db, session.id, {"job_title": "Engineer", "employment_status": "employed"})
    assert active_responses(db, session.id)[0].id == original.id
    routes._persist_collected_data(db, session.id, {"job_title": "Doctor", "employment_status": "employed"})
    title = next(row for row in active_responses(db, session.id) if row.question_id == "job_title")
    assert title.answer == "Doctor" and title.supersedes_id == original.id
    assert title.isco_code is None


def test_language_changes_never_submit_synthetic_answers(db, user, isolated_routes):
    session = make_session(db, user)
    snapshot = routes.get_conversation(session.id, db, user)
    assert snapshot["state"] == "collecting_info" and snapshot["next_field"] == "employment_status"
    for language in ("hi", "ur", "ar", "en"):
        snapshot = routes.update_conversation_language(session.id, routes.LanguageUpdateBody(language=language), db, user)
        assert snapshot["collected_data"] == {}
        assert snapshot["next_field"] == "employment_status"
        assert len(snapshot["history"]) == 1
    assert not any(turn["role"] == "user" for turn in snapshot["history"])


@pytest.mark.parametrize("saved_status,canonical,next_field", [
    ("hello", None, "employment_status"),
    ("नियोजित", "employed", "education_level"),
])
def test_resume_from_database_repairs_old_routing_answers(db, user, isolated_routes, saved_status, canonical, next_field):
    session = make_session(db, user)
    save_response_revision(db, session.id, "employment_status", saved_status)
    db.commit()
    snapshot = routes.get_conversation(session.id, db, user)
    assert snapshot["collected_data"].get("employment_status") == canonical
    assert snapshot["next_field"] == next_field
    assert snapshot["state"] == "collecting_info"


def test_repaired_resume_replaces_obsolete_validation_prompt(db, user, isolated_routes):
    session = make_session(db, user)
    routes._save_context(ConversationContext(session_id=session.id, state=ConversationState.VALIDATING,
        collected_data={"employment_status": "hello", "education_level": "secondary"},
        history=[{"role": "assistant", "content": "Confirm all these answers?"}]))
    snapshot = routes.get_conversation(session.id, db, user)
    assert snapshot["next_field"] == "employment_status"
    assert snapshot["reply"] != "Confirm all these answers?"
    assert snapshot["history"][-1]["content"] == snapshot["reply"]


@pytest.mark.parametrize("corrupt_cache", [False, True])
def test_direct_message_recovers_saved_answers_after_cache_loss(db, user, isolated_routes, monkeypatch, corrupt_cache):
    session = make_session(db, user)
    for field, value in (("employment_status", "employed"), ("education_level", "bachelor")):
        save_response_revision(db, session.id, field, value)
    db.commit()
    manager, memory = isolated_routes
    # A stale process-local context must not win over the database fallback.
    routes._contexts[session.id] = manager.new_context(session.id)
    if corrupt_cache:
        memory._redis.set(f"lfs:session:{session.id}", "{broken-json")
    for name in ("_get_isced_classifier", "_get_isic_classifier", "_get_nationality_classifier"):
        monkeypatch.setattr(routes, name, lambda: MagicMock())
    result = routes._send_message_impl(session.id, routes.MessageBody(message="Mechanical Engineering"), db, user)
    assert result.collected_data["employment_status"] == "employed"
    assert result.collected_data["education_level"] == "bachelor"
    assert result.collected_data["field_of_study"] == "Mechanical Engineering"
    saved = {row.question_id: row.answer for row in active_responses(db, session.id)}
    assert saved["employment_status"] == "employed"
    assert saved["education_level"] == "bachelor"


@pytest.mark.parametrize("body", [dict(answer_field="employment_status"), dict(answer_value="employed")])
def test_structured_answer_requires_both_parts(body):
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="must be supplied together"):
        routes.MessageBody(message="Employed", **body)


def test_resume_restores_history_answers_and_clarification(db, user, isolated_routes):
    session = make_session(db, user)
    context = ConversationContext(session_id=session.id, state=ConversationState.CLARIFYING,
        clarification_target="employment_status", clarification_count=2,
        history=[{"role": "assistant", "content": "Please clarify"}])
    routes._save_context(context)
    routes._contexts.clear()
    snapshot = routes.get_conversation(session.id, db, user)
    restored = routes._contexts[session.id]
    assert snapshot["history"] == context.history
    assert restored.clarification_count == 2
    assert restored.clarification_target == snapshot["next_field"] == "employment_status"


def test_worker_refresh_reads_other_workers_latest_answer(isolated_routes):
    manager, memory = isolated_routes
    first = ConversationContext(session_id=42, state=ConversationState.COLLECTING_INFO,
        collected_data={"employment_status": "employed"})
    routes._contexts[42] = first
    memory.save_session(42, "collecting_info", "en", {"employment_status": "employed", "education_level": "bachelor"}, [])
    fresh = routes._get_or_create_context(manager, 42, "en")
    assert fresh.collected_data["education_level"] == "bachelor"
    assert ConversationManager._get_field_order(fresh.collected_data)[2] == "field_of_study"


@pytest.mark.parametrize("change", ["job_title", "job_duties"])
def test_api_refinement_and_title_correction_persist_the_returned_code(db, user, isolated_routes, monkeypatch, change):
    session = make_session(db, user)
    manager, _ = isolated_routes
    collected = {"employment_status": "employed", "education_level": "secondary", "employment_nature": "paid_employee"}
    fields = manager._get_field_order(collected)
    for field in fields[:fields.index("job_duties")]:
        collected.setdefault(field, "provided")
    collected["job_title"] = "Engineer"
    original = save_response_revision(db, session.id, "job_title", "Engineer", isco_code="2512", confidence_score=0.9)
    db.commit()
    state = ConversationState.VALIDATING if change == "job_title" else ConversationState.COLLECTING_INFO
    ctx = ConversationContext(session_id=session.id, state=state, collected_data=collected)
    routes._save_context(ctx)
    classification = SimpleNamespace(primary=SimpleNamespace(code="1213", title_en="Managers", title_ar="", confidence=0.8),
        method="flat", hitl_required=False, stage_confidences={}, hierarchy_path=[], alternatives=[], reasoning="test")
    monkeypatch.setattr(routes, "_get_isco_classifier", lambda: SimpleNamespace(classify=lambda *args, **kwargs: classification))
    for name in ("_get_isced_classifier", "_get_isic_classifier", "_get_nationality_classifier"):
        monkeypatch.setattr(routes, name, lambda: MagicMock())
    body = routes.MessageBody(message="Doctor", correction_field="job_title", correction_value="Doctor") if change == "job_title" else routes.MessageBody(message="I direct policy and coordinate teams")
    result = routes._send_message_impl(session.id, body, db, user)
    assert result.isco_classifications[0].primary_code == "1213"
    saved = next(row for row in active_responses(db, session.id) if row.question_id == "job_title")
    assert saved.isco_code == "1213" and saved.supersedes_id == original.id
    assert saved.answer == ("Doctor" if change == "job_title" else "Engineer")
    next_result = routes._send_message_impl(session.id, routes.MessageBody(message="yes"), db, user)
    assert next_result.isco_classifications[0].primary_code == saved.isco_code


def test_missing_required_employed_occupation_still_escalates(db, user):
    session = make_session(db, user)
    save_response_revision(db, session.id, "employment_status", "employed")
    flagged, metrics, score, result = HITLQualityManager.assess_responses(session.id, active_responses(db, session.id))
    assert metrics.missing_isco_count == 1
    assert flagged[0].question_id == "job_title" and result.value == "escalated"


def test_unflushed_queue_is_retired_with_its_revision(db, user):
    assert not db.autoflush
    session = make_session(db, user)
    original = save_response_revision(db, session.id, "job_title", "Engineer", isco_code="2512", confidence_score=0.4)
    queue = HITLQueue(session_id=session.id, response_id=original.id, raw_text="Engineer",
        ai_code="2512", ai_confidence=0.4, priority="HIGH", status="pending", created_at=datetime.utcnow())
    db.add(queue)
    revised = save_response_revision(db, session.id, "job_title", "Engineer", isco_code="1213", confidence_score=0.8)
    assert original.deleted_at is not None and revised.id != original.id
    assert queue.status == "rejected"


def test_raw_response_writes_version_answers_and_invalidate_completed_reports(db, user, isolated_routes):
    session = make_session(db, user)
    first = routes.submit_response(session.id, routes.ResponseSubmitBody(question_id="job_title", answer="Engineer", isco_code="2512", confidence_score=0.4), db, user)
    second = routes.submit_response(session.id, routes.ResponseSubmitBody(question_id="job_title", answer="Doctor", isco_code="2221", confidence_score=0.8), db, user)
    assert first.deleted_at is not None
    assert len(active_responses(db, session.id)) == 1
    report = make_report(db, session)
    session.status = "completed"
    db.commit()
    corrected = routes.update_response(session.id, second.id, routes.ResponseSubmitBody(question_id="job_title", answer="Director", isco_code="1213", confidence_score=0.95), db, user)
    assert corrected.supersedes_id == second.id
    assert report.invalidated_at is not None
    assert db.query(QualityReview).filter_by(session_id=session.id).count() == 1
    assert routes._contexts[session.id].collected_data["job_title"] == "Director"


def test_raw_patch_cannot_move_answer_to_another_question(db, user, isolated_routes):
    session = make_session(db, user)
    original = routes.submit_response(session.id, routes.ResponseSubmitBody(question_id="job_title", answer="Engineer"), db, user)
    with pytest.raises(HTTPException) as error:
        routes.update_response(session.id, original.id, routes.ResponseSubmitBody(question_id="industry", answer="Technology"), db, user)
    assert error.value.status_code == 422
    assert original.deleted_at is None
    assert routes._contexts[session.id].collected_data["job_title"] == "Engineer"
    assert "industry" not in routes._contexts[session.id].collected_data


def test_report_preserves_redis_failure_status(db, user, isolated_routes, monkeypatch):
    import redis
    session = make_session(db, user, status="completed")
    manager, memory = isolated_routes
    memory._redis.lock = MagicMock(side_effect=redis.exceptions.ConnectionError("offline"))
    generator = MagicMock()
    monkeypatch.setattr(routes, "get_report_generator", lambda: generator)
    with pytest.raises(HTTPException) as error:
        routes.get_report(session.id, db=db, current_user=user)
    assert error.value.status_code == 503
    generator.generate.assert_not_called()


def test_report_rechecks_deletion_after_lock_acquisition(db, user, isolated_routes, monkeypatch):
    session = make_session(db, user, status="completed")
    @contextmanager
    def deleted_before_acquiring(session_id):
        session.deleted_at = datetime.utcnow()
        db.commit()
        yield
    monkeypatch.setattr(routes, "_session_turn", deleted_before_acquiring)
    generator = MagicMock()
    monkeypatch.setattr(routes, "get_report_generator", lambda: generator)
    with pytest.raises(HTTPException) as error:
        routes.get_report(session.id, db=db, current_user=user)
    assert error.value.status_code == 404
    generator.generate.assert_not_called()


def test_lost_lease_prevents_raw_answer_commit(db, user, isolated_routes, monkeypatch):
    session = make_session(db, user)
    manager, memory = isolated_routes
    lock = MagicMock(acquire=MagicMock(return_value=True))
    memory._redis.lock = MagicMock(return_value=lock)
    lock.owned.return_value = False
    commit = MagicMock(wraps=db.commit)
    monkeypatch.setattr(db, "commit", commit)
    with pytest.raises(HTTPException) as error:
        routes.submit_response(session.id, routes.ResponseSubmitBody(question_id="education_level", answer="bachelor"), db, user)
    assert error.value.status_code == 503
    commit.assert_not_called()
    db.rollback()
    assert active_responses(db, session.id) == []
    assert session.id not in routes._contexts


def test_coordinator_promotion_saves_code_and_replaces_unflushed_queue(db, user, isolated_routes, monkeypatch):
    session = make_session(db, user)
    manager, _ = isolated_routes
    collected = {"employment_status": "employed", "education_level": "secondary", "employment_nature": "paid_employee"}
    fields = manager._get_field_order(collected)
    for field in fields[:fields.index("job_title")]:
        collected.setdefault(field, "provided")
    routes._save_context(ConversationContext(session_id=session.id, state=ConversationState.COLLECTING_INFO, collected_data=collected))
    classification = SimpleNamespace(primary=SimpleNamespace(code="2512", title_en="Engineer", title_ar="", confidence=0.4),
        method="flat", hitl_required=True, stage_confidences={}, hierarchy_path=[], alternatives=[], reasoning="initial")
    monkeypatch.setattr(routes, "_get_isco_classifier", lambda: SimpleNamespace(classify=lambda *args, **kwargs: classification))
    for name in ("_get_isced_classifier", "_get_isic_classifier", "_get_nationality_classifier"):
        monkeypatch.setattr(routes, name, lambda: MagicMock())
    monkeypatch.setattr(routes, "_COORDINATED_CLASSIFICATION", True)
    monkeypatch.setattr("backend.agents.cross_standard_coordinator.maybe_revise_isco_with_cross_signal", lambda *args: (
        SimpleNamespace(code="1213", title_en="Director", title_ar="", confidence=0.8), "cross-standard promotion"))
    result = routes._send_message_impl(session.id, routes.MessageBody(message="Engineer"), db, user)
    occupation = next(row for row in active_responses(db, session.id) if row.question_id == "job_title")
    assert result.isco_classifications[0].primary_code == occupation.isco_code == "1213"
    pending = db.query(HITLQueue).filter_by(session_id=session.id, status="pending").all()
    assert len(pending) == 1 and pending[0].ai_code == "1213"
    assert pending[0].response_id == occupation.id


def test_report_reads_authoritative_review_state_and_json(db, user):
    session = make_session(db, user, status="completed")
    response = save_response_revision(db, session.id, "job_title", "Engineer", isco_code="2512", confidence_score=0.95)
    report = make_report(db, session)
    queue = make_queue(db, session, response)
    generator = object.__new__(ReportGenerator)
    result = generator._record_to_report(report, session)
    assert result.pending_review and result.semantic_coherence == {"score": 0.4}
    queue.status = "reviewed"
    queue.reviewed_by = user.id
    result = generator._record_to_report(report, session)
    assert not result.pending_review and result.human_review_status == "reviewed"


def test_never_worked_is_not_a_missing_occupation(db, user):
    session = make_session(db, user)
    save_response_revision(db, session.id, "employment_status", "not_in_labour_force")
    save_response_revision(db, session.id, "last_job_title", "never_worked")
    flagged, metrics, score, result = HITLQualityManager.assess_responses(session.id, active_responses(db, session.id))
    assert not flagged and metrics.missing_isco_count == 0
    assert result.value == "pass"
    assert metrics.avg_confidence == 0.0  # no invented classification confidence


@pytest.mark.parametrize("limited_key", ["sms_otp_req_ip:", "sms_otp_req_phone:"])
def test_sms_limits_precede_account_creation_and_delivery(db, monkeypatch, limited_key):
    delivery = MagicMock()
    monkeypatch.setattr(auth_routes, "generate_and_send_sms_otp", delivery)
    keys = []
    def rate_limit(key, **kwargs):
        keys.append(key)
        return not key.startswith(limited_key)
    monkeypatch.setattr(auth_routes, "check_rate_limit", rate_limit)
    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.2"))
    with pytest.raises(HTTPException) as error:
        auth_routes.request_sms_otp(auth_routes.SMSOTPRequestBody(phone="050 123 4567"), request, db)
    assert error.value.status_code == 429
    assert not delivery.called
    assert db.query(routes.User).count() == 0
    if limited_key.endswith("phone:"):
        assert keys[-1] == "sms_otp_req_phone:+971501234567"
