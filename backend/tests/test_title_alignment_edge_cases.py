"""Previous-occupation decisions and same-turn clarification retain their real fields."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.api import survey_routes as routes
from backend.agents.context_memory import ContextMemory
from backend.agents.conversation_manager import ConversationContext, ConversationManager, ConversationState
from backend.agents.report_generator import ReportGenerator
from backend.agents.semantic_relation import SemanticRelationEngine
from backend.database.models import HITLQueue, SurveySession
from backend.database.response_revisions import active_responses, save_response_revision


class MemoryRedis:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, **kwargs):
        self.values[key] = value


@pytest.fixture
def isolated_agents(monkeypatch):
    manager = object.__new__(ConversationManager)
    manager._agent_available = False
    memory = object.__new__(ContextMemory)
    memory._redis = MemoryRedis()
    memory._ttl = 86400
    routes._contexts.clear()
    monkeypatch.setattr(routes, "_FAST_MODE", True)
    monkeypatch.setattr(routes, "_CREW_CLASSIFICATION_ENABLED", False)
    monkeypatch.setattr(routes, "_COORDINATED_CLASSIFICATION", False)
    monkeypatch.setattr(routes, "_get_agents", lambda: (manager, MagicMock()))
    monkeypatch.setattr(routes, "_get_context_memory", lambda: memory)
    for name in ("_get_audit_logger", "_get_validation_agent", "_get_emotional_intelligence", "_get_person_register_svc"):
        monkeypatch.setattr(routes, name, lambda: MagicMock())
    monkeypatch.setattr(routes, "_get_nationality_classifier", lambda: SimpleNamespace(
        classify=lambda text: SimpleNamespace(method="unknown"),
    ))
    monkeypatch.setattr(routes, "_get_isced_classifier", lambda: SimpleNamespace(
        classify=lambda text: SimpleNamespace(
            level=3, level_title="Upper secondary", broad_code="", broad_title="",
            narrow_code="", narrow_title="", detailed_code="", detailed_title="",
            confidence=0.9, method="rule",
        ),
    ))
    # Neither tested path collects current-industry data.
    monkeypatch.setattr(routes, "_get_isic_classifier", MagicMock(side_effect=AssertionError("Unexpected industry classification")))
    yield manager
    routes._contexts.clear()


def session_at_previous_occupation(db, user, manager, employment_status):
    session = SurveySession(user_id=user.id, language="en")
    db.add(session)
    db.commit()
    collected = {
        "employment_status": employment_status, "education_level": "secondary",
        "job_search_active": "yes", "available_for_work": "yes", "ever_worked": "yes_in_uae",
    }
    fields = manager._get_field_order(collected)
    for field in fields[:fields.index("last_job_title")]:
        collected.setdefault(field, "provided")
    # Keep only fields that belong to this respondent's path.
    collected = {field: value for field, value in collected.items() if field in fields}
    routes._save_context(ConversationContext(
        session_id=session.id, state=ConversationState.COLLECTING_INFO, collected_data=collected,
    ))
    return session


def classification(code, confidence, review_required):
    return SimpleNamespace(
        primary=SimpleNamespace(code=code, title_en="Previous occupation", title_ar="", confidence=confidence),
        method="flat", hitl_required=review_required, stage_confidences={}, hierarchy_path=[],
        alternatives=[], reasoning="synthetic classification",
    )


@pytest.mark.parametrize("employment_status", ["unemployed", "not_in_labour_force"])
def test_previous_occupation_promotion_is_persisted_under_its_real_field(
    db, user, isolated_agents, monkeypatch, employment_status,
):
    session = session_at_previous_occupation(db, user, isolated_agents, employment_status)
    monkeypatch.setattr(routes, "_COORDINATED_CLASSIFICATION", True)
    monkeypatch.setattr(routes, "_get_isco_classifier", lambda: SimpleNamespace(
        classify=lambda *args, **kwargs: classification("2512", 0.4, True),
    ))
    monkeypatch.setattr("backend.agents.cross_standard_coordinator.maybe_revise_isco_with_cross_signal", lambda *args: (
        SimpleNamespace(code="1213", title_en="Director", title_ar="", confidence=0.8),
        "synthetic cross-standard promotion",
    ))
    semantic = SemanticRelationEngine(use_llm=False)
    analyse = MagicMock(wraps=semantic.analyse)
    monkeypatch.setattr("backend.agents.semantic_relation.get_semantic_relation_engine", lambda **kwargs: SimpleNamespace(analyse=analyse))

    result = routes._send_message_impl(session.id, routes.MessageBody(message="Engineer"), db, user)

    responses = active_responses(db, session.id)
    previous_job = next(response for response in responses if response.question_id == "last_job_title")
    assert previous_job.isco_code == result.isco_classifications[0].primary_code == "1213"
    assert not any(response.question_id == "job_title" for response in responses)
    profile = object.__new__(ReportGenerator)._build_profile(responses)
    assert profile.last_job_title == "Engineer"
    assert profile.isco_code == "1213"
    pending = db.query(HITLQueue).filter_by(session_id=session.id, status="pending").all()
    assert len(pending) == 1
    assert pending[0].response_id == previous_job.id
    assert pending[0].ai_code == "1213"
    assert pending[0].raw_text == "Engineer"
    assert analyse.call_args.kwargs["job_title"] == "Engineer"


@pytest.mark.parametrize("employment_status", ["employed", "unemployed", "not_in_labour_force"])
def test_semantic_occupation_review_badge_tracks_pending_and_completed_review(
    db, user, isolated_agents, monkeypatch, employment_status,
):
    field = "job_title" if employment_status == "employed" else "last_job_title"
    if employment_status == "employed":
        session = SurveySession(user_id=user.id, language="en")
        db.add(session)
        db.commit()
        context = context_for_question(isolated_agents, field)
        context.session_id = session.id
        routes._save_context(context)
    else:
        session = session_at_previous_occupation(db, user, isolated_agents, employment_status)
    monkeypatch.setattr(routes, "_get_isco_classifier", lambda: SimpleNamespace(
        classify=lambda *args, **kwargs: classification("2211", 0.95, False),
    ))
    semantic = SemanticRelationEngine(use_llm=False)
    monkeypatch.setattr("backend.agents.semantic_relation.get_semantic_relation_engine", lambda **kwargs: semantic)

    result = routes._send_message_impl(session.id, routes.MessageBody(message="Doctor"), db, user)

    assert any(violation["severity"] == "HIGH" for violation in result.semantic_coherence["violations"])
    occupation = next(response for response in active_responses(db, session.id) if response.question_id == field)
    pending = db.query(HITLQueue).filter_by(session_id=session.id, status="pending").all()
    assert len(pending) == 1
    assert pending[0].response_id == occupation.id
    assert pending[0].raw_text == "Doctor"
    assert result.isco_classifications[0].confidence == 0.95
    assert result.isco_classifications[0].hitl_required is True

    replay = routes._send_message_impl(session.id, routes.MessageBody(message="idk"), db, user)
    assert replay.isco_classifications[0].method == "cached"
    assert replay.isco_classifications[0].hitl_required is True

    monkeypatch.setenv("HITL_REVIEWER_USER_IDS", str(user.id))
    routes._submit_hitl_review_impl(routes.HITLReviewBody(
        escalation_id=pending[0].id, action="approve",
    ), db, user)
    reviewed = routes._send_message_impl(session.id, routes.MessageBody(message="idk"), db, user)
    assert reviewed.isco_classifications[0].method == "cached"
    assert reviewed.isco_classifications[0].hitl_required is False
    assert not db.query(HITLQueue).filter_by(session_id=session.id, status="pending").all()


@pytest.mark.parametrize("employment_status", ["employed", "unemployed", "not_in_labour_force"])
@pytest.mark.parametrize("review_status", ["pending", "reviewed"])
@pytest.mark.parametrize("linked", [True, False])
def test_cached_occupation_badge_uses_matching_review_instead_of_confidence(
    db, user, isolated_agents, monkeypatch, employment_status, review_status, linked,
):
    field = "job_title" if employment_status == "employed" else "last_job_title"
    session = SurveySession(user_id=user.id, language="en")
    db.add(session)
    db.commit()
    context = context_for_question(isolated_agents, field)
    context.session_id = session.id
    context.collected_data["employment_status"] = employment_status
    context.collected_data[field] = "Office clerk"
    routes._save_context(context)
    occupation = save_response_revision(
        db, session.id, field, "Office clerk", isco_code="4110", confidence_score=0.4,
    )
    review = HITLQueue(
        session_id=session.id, response_id=occupation.id if linked else None,
        raw_text="Office clerk", ai_code="4110", ai_confidence=0.4,
        status=review_status, priority="MEDIUM", created_at=datetime.utcnow(),
    )
    # A legacy queue item for a different answer must not flag this occupation.
    unrelated = HITLQueue(
        session_id=session.id, response_id=None, raw_text="Doctor", ai_code="2211",
        ai_confidence=0.3, status="pending", priority="HIGH", created_at=datetime.utcnow(),
    )
    db.add_all([review, unrelated])
    db.commit()
    semantic = SemanticRelationEngine(use_llm=False)
    monkeypatch.setattr("backend.agents.semantic_relation.get_semantic_relation_engine", lambda **kwargs: semantic)

    result = routes._send_message_impl(session.id, routes.MessageBody(message="idk"), db, user)

    assert result.isco_classifications[0].method == "cached"
    assert result.isco_classifications[0].confidence == 0.4
    assert result.isco_classifications[0].hitl_required is (review_status == "pending")


@pytest.mark.parametrize("language, answer", [
    ("en", "I have not completed a bachelor's degree."),
    ("ar", "لم أكمل بكالوريوس"),
    ("ur", "میں نے بیچلر ڈگری مکمل نہیں کی"),
    ("hi", "मैंने स्नातक की डिग्री पूरी नहीं की है"),
    ("tl", "Hindi ako nakatapos ng batsilyer"),
])
def test_clarification_exposes_current_question_for_next_canonical_answer(
    db, user, isolated_agents, language, answer,
):
    session = SurveySession(user_id=user.id, language=language)
    db.add(session)
    db.commit()
    routes._save_context(ConversationContext(
        session_id=session.id, language=language, state=ConversationState.COLLECTING_INFO,
        collected_data={"employment_status": "employed"},
    ))

    clarification = routes._send_message_impl(session.id, routes.MessageBody(message=answer), db, user)

    assert clarification.state == "clarifying"
    assert clarification.next_field == clarification.survey_progress.current_field == "education_level"
    assert "education_level" not in clarification.collected_data

    corrected = routes._send_message_impl(session.id, routes.MessageBody(
        message="Bachelor's degree", answer_field="education_level", answer_value="bachelor",
    ), db, user)

    assert corrected.collected_data["education_level"] == "bachelor"
    assert corrected.next_field == "field_of_study"
    assert corrected.state == "collecting_info"


@pytest.mark.parametrize("employment_status", ["employed", "unemployed", "not_in_labour_force"])
def test_failed_changed_occupation_does_not_replay_previous_answers_code(
    db, user, isolated_agents, monkeypatch, employment_status,
):
    manager = isolated_agents
    field = "job_title" if employment_status == "employed" else "last_job_title"
    if employment_status == "employed":
        session = SurveySession(user_id=user.id, language="en")
        db.add(session)
        db.commit()
        collected = {
            "employment_status": "employed", "education_level": "secondary",
            "employment_nature": "paid_employee",
        }
        fields = manager._get_field_order(collected)
        for name in fields[:fields.index(field)]:
            collected.setdefault(name, "provided")
        context = ConversationContext(session_id=session.id, collected_data=collected)
    else:
        session = session_at_previous_occupation(db, user, manager, employment_status)
        saved = routes._get_or_create_context(manager, session.id, session.language, db=db)
        context = saved
    context.state = ConversationState.VALIDATING
    context.collected_data[field] = "Engineer"
    routes._save_context(context)
    original = save_response_revision(
        db, session.id, field, "Engineer", isco_code="2512", confidence_score=0.8,
    )
    db.commit()
    failing = MagicMock(side_effect=RuntimeError("synthetic classifier unavailable"))
    monkeypatch.setattr(routes, "_get_isco_classifier", lambda: SimpleNamespace(classify=failing))

    result = routes._send_message_impl(session.id, routes.MessageBody(
        message="Correct occupation to Doctor", correction_field=field, correction_value="Doctor",
    ), db, user)

    failing.assert_called_once()
    assert result.collected_data[field] == "Doctor"
    assert result.isco_classifications == []
    current = next(response for response in active_responses(db, session.id) if response.question_id == field)
    assert current.answer == "Doctor"
    assert current.isco_code is None
    assert original.deleted_at is not None


def context_for_question(manager, field, language="en"):
    status = "unemployed" if field in ("ever_worked", "job_search_active", "available_for_work", "last_job_title") else "employed"
    baseline = {
        "employment_status": status, "education_level": "secondary", "employment_nature": "paid_employee",
        "secondary_job": "no", "platform_work": "no", "job_search_active": "yes",
        "available_for_work": "yes", "ever_worked": "yes_in_uae",
    }
    fields = manager._get_field_order(baseline)
    data = {name: baseline.get(name, "provided") for name in fields[:fields.index(field)]}
    return ConversationContext(
        session_id=1, language=language, state=ConversationState.COLLECTING_INFO, collected_data=data,
    )


@pytest.mark.parametrize("field, language, ambiguous, canonical", [
    ("employment_status", "en", "I am employed or unemployed", "employed"),
    ("education_level", "hi", "मैंने स्नातक की डिग्री पूरी नहीं की है", "bachelor"),
    ("employment_nature", "en", "I am a paid employee or self-employed", "paid_employee"),
    ("secondary_job", "en", "yes or no", "yes"),
    ("platform_work", "en", "yes_primary or no", "yes_primary"),
    ("ever_worked", "en", "yes_in_uae or never_worked", "yes_in_uae"),
    ("job_search_active", "en", "yes or no", "yes"),
    ("available_for_work", "en", "yes or no", "yes"),
])
def test_exhausted_clarification_never_stores_ambiguous_routing_gate(
    isolated_agents, field, language, ambiguous, canonical,
):
    manager = isolated_agents
    context = context_for_question(manager, field, language)
    original_answers = dict(context.collected_data)

    for _ in range(6):
        manager._transition(context, ambiguous, "")
        assert context.state == ConversationState.CLARIFYING
        assert context.clarification_target == field
        assert context.collected_data == original_answers

    manager._transition(context, canonical, "", structured_answer=(field, canonical))

    assert context.collected_data[field] == canonical
    assert context.state == ConversationState.COLLECTING_INFO
    assert context.clarification_target is None
    assert context.clarification_count == 0
    if field == "education_level":
        assert "field_of_study" in manager._get_field_order(context.collected_data)
    if field == "ever_worked":
        assert "last_job_title" in manager._get_field_order(context.collected_data)


@pytest.mark.parametrize("field", ["gender", "monthly_wage_range"])
def test_supported_refusal_choices_are_still_accepted(isolated_agents, field):
    manager = isolated_agents
    context = context_for_question(manager, field)

    manager._transition(context, "Prefer not to say", "")

    assert context.collected_data[field] == "prefer_not_to_say"
    assert context.state == ConversationState.COLLECTING_INFO


def test_exhausted_clarification_retains_open_text_fallback(isolated_agents):
    manager = isolated_agents
    context = context_for_question(manager, "main_skills")

    for _ in range(4):
        manager._transition(context, "idk", "")

    assert context.collected_data["main_skills"] == "idk"
    assert context.state == ConversationState.COLLECTING_INFO
    assert context.clarification_target is None
