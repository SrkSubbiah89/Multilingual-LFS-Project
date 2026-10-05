"""Live-route wiring preserves specialist evidence and same-turn language."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.api import survey_routes as routes
from backend.agents.conversation_manager import ConversationContext, ConversationState
from backend.agents.survey_classification_crew import SurveyClassificationResult
from backend.database.models import HITLQueue, SurveyResponse, SurveySession


@pytest.fixture
def turn(monkeypatch, db, user):
    session = SurveySession(user_id=user.id, language="en", status="in_progress")
    db.add(session)
    db.commit()
    context = ConversationContext(session_id=session.id, language="en",
        state=ConversationState.COLLECTING_INFO,
        collected_data={"employment_status": "employed", "education_level": "bachelor"})
    manager = MagicMock()
    manager.process_message.side_effect = lambda ctx, message, **kwargs: "Next question"
    processor = MagicMock()
    processor.process.return_value = routes._empty_lp_result("answer", "en")
    monkeypatch.setattr(routes, "_get_agents", lambda: (manager, processor))
    monkeypatch.setattr(routes, "_get_or_create_context", lambda *args, **kwargs: context)
    monkeypatch.setattr(routes, "_get_context_memory", lambda: MagicMock())
    monkeypatch.setattr(routes, "_get_audit_logger", lambda: MagicMock())
    monkeypatch.setattr(routes, "_get_emotional_intelligence", lambda: SimpleNamespace(
        analyze=lambda *args: SimpleNamespace(state="neutral")))
    monkeypatch.setattr(routes, "_save_context", lambda ctx: None)
    monkeypatch.setattr(routes, "_FAST_MODE", False)
    monkeypatch.setattr(routes, "_CREW_CLASSIFICATION_ENABLED", True)
    isco = SimpleNamespace(primary=SimpleNamespace(code="2512", title_en="Software developers",
        title_ar="", confidence=0.4), method="hierarchical_semantic", stage_confidences={},
        hierarchy_path=["2", "25", "251", "2512"], hitl_required=True,
        alternatives=[], reasoning="Synthetic specialist evidence")
    isic = SimpleNamespace(section="J", section_title="Information", division_code="62",
        division_title="Programming", group_code="620", group_title="Programming",
        class_code="6201", class_title="Programming", confidence=0.9, method="keyword")
    isced = SimpleNamespace(level=6, level_title="Bachelor", broad_code="06", broad_title="ICT",
        narrow_code="061", narrow_title="ICT", detailed_code="0613", detailed_title="Software",
        confidence=0.9, method="keyword")
    classifiers = {"isco": MagicMock(classify=MagicMock(return_value=isco)),
        "isic": MagicMock(classify=MagicMock(return_value=isic)),
        "isced": MagicMock(classify=MagicMock(return_value=isced))}
    for dimension in classifiers:
        monkeypatch.setattr(routes, f"_get_{dimension}_classifier", lambda dim=dimension: classifiers[dim])
    return session, context, manager, processor, classifiers, (isco, isic, isced)


def test_detected_language_applies_before_conversation_reply(turn, db, user):
    session, context, manager, processor, _, _ = turn
    processor.process.return_value = routes._empty_lp_result("नमस्ते यह मेरा उत्तर है", "hi")
    manager.process_message.side_effect = lambda ctx, message, **kwargs: f"reply-language={ctx.language}"

    response = routes._send_message_impl(session.id, routes.MessageBody(message="नमस्ते यह मेरा उत्तर है"), db, user)

    assert response.reply == "reply-language=hi"
    assert session.language == context.language == "hi"


def test_explicit_language_takes_precedence_on_same_turn(turn, db, user):
    session, context, manager, processor, _, _ = turn
    processor.process.return_value = routes._empty_lp_result("some answer", "en")
    manager.process_message.side_effect = lambda ctx, message, **kwargs: f"reply-language={ctx.language}"

    response = routes._send_message_impl(session.id,
        routes.MessageBody(message="some answer", preferred_language="ur"), db, user)

    assert response.reply == "reply-language=ur"
    assert session.language == context.language == "ur"


def test_changed_classification_inputs_use_one_crew_and_preserve_results(turn, db, user, monkeypatch):
    session, context, manager, _, classifiers, values = turn
    context.collected_data["job_title"] = "Software developer"
    def answer(ctx, message, **kwargs):
        ctx.collected_data["industry"] = message
        return "Next question"
    manager.process_message.side_effect = answer
    calls = []
    class Crew:
        def __init__(self, **dependencies):
            self.dependencies = dependencies

        def classify(self, **request):
            calls.append(request)
            occupation = self.dependencies["isco_classifier"].classify(request["job_title"], context=request["isco_context"])
            industry = self.dependencies["isic_classifier"].classify(request["industry_text"])
            return SurveyClassificationResult(isco=occupation, isic=industry, execution={
                "mode": "crewai_sequential", "crew_attempted": True, "crew_completed": True,
                "requested_dimensions": ["isco", "isic"], "audit_tool_executed": True,
                "cooperation_verified": True,
            })
    monkeypatch.setattr("backend.agents.survey_classification_crew.SurveyClassificationCrew", Crew)

    response = routes._send_message_impl(session.id, routes.MessageBody(message="software company"), db, user)

    assert len(calls) == 1
    assert calls[0]["job_title"] == "Software developer"
    assert "software company" in calls[0]["isco_context"]
    assert response.isco_classifications[0].confidence == 0.4
    assert response.isco_classifications[0].hitl_required is True
    assert response.isco_classifications[0].hierarchy_path == values[0].hierarchy_path
    assert response.isic_classification.class_code == "6201"
    assert response.classification_execution["cooperation_verified"] is True
    assert response.agent_execution["ClassificationEvidenceAuditor"] == "completed"
    classifiers["isco"].classify.assert_called_once()
    classifiers["isic"].classify.assert_called_once()
    occupation = db.query(SurveyResponse).filter_by(session_id=session.id, question_id="job_title", deleted_at=None).one()
    assert (occupation.answer, occupation.isco_code, occupation.confidence_score) == ("Software developer", "2512", 0.4)
    assert db.query(HITLQueue).filter_by(response_id=occupation.id, status="pending").count() == 1


@pytest.mark.parametrize("fast", [False, True])
def test_unchanged_answers_do_not_start_classification_crew(turn, db, user, monkeypatch, fast):
    session, _, _, _, _, _ = turn
    monkeypatch.setattr(routes, "_FAST_MODE", fast)
    crew = MagicMock(side_effect=AssertionError("Unchanged inputs must not start a crew"))
    monkeypatch.setattr("backend.agents.survey_classification_crew.SurveyClassificationCrew", crew)

    response = routes._send_message_impl(session.id, routes.MessageBody(message="some answer"), db, user)

    crew.assert_not_called()
    assert response.classification_execution["crew_attempted"] is False
    assert response.agent_execution["LanguageProcessor"] == ("skipped" if fast else "completed")


@pytest.mark.parametrize("employment_status", ["unemployed", "not_in_labour_force"])
def test_previous_occupation_uses_crew_and_saved_review(turn, db, user, monkeypatch, employment_status):
    session, context, manager, _, _, values = turn
    context.collected_data["employment_status"] = employment_status
    def answer(ctx, message, **kwargs):
        ctx.collected_data["last_job_title"] = message
        return "Next question"
    manager.process_message.side_effect = answer
    classify = MagicMock(return_value=SurveyClassificationResult(isco=values[0], execution={
        "mode": "crewai_sequential", "crew_attempted": True, "crew_completed": True,
        "requested_dimensions": ["isco"], "audit_tool_executed": True,
        "cooperation_verified": True,
    }))
    monkeypatch.setattr("backend.agents.survey_classification_crew.SurveyClassificationCrew",
        lambda **kwargs: SimpleNamespace(classify=classify))

    response = routes._send_message_impl(session.id, routes.MessageBody(message="Software developer"), db, user)

    assert classify.call_args.kwargs["job_title"] == "Software developer"
    assert response.isco_classifications[0].primary_code == "2512"
    occupation = db.query(SurveyResponse).filter_by(session_id=session.id, question_id="last_job_title", deleted_at=None).one()
    assert occupation.isco_code == "2512"
    assert db.query(HITLQueue).filter_by(response_id=occupation.id, status="pending").count() == 1
    assert db.query(SurveyResponse).filter_by(session_id=session.id, question_id="job_title").count() == 0


def test_never_worked_last_occupation_is_not_classified(turn, db, user, monkeypatch):
    session, context, manager, _, classifiers, _ = turn
    context.collected_data["employment_status"] = "unemployed"
    def answer(ctx, message, **kwargs):
        ctx.collected_data["last_job_title"] = "never_worked"
        return "Next question"
    manager.process_message.side_effect = answer
    crew = MagicMock(side_effect=AssertionError("A nonapplicable occupation has no classification"))
    monkeypatch.setattr("backend.agents.survey_classification_crew.SurveyClassificationCrew", crew)

    response = routes._send_message_impl(session.id, routes.MessageBody(message="I have never worked"), db, user)

    crew.assert_not_called()
    classifiers["isco"].classify.assert_not_called()
    assert response.isco_classifications == []


def test_cached_zero_confidence_does_not_become_confident(turn, db, user):
    session, context, _, _, _, _ = turn
    context.collected_data["job_title"] = "Software developer"
    occupation = SurveyResponse(session_id=session.id, question_id="job_title",
        answer="Software developer", isco_code="2512", confidence_score=0.0)
    db.add(occupation)
    db.flush()
    db.add(HITLQueue(session_id=session.id, response_id=occupation.id,
        raw_text="Software developer", ai_code="2512", ai_confidence=0.0,
        status="pending", priority="HIGH", created_at=datetime.utcnow()))
    db.commit()

    response = routes._send_message_impl(session.id, routes.MessageBody(message="some answer"), db, user)

    assert response.isco_classifications[0].confidence == 0.0
    assert response.isco_classifications[0].hitl_required is True
    assert response.agent_execution["ISCOClassifier"] == "cached"


@pytest.mark.parametrize("attainment_level", [0, 6])
def test_completion_forwards_classified_industry_and_attainment_to_register(turn, db, user, monkeypatch, attainment_level):
    session, context, manager, _, classifiers, values = turn
    context.collected_data["industry"] = "software company"
    values[2].level = attainment_level
    def complete(ctx, message, **kwargs):
        ctx.state = ConversationState.COMPLETING
        return "Interview complete"
    manager.process_message.side_effect = complete
    register = MagicMock()
    monkeypatch.setattr(routes, "_get_person_register_svc", lambda: register)
    monkeypatch.setattr(routes, "_ensure_isco_classification", lambda *args: None)
    monkeypatch.setattr(routes, "_trigger_quality_review", lambda *args: None)

    response = routes._send_message_impl(session.id, routes.MessageBody(message="Everything is correct"), db, user)

    assert response.session_completed is True
    assert register.update_from_session.call_args.kwargs["isic_code"] == "62"
    assert register.update_from_session.call_args.kwargs["isced_level"] == attainment_level
