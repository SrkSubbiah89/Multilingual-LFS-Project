"""Real duties reach all survey paths once; title persistence and review survive.

Classifiers/Crew are fakes; databases are isolated in-memory SQLite fixtures.
Run through scripts/run_review_tests.py to exclude live provider/startup access.
"""
from types import SimpleNamespace as NS
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from backend.agents.isco_classifier import ISCOClassification, ISCOMatch
from backend.agents.occupation_inputs import (
    classify_occupation_input, MAX_DUTIES_CHARS, DUTIES_PARENT_METHOD, PARENT_METHOD,
)
from backend.agents.survey_classification_crew import SurveyClassificationResult
from backend.api import survey_routes as routes
from backend.database.models import HITLQueue, SurveyResponse
from backend.database.response_revisions import save_response_revision
from backend.tests.test_title_alignment_routes import turn


def value(query="original", method=PARENT_METHOD):
    return ISCOClassification(query=query, language="en",
        primary=ISCOMatch(code="2512", title_en="Software developers", title_ar="", confidence=0.93),
        alternatives=[], method=method, hierarchy_path=["2", "25", "251", "2512"],
        hitl_required=True, reasoning="Synthetic official evidence; human review required")


@pytest.mark.parametrize("duties", ["", None, " N/A ", "never_worked", "UNKNOWN", "refused", "prefer_not_to_say", "prefer not to say"])
def test_missing_duties_preserve_exact_title_invocation_and_original_object(duties):
    original = value()
    classifier = NS(classify=MagicMock(return_value=original))
    trace = {}
    result = classify_occupation_input(classifier, "  Clerk  ", duties=duties, use_llm=False, trace=trace)
    classifier.classify.assert_called_once_with("  Clerk  ", use_llm=False, trace=trace)
    assert result is original
    assert trace["title_only_input"] is True and trace["duties_used"] is False
    assert "title_only_benchmark_applies" not in trace


@pytest.mark.parametrize("title,duties", [("Clerk", " clerk "), ("مطور", " مطور ")])
def test_equal_duties_are_not_appended_twice(title, duties):
    classifier = NS(classify=MagicMock(return_value=value()))
    classify_occupation_input(classifier, title, duties=duties)
    classifier.classify.assert_called_once_with(title)


@pytest.mark.parametrize("duties", ["---", "...", "123", "١٢٣", "१२३", "🙂👷", "\u200b\u200c\u200d", " \u200b---🙂123 "])
def test_uninformative_duties_preserve_title_only_method_scores_and_review(duties):
    original = value()
    classifier = NS(classify=MagicMock(return_value=original))
    trace = {}
    result = classify_occupation_input(classifier, " Assistant ", duties=duties, trace=trace)
    classifier.classify.assert_called_once_with(" Assistant ", trace=trace)
    assert result is original and result.method == PARENT_METHOD
    assert result.primary.confidence == 0.93 and result.hitl_required is True
    assert trace["duties_used"] is False and trace["title_only_input"] is True
    assert trace["query_mode"] == "title_only" and trace["retrieval_query"] == " Assistant "


@pytest.mark.parametrize("duties", [
    "أكتب برامج", "میں کوڈ لکھتا ہوں", "मैं सॉफ्टवेयर लिखता हूँ", "Nagsusulat ako ng software",
    "制作软件", "3D modelling", "C++ programming", "برمجة 3D",
])
def test_letter_guard_accepts_native_and_mixed_technical_duties_without_rewriting(duties):
    classifier = NS(classify=MagicMock(return_value=value()))
    trace = {}
    result = classify_occupation_input(classifier, "Assistant", duties=duties, trace=trace)
    classifier.classify.assert_called_once_with("Assistant " + duties, trace=trace)
    assert result.method == DUTIES_PARENT_METHOD and result.query == "Assistant"
    assert result.primary.confidence == 0.93 and result.hitl_required is True
    assert trace["duties_used"] is True and trace["duties_accuracy_evaluated"] is False


def test_unicode_duties_copy_parent_result_preserve_original_title_scores_and_review():
    original = value()
    classifier = NS(classify=MagicMock(return_value=original))
    trace = {}
    result = classify_occupation_input(classifier, "  مبرمج  ", duties="  أكتب برامج وأصلح الأخطاء  ",
        context="industry=software | language=ar", language="ar", use_llm=False, trace=trace)
    args, kwargs = classifier.classify.call_args
    assert args == ("مبرمج أكتب برامج وأصلح الأخطاء",)
    assert kwargs["context"] == "industry=software | language=ar"
    assert result.query == "مبرمج" and result.method == DUTIES_PARENT_METHOD
    assert result.primary is original.primary and result.primary.confidence == 0.93
    assert result.hitl_required is True and original.method == PARENT_METHOD and original.query == "original"
    assert trace["query_mode"] == "title_and_duties" and trace["duties_accuracy_evaluated"] is False
    assert trace["method"] == DUTIES_PARENT_METHOD and trace["retrieval_query"] == args[0]


def test_long_duties_are_bounded_and_truncation_is_explicit():
    classifier = NS(classify=MagicMock(return_value=value()))
    trace = {}
    classify_occupation_input(classifier, "Developer", duties="क" * (MAX_DUTIES_CHARS + 50), trace=trace)
    assert classifier.classify.call_args.args[0] == "Developer " + "क" * MAX_DUTIES_CHARS
    assert trace["duties_truncated"] is True


@pytest.mark.parametrize("prefix", ["1", "\u200b"])
def test_only_letters_outside_retained_duties_cannot_change_title_only_result(prefix):
    original = value()
    classifier = NS(classify=MagicMock(return_value=original))
    trace = {}
    duties = prefix * MAX_DUTIES_CHARS + "write code"
    result = classify_occupation_input(classifier, "Developer", duties=duties, trace=trace)
    classifier.classify.assert_called_once_with("Developer", trace=trace)
    assert result is original and trace["duties_used"] is False
    assert trace["duties_truncated"] is True and trace["title_only_input"] is True


def test_classifier_failure_does_not_retry_and_retains_input_trace():
    classifier = NS(classify=MagicMock(side_effect=TimeoutError))
    trace = {}
    with pytest.raises(TimeoutError):
        classify_occupation_input(classifier, "Developer", duties="write code", trace=trace)
    assert classifier.classify.call_count == 1 and trace["duties_used"] is True


def parent_classifier(turn):
    classifiers = turn[4]
    classifiers["isco"].classify.side_effect = lambda query, **kwargs: value(query=query)
    return classifiers["isco"]


def real_tool_fake_crew(monkeypatch):
    class Crew:
        def __init__(self, **dependencies):
            self.dependencies = dependencies

        def classify(self, **request):
            result = self.dependencies["isco_classifier"].classify(
                request["job_title"], context=request["isco_context"], language=request["language"], use_llm=request["use_llm"])
            return SurveyClassificationResult(isco=result, execution={
                "requested_dimensions": ["isco"], "mode": "crewai_sequential",
                "crew_attempted": True, "crew_completed": True, "audit_tool_executed": True,
                "cooperation_verified": True,
            })
    monkeypatch.setattr("backend.agents.survey_classification_crew.SurveyClassificationCrew", Crew)


@pytest.mark.parametrize("crew", [False, True])
def test_new_title_with_duties_uses_identical_query_and_persists_reported_title(turn, db, user, monkeypatch, crew):
    session, context, manager, _, _, _ = turn
    context.collected_data.update(job_duties="write code", employment_sector="private")
    manager.process_message.side_effect = lambda ctx, message, **kwargs: (ctx.collected_data.update(job_title=message) or "Next question")
    classifier = parent_classifier(turn)
    monkeypatch.setattr(routes, "_CREW_CLASSIFICATION_ENABLED", crew)
    if crew:
        real_tool_fake_crew(monkeypatch)
    response = routes._send_message_impl(session.id, routes.MessageBody(message="Developer"), db, user)
    assert classifier.classify.call_count == 1
    assert classifier.classify.call_args.args == ("Developer write code",)
    assert "write code" not in classifier.classify.call_args.kwargs["context"]
    assert "private" in classifier.classify.call_args.kwargs["context"]
    assert response.isco_classifications[0].method == DUTIES_PARENT_METHOD
    row = db.query(SurveyResponse).filter_by(session_id=session.id, question_id="job_title", deleted_at=None).one()
    assert row.answer == "Developer" and row.isco_code == "2512" and row.confidence_score == 0.93
    reviews = db.query(HITLQueue).filter_by(response_id=row.id, status="pending").all()
    assert len(reviews) == 1 and reviews[0].raw_text == "Developer"
    assert response.isco_classifications[0].hitl_required is True


@pytest.mark.parametrize("crew", [False, True])
@pytest.mark.parametrize("correction", [False, True])
def test_duties_update_and_correction_reclassify_once_and_retire_old_review(turn, db, user, monkeypatch, crew, correction):
    session, context, manager, _, _, _ = turn
    context.collected_data.update(job_title="Developer", job_duties="old duties")
    old = save_response_revision(db, session.id, "job_title", "Developer", isco_code="4110", confidence_score=0.6)
    db.add(HITLQueue(session_id=session.id, response_id=old.id, raw_text="Developer", ai_code="4110", ai_confidence=0.6, status="pending", priority="MEDIUM", created_at=datetime.utcnow()))
    db.commit()
    def answer(ctx, message, **kwargs):
        supplied = kwargs.get("structured_correction")
        duties = supplied[1] if supplied else message
        ctx.collected_data["job_duties"] = duties
        return "Next question"
    manager.process_message.side_effect = answer
    classifier = parent_classifier(turn)
    monkeypatch.setattr(routes, "_CREW_CLASSIFICATION_ENABLED", crew)
    if crew:
        real_tool_fake_crew(monkeypatch)
    payload = routes.MessageBody(message="repair production software",
        correction_field="job_duties" if correction else None,
        correction_value="repair production software" if correction else None)
    response = routes._send_message_impl(session.id, payload, db, user)
    classifier.classify.assert_called_once()
    assert classifier.classify.call_args.args == ("Developer repair production software",)
    active = db.query(SurveyResponse).filter_by(session_id=session.id, question_id="job_title", deleted_at=None).one()
    assert active.answer == "Developer" and active.isco_code == "2512" and active.supersedes_id == old.id
    assert old.deleted_at is not None
    pending = db.query(HITLQueue).filter_by(session_id=session.id, status="pending").all()
    assert len(pending) == 1 and pending[0].response_id == active.id
    assert response.isco_classifications[0].hitl_required is True


@pytest.mark.parametrize("employment_status", ["unemployed", "not_in_labour_force"])
@pytest.mark.parametrize("crew", [False, True])
def test_last_job_never_receives_stale_current_duties(turn, db, user, monkeypatch, employment_status, crew):
    session, context, manager, _, _, _ = turn
    context.collected_data.update(employment_status=employment_status, job_duties="current tasks", job_title="Current title")
    manager.process_message.side_effect = lambda ctx, message, **kwargs: (ctx.collected_data.update(last_job_title=message) or "Next question")
    classifier = parent_classifier(turn)
    monkeypatch.setattr(routes, "_CREW_CLASSIFICATION_ENABLED", crew)
    if crew:
        real_tool_fake_crew(monkeypatch)
    response = routes._send_message_impl(session.id, routes.MessageBody(message="Previous developer"), db, user)
    classifier.classify.assert_called_once()
    assert classifier.classify.call_args.args == ("Previous developer",)
    assert "current tasks" not in classifier.classify.call_args.kwargs.get("context", "")
    assert response.isco_classifications[0].method == PARENT_METHOD
    assert db.query(SurveyResponse).filter_by(session_id=session.id, question_id="job_title").count() == 1  # existing raw field only


@pytest.mark.parametrize("employment_status,field", [("employed", "job_title"), ("unemployed", "last_job_title")])
def test_completion_uses_same_adapter_without_persisting_combined_title(turn, db, user, employment_status, field):
    session, _, _, _, _, _ = turn
    row = save_response_revision(db, session.id, field, "Developer")
    db.commit()
    classifier = parent_classifier(turn)
    routes._ensure_isco_classification(db, session.id, {
        "employment_status": employment_status, field: "Developer", "job_duties": "write code", "industry": "software"})
    classifier.classify.assert_called_once()
    assert classifier.classify.call_args.args == ("Developer write code" if field == "job_title" else "Developer",)
    assert row.answer == "Developer" and row.isco_code == "2512" and row.confidence_score == 0.93
    assert db.query(HITLQueue).filter_by(response_id=row.id, status="pending").count() == 1
    routes._ensure_isco_classification(db, session.id, {"employment_status": employment_status, field: "Developer", "job_duties": "write code"})
    assert classifier.classify.call_count == 1


def test_never_worked_completion_does_not_classify_even_with_stale_duties(turn, db):
    session = turn[0]
    classifier = parent_classifier(turn)
    routes._ensure_isco_classification(db, session.id, {
        "employment_status": "unemployed", "last_job_title": "never_worked", "job_duties": "write code"})
    classifier.classify.assert_not_called()
