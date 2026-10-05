"""Regressions for review findings 2, 3, 15, 24, 27 and 28 (no AI calls)."""

import json
from unittest.mock import MagicMock

import pytest
import redis

from backend.agents.context_memory import ContextMemory
from backend.agents.conversation_manager import (
    ConversationContext,
    ConversationManager,
    ConversationState,
    StructuredAnswerError,
)


@pytest.fixture
def manager(monkeypatch):
    manager = ConversationManager.__new__(ConversationManager)
    monkeypatch.setattr(manager, "_llm_extract_correction", MagicMock(return_value=False))
    return manager


def context_at_field(manager, field, language="en", status="employed"):
    data = {"employment_status": status, "education_level": "secondary"}
    if status == "employed":
        data["employment_nature"] = "paid_employee"
    for name in manager._get_field_order(data):
        if name == field:
            break
        data.setdefault(name, "answered")
    return ConversationContext(
        session_id=1, language=language, state=ConversationState.COLLECTING_INFO,
        collected_data=data,
    )


@pytest.mark.parametrize("language,labels", [
    ("en", ("Employed", "Unemployed", "Not in the labour force")),
    ("ar", ("موظف", "عاطل عن العمل", "خارج سوق العمل")),
    ("ur", ("ملازم", "بے روزگار", "افرادی قوت سے باہر")),
    ("hi", ("नियोजित", "बेरोजगार", "श्रम बल से बाहर")),
    ("tl", ("Employed", "Unemployed", "Not in the labour force")),
])
def test_employment_buttons_route_all_supported_languages(manager, language, labels):
    for label, status in zip(labels, ("employed", "unemployed", "not_in_labour_force")):
        ctx = ConversationContext(1, language=language, state=ConversationState.COLLECTING_INFO)
        manager._transition(ctx, label, "")
        assert ctx.collected_data["employment_status"] == status
        assert ctx.state == ConversationState.COLLECTING_INFO
        assert "gender" in manager._get_field_order(ctx.collected_data)


@pytest.mark.parametrize("status", ("employed", "unemployed", "not_in_labour_force"))
def test_canonical_employment_answers_are_supported(manager, status):
    ctx = ConversationContext(1, state=ConversationState.COLLECTING_INFO)
    manager._transition(ctx, status, "")
    assert ctx.collected_data["employment_status"] == status


def test_repeated_invalid_status_answers_never_collapse_survey_path(manager):
    ctx = ConversationContext(1, state=ConversationState.COLLECTING_INFO)
    for _ in range(8):
        manager._transition(ctx, "hello", "")
        assert ctx.state == ConversationState.CLARIFYING
        assert ctx.clarification_target == "employment_status"
        assert "employment_status" not in ctx.collected_data
    manager._transition(ctx, "बेरोजगार", "")
    assert ctx.state == ConversationState.COLLECTING_INFO
    assert ctx.collected_data["employment_status"] == "unemployed"
    assert ctx.clarification_count == 0


def test_invalid_employment_correction_is_rejected(manager):
    ctx = ConversationContext(1, state=ConversationState.VALIDATING,
                              collected_data={"employment_status": "employed"})
    manager._transition(ctx, "change employment status to hello", "")
    assert ctx.collected_data["employment_status"] == "employed"
    assert ctx.correction_rejected_field == "employment_status"
    assert ctx.state == ConversationState.VALIDATING


@pytest.mark.parametrize("field,answer", [
    ("uae_residence_duration", "5-9 years"),
    ("actual_hours_worked", "40 hours last week"),
    ("job_duties", "Manage 12 employees and 3 branches"),
    ("gender", "Prefer not to say"),
])
def test_unrelated_answers_do_not_supply_wages(manager, field, answer):
    ctx = context_at_field(manager, field)
    manager._extract_fields(ctx, answer)
    assert field in ctx.collected_data
    assert "monthly_wage_range" not in ctx.collected_data


@pytest.mark.parametrize("language,under,over,refusal", [
    ("en", "Less than 5,000", "More than 50,000", "Prefer not to say"),
    ("ar", "أقل من 5,000", "أكثر من 50,000", "أفضل عدم الإفصاح"),
    ("ur", "5,000 سے کم", "50,000 سے زیادہ", "بتانا نہیں چاہتا"),
    ("hi", "5,000 से कम", "50,000 से अधिक", "बताना नहीं चाहते"),
    ("tl", "Wala pang 5,000", "Higit sa 50,000", "Ayaw sabihin"),
])
def test_wage_buttons_keep_intended_band(manager, language, under, over, refusal):
    for answer, band in (
        (under, "under_5000"), (over, "over_50000"), (refusal, "prefer_not_to_say"),
        ("5,000–10,000", "5000_10000"), ("10,001–20,000", "10001_20000"),
        ("20,001–50,000", "20001_50000"),
    ):
        ctx = context_at_field(manager, "monthly_wage_range", language)
        manager._transition(ctx, answer, "")
        assert ctx.collected_data["monthly_wage_range"] == band
        assert ctx.state == ConversationState.COLLECTING_INFO


@pytest.mark.parametrize("answer,band", [
    ("under_5000", "under_5000"), ("5000_10000", "5000_10000"),
    ("10001_20000", "10001_20000"), ("20001_50000", "20001_50000"),
    ("over_50000", "over_50000"), ("prefer_not_to_say", "prefer_not_to_say"),
    ("4,999", "under_5000"), ("5k", "5000_10000"), ("10,000", "5000_10000"),
    ("10001", "10001_20000"), ("20,000", "10001_20000"),
    ("20001", "20001_50000"), ("50,000", "20001_50000"), ("50001", "over_50000"),
    ("أقل من ٥٬٠٠٠", "under_5000"), ("५०,००१", "over_50000"),
    ("Around 8,000 per month", "5000_10000"),
])
def test_canonical_wage_answers_and_numeric_boundaries(manager, answer, band):
    ctx = context_at_field(manager, "monthly_wage_range")
    manager._extract_fields(ctx, answer)
    assert ctx.collected_data["monthly_wage_range"] == band


@pytest.mark.parametrize("answer,band", [
    ("My monthly salary is 8,000 AED", "5000_10000"),
    ("راتبي أقل من 5,000 درهم", "under_5000"),
    ("मेरा वेतन 50,001 AED है", "over_50000"),
])
def test_explicit_wage_answers_can_be_collected_before_wage_turn(manager, answer, band):
    ctx = context_at_field(manager, "uae_residence_duration")
    manager._extract_fields(ctx, answer)
    assert ctx.collected_data["monthly_wage_range"] == band


def test_explicit_salary_uses_salary_amount_after_unrelated_number(manager):
    ctx = context_at_field(manager, "uae_residence_duration")
    manager._extract_fields(ctx, "I have 5 years in UAE; my salary is 8,000 AED")
    assert ctx.collected_data["monthly_wage_range"] == "5000_10000"


def test_income_keyword_does_not_turn_case_count_into_wages(manager):
    ctx = context_at_field(manager, "job_duties")
    manager._extract_fields(ctx, "I manage 12 income tax cases")
    assert "monthly_wage_range" not in ctx.collected_data


@pytest.mark.parametrize("answer", ("40 hours", "5 years", "I manage 12 employees"))
def test_unrelated_numbers_on_wage_turn_require_clarification(manager, answer):
    ctx = context_at_field(manager, "monthly_wage_range")
    manager._transition(ctx, answer, "")
    assert "monthly_wage_range" not in ctx.collected_data
    assert ctx.state == ConversationState.CLARIFYING
    assert ctx.clarification_target == "monthly_wage_range"


@pytest.mark.parametrize("language,answer", [
    ("en", "not right"), ("en", "yes, but not correct"),
    ("en", "not sure"), ("en", "I cannot confirm"), ("en", "isn't right"),
    ("ar", "غير صحيح"), ("ar-gulf", "غير صحيح"),
    ("ur", "درست نہیں"), ("hi", "सही नहीं"), ("tl", "hindi tama"),
])
def test_negated_confirmations_open_correction(manager, language, answer):
    ctx = ConversationContext(1, language=language, state=ConversationState.VALIDATING)
    manager._transition(ctx, answer, "")
    assert manager._is_confirmed(answer, language) is False
    assert ctx.state == ConversationState.VALIDATING
    assert ctx.correction_no_target is True
    manager._llm_extract_correction.assert_not_called()


def test_confirmation_with_correction_applies_change_before_completion(manager):
    ctx = context_at_field(manager, None)
    ctx.state = ConversationState.VALIDATING
    ctx.collected_data["industry"] = "technology"
    manager._transition(ctx, "Yes, but change my industry to healthcare", "")
    assert ctx.collected_data["industry"] == "healthcare"
    assert ctx.state == ConversationState.VALIDATING
    assert ctx.correction_applied is True
    assert ctx.corrected_fields == {"industry"}
    manager._llm_extract_correction.assert_not_called()


@pytest.mark.parametrize("language,answer", [
    ("en", "yes, that's correct"), ("ar", "نعم، هذا صحيح"),
    ("ur", "ہاں، یہ درست ہے"), ("hi", "हाँ, यह सही है"), ("tl", "Oo, tama iyan"),
])
def test_positive_confirmations_still_complete(manager, language, answer):
    ctx = ConversationContext(1, language=language, state=ConversationState.VALIDATING)
    manager._transition(ctx, answer, "")
    assert ctx.state == ConversationState.COMPLETING


@pytest.mark.parametrize("answer", ["isn’t right", "wasn't correct", "aren't right", "doesn't look right"])
def test_negated_contractions_never_confirm(manager, answer):
    ctx = ConversationContext(1, state=ConversationState.VALIDATING)
    manager._transition(ctx, answer, "")
    assert ctx.state == ConversationState.VALIDATING
    assert manager._is_confirmed(answer, "en") is False


@pytest.mark.parametrize("language,label", [
    ("en", "Bachelor's degree"), ("ar", "بكالوريوس"), ("ur", "بیچلر ڈگری"),
    ("hi", "स्नातक"), ("tl", "Batsilyer"),
])
@pytest.mark.parametrize("structured", [False, True])
def test_bachelor_labels_enable_field_of_study(manager, language, label, structured):
    ctx = context_at_field(manager, "education_level", language)
    ctx.collected_data.pop("education_level", None)
    manager._transition(ctx, label, "", structured_answer=("education_level", "bachelor") if structured else None)
    assert ctx.collected_data["education_level"] == "bachelor"
    assert "field_of_study" in manager._get_field_order(ctx.collected_data)


@pytest.mark.parametrize("language,label", [
    ("en", "Paid employee"), ("ar", "موظف براتب"), ("ur", "تنخواہ دار ملازم"),
    ("hi", "वेतनभोगी कर्मचारी"), ("tl", "Bayad na empleyado"),
])
@pytest.mark.parametrize("structured", [False, True])
def test_paid_employee_labels_enable_contract_and_wage_questions(manager, language, label, structured):
    ctx = context_at_field(manager, "employment_nature", language)
    ctx.collected_data.pop("employment_nature", None)
    manager._transition(ctx, label, "", structured_answer=("employment_nature", "paid_employee") if structured else None)
    assert ctx.collected_data["employment_nature"] == "paid_employee"
    path = manager._get_field_order(ctx.collected_data)
    assert "contract_type" in path
    assert "monthly_wage_range" in path


@pytest.mark.parametrize("language,label", [
    ("en", "No — never worked"), ("ar", "لا — لم أعمل أبداً"), ("ur", "نہیں — کبھی نہیں کیا"),
    ("hi", "नहीं — कभी नहीं किया"), ("tl", "Hindi — hindi pa nagtrabaho"),
])
def test_never_worked_labels_skip_previous_occupation(manager, language, label):
    ctx = context_at_field(manager, "ever_worked", language, status="unemployed")
    manager._transition(ctx, label, "")
    assert ctx.collected_data["ever_worked"] == "never_worked"
    assert "last_job_title" not in manager._get_field_order(ctx.collected_data)


@pytest.mark.parametrize("language,label", [
    ("en", "Yes — primary income"), ("ar", "نعم — دخل رئيسي"), ("ur", "ہاں — بنیادی آمدنی"),
    ("hi", "हाँ — मुख्य आय"), ("tl", "Oo — pangunahing kita"),
])
def test_platform_income_labels_enable_platform_details(manager, language, label):
    ctx = context_at_field(manager, "platform_work", language)
    manager._transition(ctx, label, "")
    assert ctx.collected_data["platform_work"] == "yes_primary"
    path = manager._get_field_order(ctx.collected_data)
    assert "platform_names" in path
    assert "platform_hours" in path


@pytest.mark.parametrize("label", ["Emirati", "إماراتي", "اماراتی", "इमिराती"])
def test_uae_national_labels_enable_emiratization(manager, label):
    ctx = context_at_field(manager, "nationality")
    manager._transition(ctx, label, "")
    assert ctx.collected_data["nationality"] == "uae_national"
    assert "emiratization_program" in manager._get_field_order(ctx.collected_data)


@pytest.mark.parametrize("language,yes,no", [
    ("en", "Yes", "No"), ("ar", "نعم", "لا"), ("ur", "ہاں", "نہیں"),
    ("hi", "हाँ", "नहीं"), ("tl", "Oo", "Hindi"),
])
def test_binary_labels_apply_search_and_secondary_job_skip_gates(manager, language, yes, no):
    ctx = context_at_field(manager, "secondary_job", language)
    manager._transition(ctx, yes, "")
    assert ctx.collected_data["secondary_job"] == "yes"
    assert "secondary_job_hours" in manager._get_field_order(ctx.collected_data)
    ctx = context_at_field(manager, "job_search_active", language, status="unemployed")
    manager._transition(ctx, yes, "")
    assert ctx.collected_data["job_search_active"] == "yes"
    assert "job_search_methods" in manager._get_field_order(ctx.collected_data)
    ctx = context_at_field(manager, "available_for_work", language, status="unemployed")
    ctx.collected_data["job_search_active"] = "no"
    manager._transition(ctx, no, "", structured_answer=("available_for_work", "no"))
    assert ctx.collected_data["employment_status"] == "not_in_labour_force"


def test_structured_answer_preserves_display_label_and_only_updates_current_field(manager):
    manager._agent_available = False
    ctx = context_at_field(manager, "employment_nature", "hi")
    ctx.collected_data.pop("employment_nature", None)
    label = "वेतनभोगी कर्मचारी"
    manager.process_message(ctx, label, structured_answer=("employment_nature", "paid_employee"))
    assert ctx.history[0] == {"role": "user", "content": label}
    assert ctx.collected_data["employment_nature"] == "paid_employee"
    assert "job_title" not in ctx.collected_data


@pytest.mark.parametrize("answer", [
    ("employment_status", "unemployed"), ("education_level", "hello"),
    ("job_title", "doctor"), ("monthly_wage_range", "under_5000"),
])
def test_stale_or_invalid_quick_answers_are_rejected_without_mutation(manager, answer):
    ctx = ConversationContext(1, state=ConversationState.COLLECTING_INFO,
                              collected_data={"employment_status": "employed"})
    before = dict(ctx.collected_data)
    with pytest.raises(StructuredAnswerError):
        manager.process_message(ctx, "translated label", structured_answer=answer)
    assert ctx.collected_data == before
    assert ctx.history == []


def test_structured_answer_resumes_clarification_target(manager):
    ctx = ConversationContext(1, state=ConversationState.CLARIFYING,
        collected_data={"employment_status": "employed"}, clarification_target="education_level",
        clarification_count=2)
    manager._transition(ctx, "स्नातक", "", structured_answer=("education_level", "bachelor"))
    assert ctx.collected_data["education_level"] == "bachelor"
    assert ctx.state == ConversationState.COLLECTING_INFO
    assert ctx.clarification_count == 0
    assert ctx.clarification_target is None


@pytest.mark.parametrize("field,label,canonical,opened", [
    ("education_level", "स्नातक", "bachelor", "field_of_study"),
    ("secondary_job", "ہاں", "yes", "secondary_job_hours"),
    ("platform_work", "Oo — pangunahing kita", "yes_primary", "platform_names"),
])
def test_localized_corrections_collect_newly_applicable_questions(manager, field, label, canonical, opened):
    ctx = context_at_field(manager, None)
    ctx.state = ConversationState.VALIDATING
    manager._transition(ctx, "Correction", "", structured_correction=(field, label))
    assert ctx.collected_data[field] == canonical
    assert opened in manager._get_field_order(ctx.collected_data)
    assert opened not in ctx.collected_data
    assert ctx.state == ConversationState.COLLECTING_INFO


@pytest.fixture
def memory():
    memory = ContextMemory.__new__(ContextMemory)
    memory._ttl = 86400
    store = {}
    memory._redis = MagicMock()
    memory._redis.get.side_effect = store.get
    memory._redis.set.side_effect = lambda key, value, ex=None: store.__setitem__(key, value)
    return memory


def test_memory_round_trips_fsm_resume_fields(memory):
    values = dict(
        clarification_target="employment_status", clarification_count=2,
        is_returning=True, correction_applied=True, corrected_fields={"industry"},
        correction_rejected_field="job_title", correction_no_target=True,
        prefilled_fields=["employment_status", "education_level"],
    )
    memory.save_session(1, "clarifying", "hi", {}, [], **values)
    loaded = memory.load_session(1)
    assert {key: getattr(loaded, key) for key in values} == values


def test_legacy_memory_documents_have_resume_defaults(memory):
    saved = memory.save_session(1, "clarifying", "en", {}, [])
    legacy = saved.model_dump()
    for key in (
        "clarification_target", "clarification_count", "is_returning", "correction_applied",
        "corrected_fields", "correction_rejected_field", "correction_no_target",
        "prefilled_fields",
    ):
        legacy.pop(key)
    memory._redis.set("lfs:session:1", json.dumps(legacy))
    loaded = memory.load_session(1)
    assert loaded.clarification_target is None
    assert loaded.clarification_count == 0
    assert loaded.corrected_fields == set()
    assert loaded.prefilled_fields == []


def test_session_lock_releases_after_body_error(memory):
    lock = memory._redis.lock.return_value
    lock.acquire.return_value = True
    with pytest.raises(ValueError, match="turn failed"):
        with memory.session_lock(1):
            raise ValueError("turn failed")
    memory._redis.lock.assert_called_once_with(
        "lfs:session-lock:1", timeout=600, blocking_timeout=5, thread_local=False,
    )
    lock.release.assert_called_once()


def test_session_lock_rejects_busy_session(memory):
    lock = memory._redis.lock.return_value
    lock.acquire.return_value = False
    with pytest.raises(redis.exceptions.LockError):
        with memory.session_lock(1):
            pytest.fail("busy session entered lock")
    lock.release.assert_not_called()


def test_session_lock_does_not_fall_back_when_redis_is_unavailable(memory):
    memory._redis.lock.return_value.acquire.side_effect = redis.exceptions.ConnectionError("offline")
    with pytest.raises(redis.exceptions.ConnectionError):
        with memory.session_lock(1):
            pytest.fail("unavailable lock entered")


@pytest.mark.parametrize("renewal_fails", (False, True))
def test_session_lock_renews_long_turns_and_surfaces_renewal_failure(memory, monkeypatch, renewal_fails):
    stop = MagicMock()
    stop.wait.side_effect = [False, True]
    monkeypatch.setattr("backend.agents.context_memory.Event", lambda: stop)

    def run_immediately(target, **kwargs):
        thread = MagicMock()
        thread.start.side_effect = target
        return thread

    monkeypatch.setattr("backend.agents.context_memory.Thread", run_immediately)
    lock = memory._redis.lock.return_value
    lock.acquire.return_value = True
    if renewal_fails:
        lock.extend.side_effect = redis.exceptions.ConnectionError("lease unavailable")
        with pytest.raises(redis.exceptions.LockError, match="renewal failed"):
            with memory.session_lock(1):
                pass
    else:
        with memory.session_lock(1):
            pass
    lock.extend.assert_called_once_with(600, replace_ttl=True)
    lock.release.assert_called_once()
