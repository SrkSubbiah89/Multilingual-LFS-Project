"""
Tests for backend/agents/conversation_manager.py

Pure logic (field extraction, ambiguity detection, confirmation detection,
FSM transitions) is tested without any mocking.  The full process_message()
turn patches Agent and Crew at the module level so CrewAI's Pydantic LLM
validation is bypassed entirely.
"""

from unittest.mock import MagicMock

import pytest

from backend.agents.conversation_manager import (
    ConversationContext,
    ConversationManager,
    ConversationState,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MOCK_REPLY = "Thank you! Can you tell me how many hours you work per week?"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_crew(monkeypatch):
    """
    Patches Agent and Crew inside conversation_manager.
    Returns the Crew *instance* mock so tests can control kickoff() return values.
    """
    crew_instance = MagicMock()
    crew_instance.kickoff.return_value = _MOCK_REPLY
    crew_class = MagicMock(return_value=crew_instance)

    monkeypatch.setattr("backend.agents.conversation_manager.Agent", MagicMock())
    monkeypatch.setattr("backend.agents.conversation_manager.Crew", crew_class)
    monkeypatch.setattr("backend.agents.conversation_manager.Task", MagicMock())
    return crew_instance


@pytest.fixture
def mgr(monkeypatch, mock_crew):
    """ConversationManager with LLM construction patched out."""
    monkeypatch.setattr(
        "backend.agents.conversation_manager.get_llm",
        lambda *a, **kw: MagicMock(),
    )
    return ConversationManager()


@pytest.fixture
def ctx(mgr):
    """Fresh English ConversationContext at session_id=1."""
    return mgr.new_context(session_id=1, language="en")


@pytest.fixture
def full_ctx(mgr):
    """Context with all required fields already collected (employed path)."""
    c = mgr.new_context(session_id=2, language="en")
    c.collected_data = {
        # Core (B)
        "employment_status":  "employed",
        "education_level":    "bachelor",
        # Demographics (B)
        "gender":             "male",
        "nationality":        "uae_national",
        "marital_status":     "married",
        "emirate":            "dubai",
        # Employed path (C/D/E)
        "employment_nature":  "paid_employee",
        "employment_sector":  "private",
        "job_title":          "software engineer",
        "job_duties":         "writes code and reviews pull requests",
        "industry":           "technology",
        "hours_per_week":     "40",
        "employment_type":    "full_time",
        "monthly_wage_range": "10001_20000",
        # Skills & digital (H/I)
        "main_skills":        "programming",
        "platform_work":      "no",
        # Quality of work (J)
        "job_satisfaction":   "satisfied",
        # Feedback (K)
        "question_clarity":   "very_clear",
        "ai_preference":      "prefer_ai",
        "data_confidence":    "very_confident",
    }
    c.state = ConversationState.COLLECTING_INFO
    return c


# ---------------------------------------------------------------------------
# new_context
# ---------------------------------------------------------------------------

class TestNewContext:
    def test_initial_state_is_greeting(self, mgr):
        ctx = mgr.new_context(session_id=1, language="en")
        assert ctx.state == ConversationState.GREETING

    def test_language_en_stored(self, mgr):
        ctx = mgr.new_context(session_id=1, language="en")
        assert ctx.language == "en"

    def test_language_ar_stored(self, mgr):
        ctx = mgr.new_context(session_id=1, language="ar")
        assert ctx.language == "ar"

    def test_invalid_language_defaults_to_en(self, mgr):
        ctx = mgr.new_context(session_id=1, language="fr")
        assert ctx.language == "en"

    def test_history_starts_empty(self, mgr):
        ctx = mgr.new_context(session_id=1, language="en")
        assert ctx.history == []

    def test_collected_data_starts_empty(self, mgr):
        ctx = mgr.new_context(session_id=1, language="en")
        assert ctx.collected_data == {}

    def test_session_id_stored(self, mgr):
        ctx = mgr.new_context(session_id=42, language="en")
        assert ctx.session_id == 42


# ---------------------------------------------------------------------------
# _extract_fields — employment status
# ---------------------------------------------------------------------------

class TestExtractFieldsEmploymentStatus:
    def test_employed_keyword(self, mgr, ctx):
        mgr._extract_fields(ctx, "I am currently employed full time")
        assert ctx.collected_data.get("employment_status") == "employed"

    def test_working_keyword(self, mgr, ctx):
        mgr._extract_fields(ctx, "I have been working for five years")
        assert ctx.collected_data.get("employment_status") == "employed"

    def test_unemployed_keyword(self, mgr, ctx):
        mgr._extract_fields(ctx, "I am unemployed at the moment")
        assert ctx.collected_data.get("employment_status") == "unemployed"

    def test_looking_for_work_keyword(self, mgr, ctx):
        mgr._extract_fields(ctx, "I am looking for work right now")
        assert ctx.collected_data.get("employment_status") == "unemployed"

    def test_retired_maps_to_not_in_labour_force(self, mgr, ctx):
        mgr._extract_fields(ctx, "I am retired")
        assert ctx.collected_data.get("employment_status") == "not_in_labour_force"

    def test_arabic_employed_keyword(self, mgr, ctx):
        mgr._extract_fields(ctx, "أنا أعمل في شركة تقنية")
        assert ctx.collected_data.get("employment_status") == "employed"

    def test_arabic_unemployed_keyword(self, mgr, ctx):
        mgr._extract_fields(ctx, "أنا عاطل عن العمل")
        assert ctx.collected_data.get("employment_status") == "unemployed"

    def test_does_not_overwrite_existing_status(self, mgr, ctx):
        ctx.collected_data["employment_status"] = "employed"
        mgr._extract_fields(ctx, "I am unemployed")
        assert ctx.collected_data["employment_status"] == "employed"


# ---------------------------------------------------------------------------
# _extract_fields — hours per week
# ---------------------------------------------------------------------------

class TestExtractFieldsHours:
    def test_extracts_digit_hours_pattern(self, mgr, ctx):
        mgr._extract_fields(ctx, "I work 40 hours a week")
        assert ctx.collected_data.get("hours_per_week") == "40"

    def test_extracts_hrs_abbreviation(self, mgr, ctx):
        mgr._extract_fields(ctx, "Usually about 35 hrs per week")
        assert ctx.collected_data.get("hours_per_week") == "35"

    def test_extracts_arabic_hours_keyword(self, mgr, ctx):
        mgr._extract_fields(ctx, "أعمل 40 ساعة أسبوعيًا")
        assert ctx.collected_data.get("hours_per_week") == "40"

    def test_no_hours_pattern_leaves_field_absent(self, mgr, ctx):
        mgr._extract_fields(ctx, "I work a lot")
        assert "hours_per_week" not in ctx.collected_data


# ---------------------------------------------------------------------------
# _extract_fields — employment type
# ---------------------------------------------------------------------------

class TestExtractFieldsEmploymentType:
    def test_full_time_extracted(self, mgr, ctx):
        mgr._extract_fields(ctx, "I work full time")
        assert ctx.collected_data.get("employment_type") == "full_time"

    def test_fulltime_hyphenated(self, mgr, ctx):
        mgr._extract_fields(ctx, "I have a full-time position")
        assert ctx.collected_data.get("employment_type") == "full_time"

    def test_part_time_extracted(self, mgr, ctx):
        mgr._extract_fields(ctx, "I only work part time")
        assert ctx.collected_data.get("employment_type") == "part_time"

    def test_self_employed_extracted(self, mgr, ctx):
        mgr._extract_fields(ctx, "I am self-employed as a consultant")
        assert ctx.collected_data.get("employment_type") == "self_employed"

    def test_freelance_maps_to_self_employed(self, mgr, ctx):
        mgr._extract_fields(ctx, "I freelance as a designer")
        assert ctx.collected_data.get("employment_type") == "self_employed"


# ---------------------------------------------------------------------------
# _extract_fields — job title
# ---------------------------------------------------------------------------

_PRE_JOB_TITLE_DATA = {
    "employment_status":      "employed",
    "education_level":        "diploma",  # no field_of_study branch
    "gender":                 "male",
    "nationality":            "uae_national",
    "marital_status":         "married",
    "emirate":                "dubai",
    "uae_residence_duration": "5_to_10",
    "vocational_training":    "no",
    "employment_nature":      "paid_employee",
    "employment_sector":      "private",
}


class TestExtractFieldsJobTitle:
    def test_engineer_keyword_captures_raw_text(self, mgr, ctx):
        ctx.collected_data = dict(_PRE_JOB_TITLE_DATA)
        mgr._extract_fields(ctx, "I am a civil engineer")
        assert ctx.collected_data.get("job_title") is not None

    def test_doctor_keyword_captured(self, mgr, ctx):
        ctx.collected_data = dict(_PRE_JOB_TITLE_DATA)
        mgr._extract_fields(ctx, "I work as a doctor in a clinic")
        assert ctx.collected_data.get("job_title") is not None

    def test_arabic_doctor_keyword_captured(self, mgr, ctx):
        ctx.collected_data = dict(_PRE_JOB_TITLE_DATA)
        mgr._extract_fields(ctx, "أنا طبيب في عيادة خاصة")
        assert ctx.collected_data.get("job_title") is not None


# ---------------------------------------------------------------------------
# _extract_fields — industry
# ---------------------------------------------------------------------------

class TestExtractFieldsIndustry:
    def test_technology_sector(self, mgr, ctx):
        mgr._extract_fields(ctx, "I work in software and tech")
        assert ctx.collected_data.get("industry") == "technology"

    def test_healthcare_sector(self, mgr, ctx):
        mgr._extract_fields(ctx, "I work in a hospital")
        assert ctx.collected_data.get("industry") == "healthcare"

    def test_education_sector(self, mgr, ctx):
        mgr._extract_fields(ctx, "I teach at a university")
        assert ctx.collected_data.get("industry") == "education"

    def test_government_sector(self, mgr, ctx):
        mgr._extract_fields(ctx, "I work for the ministry of finance")
        assert ctx.collected_data.get("industry") == "government"

    def test_finance_sector(self, mgr, ctx):
        mgr._extract_fields(ctx, "I work at a bank as an analyst")
        assert ctx.collected_data.get("industry") == "finance"

    def test_arabic_technology_sector(self, mgr, ctx):
        mgr._extract_fields(ctx, "أعمل في قطاع التقنية والبرمجة")
        assert ctx.collected_data.get("industry") == "technology"


# ---------------------------------------------------------------------------
# _is_ambiguous
# ---------------------------------------------------------------------------

class TestIsAmbiguous:
    def test_empty_string_is_ambiguous(self, mgr):
        assert mgr._is_ambiguous("") is True

    def test_very_short_string_is_ambiguous(self, mgr):
        assert mgr._is_ambiguous("ok") is True

    def test_single_vague_word_yes_is_ambiguous(self, mgr):
        assert mgr._is_ambiguous("yes") is True

    def test_ok_is_ambiguous(self, mgr):
        assert mgr._is_ambiguous("okay") is True

    def test_arabic_vague_word_is_ambiguous(self, mgr):
        assert mgr._is_ambiguous("نعم") is True

    def test_substantive_english_not_ambiguous(self, mgr):
        assert mgr._is_ambiguous("I work as a software engineer full time") is False

    def test_substantive_arabic_not_ambiguous(self, mgr):
        assert mgr._is_ambiguous("أنا مهندس برمجيات في شركة تقنية") is False


# ---------------------------------------------------------------------------
# _is_confirmed
# ---------------------------------------------------------------------------

class TestIsConfirmed:
    def test_yes_confirms_english(self, mgr):
        assert mgr._is_confirmed("yes", "en") is True

    def test_correct_confirms_english(self, mgr):
        assert mgr._is_confirmed("that's correct", "en") is True

    def test_looks_good_confirms_english(self, mgr):
        assert mgr._is_confirmed("looks good", "en") is True

    def test_arabic_confirmation(self, mgr):
        assert mgr._is_confirmed("نعم، هذا صحيح", "ar") is True

    def test_arabic_موافق_confirms(self, mgr):
        assert mgr._is_confirmed("موافق", "ar") is True

    def test_unrelated_text_not_confirmed(self, mgr):
        assert mgr._is_confirmed("I want to correct my industry", "en") is False

    def test_empty_string_not_confirmed(self, mgr):
        assert mgr._is_confirmed("", "en") is False


# ---------------------------------------------------------------------------
# FSM transitions — _transition
# ---------------------------------------------------------------------------

class TestTransition:
    def test_greeting_always_advances_to_collecting(self, mgr, ctx):
        ctx.state = ConversationState.GREETING
        mgr._transition(ctx, "Hello", "Welcome!")
        assert ctx.state == ConversationState.COLLECTING_INFO

    def test_collecting_stays_when_fields_incomplete(self, mgr, ctx):
        ctx.state = ConversationState.COLLECTING_INFO
        ctx.collected_data = {}
        mgr._transition(ctx, "I am a software engineer", "Got it!")
        # Not all required fields present → stays in COLLECTING or moves to CLARIFYING
        assert ctx.state in (ConversationState.COLLECTING_INFO, ConversationState.CLARIFYING)

    def test_collecting_advances_to_validating_when_all_fields_present(self, mgr, full_ctx):
        # All required fields are already present in full_ctx — any input should
        # trigger a VALIDATING transition since required fields are complete.
        full_ctx.state = ConversationState.COLLECTING_INFO
        mgr._transition(full_ctx, "That is all my information", "")
        assert full_ctx.state == ConversationState.VALIDATING

    def test_collecting_moves_to_clarifying_on_ambiguous_input(self, mgr, ctx):
        ctx.state = ConversationState.COLLECTING_INFO
        ctx.collected_data = {}
        mgr._transition(ctx, "yes", "Could you clarify?")
        assert ctx.state == ConversationState.CLARIFYING

    def test_clarifying_returns_to_collecting(self, mgr, ctx):
        ctx.state = ConversationState.CLARIFYING
        ctx.clarification_target = "employment_status"
        mgr._transition(ctx, "I am employed full time", "Thank you!")
        assert ctx.state == ConversationState.COLLECTING_INFO

    def test_validating_moves_to_completing_on_confirmation(self, mgr, ctx):
        ctx.state = ConversationState.VALIDATING
        mgr._transition(ctx, "yes, that's correct", "")
        assert ctx.state == ConversationState.COMPLETING

    def test_validating_correction_opening_new_field_path_returns_to_collecting(self, mgr, ctx):
        # A correction that changes employment_status opens a whole new set of
        # required fields (e.g. job_search_active for the unemployed path) that
        # aren't in collected_data yet, so VALIDATING must drop back to
        # COLLECTING_INFO to gather them.
        ctx.state = ConversationState.VALIDATING
        ctx.collected_data["employment_status"] = "employed"
        mgr._transition(ctx, "change the employment status to unemployed", "")
        assert ctx.state == ConversationState.COLLECTING_INFO
        assert ctx.collected_data["employment_status"] == "unemployed"

    def test_validating_unrecognised_non_confirmation_stays_validating(self, mgr, ctx, monkeypatch):
        # Regression test: `correction_applied` (and any state transition) must
        # only fire when a correction was actually understood and applied.
        # Previously this branch transitioned unconditionally, so a message
        # that wasn't a confirmation AND wasn't a parseable correction would
        # still make the system claim "I've updated that for you" with
        # nothing having changed. Mock the LLM fallback so this test is
        # deterministic regardless of whether Ollama/Anthropic are reachable.
        monkeypatch.setattr(mgr, "_llm_extract_correction", lambda ctx, text: False)
        ctx.state = ConversationState.VALIDATING
        mgr._transition(ctx, "wait, that's not it", "")
        assert ctx.state == ConversationState.VALIDATING
        assert ctx.correction_applied is False
        assert ctx.correction_no_target is True

    def test_validating_bare_no_sets_correction_no_target(self, mgr, ctx, monkeypatch):
        # Regression test for a real, live-reproduced dead-end: a respondent
        # replying with a bare "no" (no field named) previously left every
        # correction flag unset, so the reply just re-showed the identical
        # validation summary forever with no indication the "no" was ever
        # understood. correction_no_target must now be set so the reply asks
        # what to correct instead of silently repeating the same message.
        monkeypatch.setattr(mgr, "_llm_extract_correction", lambda ctx, text: False)
        ctx.state = ConversationState.VALIDATING
        mgr._transition(ctx, "no", "")
        assert ctx.state == ConversationState.VALIDATING
        assert ctx.correction_applied is False
        assert ctx.correction_rejected_field is None
        assert ctx.correction_no_target is True

    def test_validating_correction_no_target_stub_reply_asks_what_to_correct(self, mgr, ctx, monkeypatch):
        # The FAST_MODE / no-LLM-available template must not repeat the
        # identical summary when correction_no_target is set — it must ask
        # specifically what the respondent wants to correct.
        monkeypatch.setattr(mgr, "_llm_extract_correction", lambda ctx, text: False)
        ctx.state = ConversationState.VALIDATING
        ctx.collected_data["employment_status"] = "employed"
        mgr._transition(ctx, "no", "")
        reply = mgr._dev_stub_response(ctx)
        assert "what would you like to correct" in reply.lower()
        assert "here's a summary of what i've collected" not in reply.lower()
        assert ctx.correction_no_target is False  # consumed after one reply

    def test_validating_correction_updates_field(self, mgr, ctx):
        ctx.state = ConversationState.VALIDATING
        ctx.collected_data["nationality"] = "uae_national"
        mgr._transition(ctx, "change the nationality to Indian", "")
        assert ctx.state == ConversationState.COLLECTING_INFO
        assert ctx.collected_data["nationality"] == "Indian"

    def test_validating_correction_arabic(self, mgr, ctx):
        ctx.state = ConversationState.VALIDATING
        ctx.collected_data["nationality"] = "uae_national"
        ctx.language = "ar"
        mgr._transition(ctx, "غير الجنسية إلى هندي", "")
        assert ctx.state == ConversationState.COLLECTING_INFO
        assert ctx.collected_data["nationality"] == "هندي"

    def test_validating_correction_unknown_field_stays_validating(self, mgr, ctx, monkeypatch):
        # "something" doesn't match any known field alias, so _extract_correction
        # (regex) finds nothing. Mock the LLM fallback to also find nothing —
        # deterministic regardless of live Ollama/Anthropic availability. Since
        # no field was actually corrected, the state must stay VALIDATING and
        # correction_applied must stay False, not silently claim success.
        monkeypatch.setattr(mgr, "_llm_extract_correction", lambda ctx, text: False)
        ctx.state = ConversationState.VALIDATING
        mgr._transition(ctx, "change something to value", "")
        assert ctx.state == ConversationState.VALIDATING
        assert ctx.correction_applied is False
        assert ctx.correction_no_target is True

    def test_completing_is_terminal(self, mgr, ctx):
        ctx.state = ConversationState.COMPLETING
        mgr._transition(ctx, "anything", "")
        assert ctx.state == ConversationState.COMPLETING


# ---------------------------------------------------------------------------
# Full turn — process_message
# ---------------------------------------------------------------------------

class TestProcessMessage:
    def test_returns_string_reply(self, mgr, ctx):
        reply = mgr.process_message(ctx, "I am a software engineer")
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_user_message_appended_to_history(self, mgr, ctx):
        mgr.process_message(ctx, "I am a nurse")
        user_msgs = [m for m in ctx.history if m["role"] == "user"]
        assert any("nurse" in m["content"] for m in user_msgs)

    def test_assistant_reply_appended_to_history(self, mgr, ctx):
        mgr.process_message(ctx, "Hello")
        assistant_msgs = [m for m in ctx.history if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1

    def test_state_transitions_after_turn(self, mgr, ctx):
        mgr.process_message(ctx, "Hello")
        # Greeting → Collecting after first turn
        assert ctx.state == ConversationState.COLLECTING_INFO

    def test_history_grows_with_each_turn(self, mgr, ctx):
        mgr.process_message(ctx, "Hello")
        mgr.process_message(ctx, "I am a nurse")
        assert len(ctx.history) == 4  # 2 user + 2 assistant
