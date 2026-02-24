"""
Tests for backend/agents/conversation_manager.py

Pure logic (field extraction, ambiguity detection, confirmation detection,
FSM transitions) is tested without any mocking.  The full process_message()
turn patches Crew.kickoff to avoid real LLM calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.agents.conversation_manager import (
    ConversationContext,
    ConversationManager,
    ConversationState,
    _REQUIRED_FIELDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mgr(monkeypatch):
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
    """Context with all required fields already collected."""
    c = mgr.new_context(session_id=2, language="en")
    c.collected_data = {
        "employment_status": "employed",
        "job_title":         "software engineer",
        "industry":          "technology",
        "hours_per_week":    "40",
        "employment_type":   "full_time",
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

class TestExtractFieldsJobTitle:
    def test_engineer_keyword_captures_raw_text(self, mgr, ctx):
        mgr._extract_fields(ctx, "I am a civil engineer")
        assert ctx.collected_data.get("job_title") is not None

    def test_doctor_keyword_captured(self, mgr, ctx):
        mgr._extract_fields(ctx, "I work as a doctor in a clinic")
        assert ctx.collected_data.get("job_title") is not None

    def test_arabic_doctor_keyword_captured(self, mgr, ctx):
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
        full_ctx.state = ConversationState.COLLECTING_INFO
        # Use non-ambiguous input (len > 5 and not a vague keyword)
        mgr._transition(full_ctx, "I work full time 40 hours a week in technology", "")
        assert full_ctx.state == ConversationState.VALIDATING

    def test_collecting_moves_to_clarifying_on_ambiguous_input(self, mgr, ctx):
        ctx.state = ConversationState.COLLECTING_INFO
        ctx.collected_data = {}
        mgr._transition(ctx, "yes", "Could you clarify?")
        assert ctx.state == ConversationState.CLARIFYING

    def test_clarifying_returns_to_collecting(self, mgr, ctx):
        ctx.state = ConversationState.CLARIFYING
        mgr._transition(ctx, "I meant I work as a nurse", "Thank you!")
        assert ctx.state == ConversationState.COLLECTING_INFO

    def test_validating_moves_to_completing_on_confirmation(self, mgr, ctx):
        ctx.state = ConversationState.VALIDATING
        mgr._transition(ctx, "yes, that's correct", "")
        assert ctx.state == ConversationState.COMPLETING

    def test_validating_returns_to_collecting_on_non_confirmation(self, mgr, ctx):
        ctx.state = ConversationState.VALIDATING
        mgr._transition(ctx, "actually my industry is healthcare not tech", "")
        assert ctx.state == ConversationState.COLLECTING_INFO

    def test_completing_is_terminal(self, mgr, ctx):
        ctx.state = ConversationState.COMPLETING
        mgr._transition(ctx, "anything", "")
        assert ctx.state == ConversationState.COMPLETING


# ---------------------------------------------------------------------------
# Full turn — process_message
# ---------------------------------------------------------------------------

_MOCK_REPLY = "Thank you! Can you tell me how many hours you work per week?"


class TestProcessMessage:
    def test_returns_string_reply(self, mgr, ctx):
        with patch("crewai.Crew.kickoff", return_value=_MOCK_REPLY):
            reply = mgr.process_message(ctx, "I am a software engineer")
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_user_message_appended_to_history(self, mgr, ctx):
        with patch("crewai.Crew.kickoff", return_value=_MOCK_REPLY):
            mgr.process_message(ctx, "I am a nurse")
        user_msgs = [m for m in ctx.history if m["role"] == "user"]
        assert any("nurse" in m["content"] for m in user_msgs)

    def test_assistant_reply_appended_to_history(self, mgr, ctx):
        with patch("crewai.Crew.kickoff", return_value=_MOCK_REPLY):
            mgr.process_message(ctx, "Hello")
        assistant_msgs = [m for m in ctx.history if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1

    def test_state_transitions_after_turn(self, mgr, ctx):
        with patch("crewai.Crew.kickoff", return_value=_MOCK_REPLY):
            mgr.process_message(ctx, "Hello")
        # Greeting → Collecting after first turn
        assert ctx.state == ConversationState.COLLECTING_INFO

    def test_history_grows_with_each_turn(self, mgr, ctx):
        with patch("crewai.Crew.kickoff", return_value=_MOCK_REPLY):
            mgr.process_message(ctx, "Hello")
            mgr.process_message(ctx, "I am a nurse")
        assert len(ctx.history) == 4  # 2 user + 2 assistant
