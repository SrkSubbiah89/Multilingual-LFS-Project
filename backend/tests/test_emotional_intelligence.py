"""
backend/tests/test_emotional_intelligence.py

Tests for EmotionalIntelligence agent.

Coverage
--------
- EmotionalState / SurveyAction enum values
- _detect_script() module-level helper
- _detect_emotion_rules() rule-based stage (keywords, punctuation, caps, Arabic)
- _determine_action() static action mapper
- _build_fallback_analysis() deterministic fallback builder
- _parse_llm_response() JSON parsing + fallbacks
- analyze() public API (happy path + LLM error path)
- get_emotional_intelligence() singleton
"""
import pytest

import backend.agents.emotional_intelligence as _ei_module
from backend.agents.emotional_intelligence import (
    EmotionalAnalysis,
    EmotionalIntelligence,
    EmotionalSignal,
    EmotionalState,
    SurveyAction,
    _detect_script,
    get_emotional_intelligence,
    _ADAPTED_PROMPTS,
    _SUPPORT_MESSAGES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ei(monkeypatch):
    """EmotionalIntelligence instance with LLM / CrewAI components mocked out."""
    monkeypatch.setattr(_ei_module, "get_llm", lambda *a, **kw: object())

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(_ei_module, "Agent", FakeAgent)
    return EmotionalIntelligence()


def _mock_crew(monkeypatch, response: str) -> None:
    """Patch Crew and Task so kickoff() returns *response* without hitting the API."""

    class FakeTask:
        def __init__(self, **kwargs):
            pass

    class FakeCrew:
        def __init__(self, **kwargs):
            pass

        def kickoff(self):
            return response

    monkeypatch.setattr(_ei_module, "Task", FakeTask)
    monkeypatch.setattr(_ei_module, "Crew", FakeCrew)


# ---------------------------------------------------------------------------
# TestEmotionalStateEnum
# ---------------------------------------------------------------------------


class TestEmotionalStateEnum:
    def test_stressed_value(self):
        assert EmotionalState.STRESSED == "stressed"

    def test_confused_value(self):
        assert EmotionalState.CONFUSED == "confused"

    def test_frustrated_value(self):
        assert EmotionalState.FRUSTRATED == "frustrated"

    def test_neutral_value(self):
        assert EmotionalState.NEUTRAL == "neutral"

    def test_engaged_value(self):
        assert EmotionalState.ENGAGED == "engaged"


# ---------------------------------------------------------------------------
# TestSurveyActionEnum
# ---------------------------------------------------------------------------


class TestSurveyActionEnum:
    def test_continue_value(self):
        assert SurveyAction.CONTINUE == "continue"

    def test_slow_down_value(self):
        assert SurveyAction.SLOW_DOWN == "slow_down"

    def test_pause_value(self):
        assert SurveyAction.PAUSE == "pause"

    def test_end_value(self):
        assert SurveyAction.END == "end"


# ---------------------------------------------------------------------------
# TestDetectScript
# ---------------------------------------------------------------------------


class TestDetectScript:
    def test_empty_string_returns_en(self):
        assert _detect_script("") == "en"

    def test_pure_english_returns_en(self):
        assert _detect_script("Hello world this is English") == "en"

    def test_pure_arabic_returns_ar(self):
        # No Latin characters at all
        assert _detect_script("مرحبا بالعالم") == "ar"

    def test_numbers_only_returns_en(self):
        # No alphabetic characters at all → total == 0 → "en"
        assert _detect_script("12345 67890") == "en"

    def test_mostly_arabic_returns_ar(self):
        # ~12 Arabic alpha chars vs 1 Latin → ar_ratio ≈ 0.923 ≥ 0.90
        assert _detect_script("مرحبا بالعالم a") == "ar"

    def test_mixed_returns_mixed(self):
        # roughly equal Arabic and Latin
        assert _detect_script("Hello مرحبا world") == "mixed"

    def test_low_arabic_returns_en(self):
        # 1 Arabic char among 12 Latin → ar_ratio ≈ 0.077 < 0.10
        assert _detect_script("Hello world ok ا") == "en"


# ---------------------------------------------------------------------------
# TestDetectEmotionRules
# ---------------------------------------------------------------------------


class TestDetectEmotionRules:
    def test_neutral_for_plain_text(self, ei):
        state, conf, intensity, _ = ei._detect_emotion_rules("The weather is nice today.")
        assert state == EmotionalState.NEUTRAL
        assert conf == 1.0
        assert intensity == 0.0

    def test_neutral_signals_empty(self, ei):
        _, _, _, signals = ei._detect_emotion_rules("The weather is nice today.")
        assert signals == []

    def test_stressed_keyword_detected(self, ei):
        state, _, _, _ = ei._detect_emotion_rules("I'm so stressed about this survey.")
        assert state == EmotionalState.STRESSED

    def test_confused_keyword_detected(self, ei):
        state, _, _, _ = ei._detect_emotion_rules("I don't understand the question at all.")
        assert state == EmotionalState.CONFUSED

    def test_frustrated_keyword_detected(self, ei):
        state, _, _, _ = ei._detect_emotion_rules("This is so frustrating!")
        assert state == EmotionalState.FRUSTRATED

    def test_engaged_keyword_detected(self, ei):
        state, _, _, _ = ei._detect_emotion_rules("Of course, I am happy to help.")
        assert state == EmotionalState.ENGAGED

    def test_triple_exclamation_adds_punctuation_signal(self, ei):
        _, _, _, signals = ei._detect_emotion_rules("Stop it!!!")
        types = [s.signal_type for s in signals]
        assert "punctuation" in types

    def test_triple_exclamation_boosts_frustrated(self, ei):
        # "enough" → frustrated keyword (+0.30) + "!!!" → punctuation (+0.20)
        state, _, _, _ = ei._detect_emotion_rules("Enough!!!")
        assert state == EmotionalState.FRUSTRATED

    def test_triple_question_marks_boosts_confused(self, ei):
        state, _, _, _ = ei._detect_emotion_rules("What does this mean???")
        assert state == EmotionalState.CONFUSED

    def test_caps_words_boost_frustrated(self, ei):
        # "THIS" "SURVEY" "WRONG" all have length > 2 and are uppercase alpha → caps signal
        # (no punctuation attached to caps words, "IS" excluded because len == 2)
        state, _, _, _ = ei._detect_emotion_rules("THIS SURVEY WRONG")
        assert state == EmotionalState.FRUSTRATED

    def test_arabic_stressed_pattern(self, ei):
        state, _, _, _ = ei._detect_emotion_rules("أنا متوتر جدًا من هذا الاستبيان.")
        assert state == EmotionalState.STRESSED

    def test_arabic_confused_pattern(self, ei):
        state, _, _, _ = ei._detect_emotion_rules("لا أفهم هذا السؤال.")
        assert state == EmotionalState.CONFUSED

    def test_keyword_signal_weight_is_correct(self, ei):
        _, _, _, signals = ei._detect_emotion_rules("I'm confused about this.")
        keyword_signals = [s for s in signals if s.signal_type == "keyword"]
        assert keyword_signals
        assert all(s.weight == 0.30 for s in keyword_signals)

    def test_confidence_capped_at_0_95(self, ei):
        # Multiple keyword hits can push score > 0.95; confidence must be capped
        text = "stressed overwhelmed anxious worried nervous panicking exhausted burned out scared"
        _, conf, _, _ = ei._detect_emotion_rules(text)
        assert conf <= 0.95

    def test_intensity_capped_at_1_0(self, ei):
        text = "stressed overwhelmed anxious worried nervous panicking exhausted burned out scared"
        _, _, intensity, _ = ei._detect_emotion_rules(text)
        assert intensity <= 1.0


# ---------------------------------------------------------------------------
# TestDetermineAction
# ---------------------------------------------------------------------------


class TestDetermineAction:
    def test_neutral_returns_continue(self):
        action, reason = EmotionalIntelligence._determine_action(EmotionalState.NEUTRAL, 0.5)
        assert action == SurveyAction.CONTINUE
        assert reason

    def test_engaged_returns_continue(self):
        action, _ = EmotionalIntelligence._determine_action(EmotionalState.ENGAGED, 0.8)
        assert action == SurveyAction.CONTINUE

    def test_confused_low_intensity_returns_slow_down(self):
        action, _ = EmotionalIntelligence._determine_action(EmotionalState.CONFUSED, 0.30)
        assert action == SurveyAction.SLOW_DOWN

    def test_confused_high_intensity_returns_pause(self):
        action, _ = EmotionalIntelligence._determine_action(EmotionalState.CONFUSED, 0.60)
        assert action == SurveyAction.PAUSE

    def test_stressed_low_intensity_returns_slow_down(self):
        action, _ = EmotionalIntelligence._determine_action(EmotionalState.STRESSED, 0.40)
        assert action == SurveyAction.SLOW_DOWN

    def test_stressed_high_intensity_returns_pause(self):
        action, _ = EmotionalIntelligence._determine_action(EmotionalState.STRESSED, 0.70)
        assert action == SurveyAction.PAUSE

    def test_frustrated_low_intensity_returns_slow_down(self):
        action, _ = EmotionalIntelligence._determine_action(EmotionalState.FRUSTRATED, 0.30)
        assert action == SurveyAction.SLOW_DOWN

    def test_frustrated_high_intensity_returns_end(self):
        action, _ = EmotionalIntelligence._determine_action(EmotionalState.FRUSTRATED, 0.75)
        assert action == SurveyAction.END

    def test_all_actions_have_reason_string(self):
        for state in EmotionalState:
            for intensity in (0.20, 0.70):
                _, reason = EmotionalIntelligence._determine_action(state, intensity)
                assert isinstance(reason, str) and reason


# ---------------------------------------------------------------------------
# TestBuildFallbackAnalysis
# ---------------------------------------------------------------------------


class TestBuildFallbackAnalysis:
    def test_returns_emotional_analysis_instance(self, ei):
        result = ei._build_fallback_analysis(
            "test", "en", EmotionalState.NEUTRAL, 0.9, 0.0, []
        )
        assert isinstance(result, EmotionalAnalysis)

    def test_en_adapted_prompt_from_template(self, ei):
        result = ei._build_fallback_analysis(
            "test", "en", EmotionalState.STRESSED, 0.6, 0.5, []
        )
        assert result.adapted_prompt_en == _ADAPTED_PROMPTS[EmotionalState.STRESSED]["en"]

    def test_ar_adapted_prompt_from_template(self, ei):
        result = ei._build_fallback_analysis(
            "test", "ar", EmotionalState.CONFUSED, 0.6, 0.4, []
        )
        assert result.adapted_prompt_ar == _ADAPTED_PROMPTS[EmotionalState.CONFUSED]["ar"]

    def test_support_messages_from_template(self, ei):
        result = ei._build_fallback_analysis(
            "test", "en", EmotionalState.FRUSTRATED, 0.7, 0.6, []
        )
        assert result.support_message_en == _SUPPORT_MESSAGES[EmotionalState.FRUSTRATED]["en"]
        assert result.support_message_ar == _SUPPORT_MESSAGES[EmotionalState.FRUSTRATED]["ar"]

    def test_propagates_signals(self, ei):
        sig = EmotionalSignal(signal_type="keyword", text="stressed", weight=0.30)
        result = ei._build_fallback_analysis(
            "test", "en", EmotionalState.STRESSED, 0.5, 0.5, [sig]
        )
        assert result.signals == [sig]

    def test_raw_text_preserved(self, ei):
        result = ei._build_fallback_analysis(
            "hello world", "en", EmotionalState.NEUTRAL, 1.0, 0.0, []
        )
        assert result.raw_text == "hello world"

    def test_action_from_high_frustration(self, ei):
        result = ei._build_fallback_analysis(
            "test", "en", EmotionalState.FRUSTRATED, 0.7, 0.70, []
        )
        assert result.survey_action == SurveyAction.END

    def test_action_from_low_stress(self, ei):
        result = ei._build_fallback_analysis(
            "test", "en", EmotionalState.STRESSED, 0.4, 0.40, []
        )
        assert result.survey_action == SurveyAction.SLOW_DOWN


# ---------------------------------------------------------------------------
# TestParseLLMResponse
# ---------------------------------------------------------------------------

_FULL_JSON = (
    '{"emotional_state": "stressed", "confidence": 0.8, "intensity": 0.7, '
    '"adapted_prompt_en": "Take it easy.", "adapted_prompt_ar": "خذ وقتك.", '
    '"support_message_en": "You can stop anytime.", '
    '"support_message_ar": "يمكنك التوقف.", '
    '"survey_action": "slow_down", "action_reason": "Mild stress detected."}'
)


class TestParseLLMResponse:
    def test_valid_json_state_parsed(self, ei):
        result = ei._parse_llm_response(
            _FULL_JSON, "test", "en", EmotionalState.NEUTRAL, 0.5, 0.0, []
        )
        assert result.emotional_state == EmotionalState.STRESSED

    def test_valid_json_confidence_parsed(self, ei):
        result = ei._parse_llm_response(
            _FULL_JSON, "test", "en", EmotionalState.NEUTRAL, 0.5, 0.0, []
        )
        assert result.confidence == 0.8

    def test_valid_json_action_parsed(self, ei):
        result = ei._parse_llm_response(
            _FULL_JSON, "test", "en", EmotionalState.NEUTRAL, 0.5, 0.0, []
        )
        assert result.survey_action == SurveyAction.SLOW_DOWN

    def test_valid_json_text_fields_parsed(self, ei):
        result = ei._parse_llm_response(
            _FULL_JSON, "test", "en", EmotionalState.NEUTRAL, 0.5, 0.0, []
        )
        assert result.adapted_prompt_en == "Take it easy."
        assert result.adapted_prompt_ar == "خذ وقتك."

    def test_markdown_fences_stripped(self, ei):
        fenced = f"```json\n{_FULL_JSON}\n```"
        result = ei._parse_llm_response(
            fenced, "test", "en", EmotionalState.NEUTRAL, 0.5, 0.0, []
        )
        assert result.emotional_state == EmotionalState.STRESSED

    def test_invalid_state_falls_back_to_rb_state(self, ei):
        raw = (
            '{"emotional_state": "UNKNOWN", "confidence": 0.8, "intensity": 0.5,'
            '"survey_action": "continue", "action_reason": "ok"}'
        )
        result = ei._parse_llm_response(
            raw, "test", "en", EmotionalState.CONFUSED, 0.6, 0.4, []
        )
        assert result.emotional_state == EmotionalState.CONFUSED

    def test_empty_string_triggers_fallback(self, ei):
        result = ei._parse_llm_response(
            "", "test", "en", EmotionalState.STRESSED, 0.7, 0.6, []
        )
        assert result.emotional_state == EmotionalState.STRESSED

    def test_invalid_json_triggers_fallback(self, ei):
        result = ei._parse_llm_response(
            "not json at all", "test", "en", EmotionalState.CONFUSED, 0.6, 0.4, []
        )
        assert result.emotional_state == EmotionalState.CONFUSED

    def test_missing_text_fields_use_templates(self, ei):
        raw = (
            '{"emotional_state": "neutral", "confidence": 0.9, "intensity": 0.0,'
            '"survey_action": "continue", "action_reason": "calm"}'
        )
        result = ei._parse_llm_response(
            raw, "test", "en", EmotionalState.NEUTRAL, 0.9, 0.0, []
        )
        assert result.adapted_prompt_en == _ADAPTED_PROMPTS[EmotionalState.NEUTRAL]["en"]

    def test_confidence_clamped_above_one(self, ei):
        raw = (
            '{"emotional_state": "neutral", "confidence": 99.0, "intensity": 0.0,'
            '"survey_action": "continue", "action_reason": "ok"}'
        )
        result = ei._parse_llm_response(
            raw, "test", "en", EmotionalState.NEUTRAL, 0.9, 0.0, []
        )
        assert result.confidence <= 1.0

    def test_json_embedded_in_surrounding_text_extracted(self, ei):
        raw = f"Here is my analysis: {_FULL_JSON} end."
        result = ei._parse_llm_response(
            raw, "test", "en", EmotionalState.NEUTRAL, 0.5, 0.0, []
        )
        assert result.emotional_state == EmotionalState.STRESSED

    def test_action_reason_from_data(self, ei):
        result = ei._parse_llm_response(
            _FULL_JSON, "test", "en", EmotionalState.NEUTRAL, 0.5, 0.0, []
        )
        assert result.action_reason == "Mild stress detected."

    def test_missing_action_reason_uses_determine_action(self, ei):
        raw = (
            '{"emotional_state": "stressed", "confidence": 0.7, "intensity": 0.4,'
            '"adapted_prompt_en": "ok", "adapted_prompt_ar": "حسنا",'
            '"support_message_en": "ok", "support_message_ar": "حسنا",'
            '"survey_action": "slow_down"}'
        )
        result = ei._parse_llm_response(
            raw, "test", "en", EmotionalState.STRESSED, 0.7, 0.4, []
        )
        assert isinstance(result.action_reason, str) and result.action_reason


# ---------------------------------------------------------------------------
# TestAnalyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_empty_text_returns_neutral(self, ei):
        result = ei.analyze("")
        assert result.emotional_state == EmotionalState.NEUTRAL
        assert result.survey_action == SurveyAction.CONTINUE

    def test_whitespace_only_returns_neutral(self, ei):
        result = ei.analyze("   \t\n  ")
        assert result.emotional_state == EmotionalState.NEUTRAL

    def test_empty_text_language_hint_en_preserved(self, ei):
        result = ei.analyze("", language="en")
        assert result.detected_language == "en"

    def test_empty_text_language_hint_ar_preserved(self, ei):
        result = ei.analyze("", language="ar")
        assert result.detected_language == "ar"

    def test_llm_result_propagated(self, ei, monkeypatch):
        _mock_crew(monkeypatch, (
            '{"emotional_state": "engaged", "confidence": 0.95, "intensity": 0.8,'
            '"adapted_prompt_en": "Great work!", "adapted_prompt_ar": "عمل رائع!",'
            '"support_message_en": "Keep going.", "support_message_ar": "استمر.",'
            '"survey_action": "continue", "action_reason": "Very cooperative."}'
        ))
        result = ei.analyze("Of course, happy to help!")
        assert result.emotional_state == EmotionalState.ENGAGED
        assert result.adapted_prompt_en == "Great work!"

    def test_llm_exception_triggers_fallback(self, ei, monkeypatch):
        class BrokenCrew:
            def __init__(self, **kwargs):
                pass

            def kickoff(self):
                raise RuntimeError("LLM unavailable")

        class FakeTask:
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr(_ei_module, "Crew", BrokenCrew)
        monkeypatch.setattr(_ei_module, "Task", FakeTask)
        # Should not raise — fallback with rule-based state
        result = ei.analyze("I'm so stressed about this!")
        assert isinstance(result, EmotionalAnalysis)
        assert result.emotional_state == EmotionalState.STRESSED

    def test_detected_language_set_in_result(self, ei, monkeypatch):
        _mock_crew(monkeypatch, (
            '{"emotional_state": "neutral", "confidence": 0.9, "intensity": 0.0,'
            '"adapted_prompt_en": "ok", "adapted_prompt_ar": "حسنا",'
            '"support_message_en": "ok", "support_message_ar": "حسنا",'
            '"survey_action": "continue", "action_reason": "neutral"}'
        ))
        result = ei.analyze("مرحبا بالعالم")
        assert result.detected_language == "ar"

    def test_signals_in_result(self, ei, monkeypatch):
        _mock_crew(monkeypatch, (
            '{"emotional_state": "confused", "confidence": 0.7, "intensity": 0.5,'
            '"adapted_prompt_en": "Let me help.", "adapted_prompt_ar": "دعني أساعدك.",'
            '"support_message_en": "ok", "support_message_ar": "حسنا",'
            '"survey_action": "slow_down", "action_reason": "confused"}'
        ))
        result = ei.analyze("I don't understand this question at all???")
        assert isinstance(result.signals, list)

    def test_raw_text_preserved_in_result(self, ei, monkeypatch):
        _mock_crew(monkeypatch, (
            '{"emotional_state": "neutral", "confidence": 0.9, "intensity": 0.0,'
            '"adapted_prompt_en": "ok", "adapted_prompt_ar": "حسنا",'
            '"support_message_en": "ok", "support_message_ar": "حسنا",'
            '"survey_action": "continue", "action_reason": "neutral"}'
        ))
        msg = "I work as a teacher."
        result = ei.analyze(msg)
        assert result.raw_text == msg

    def test_analyze_returns_emotional_analysis(self, ei, monkeypatch):
        _mock_crew(monkeypatch, (
            '{"emotional_state": "neutral", "confidence": 0.9, "intensity": 0.0,'
            '"adapted_prompt_en": "ok", "adapted_prompt_ar": "حسنا",'
            '"support_message_en": "ok", "support_message_ar": "حسنا",'
            '"survey_action": "continue", "action_reason": "neutral"}'
        ))
        result = ei.analyze("I am fine.")
        assert isinstance(result, EmotionalAnalysis)


# ---------------------------------------------------------------------------
# TestGetEmotionalIntelligence
# ---------------------------------------------------------------------------


class TestGetEmotionalIntelligence:
    def test_returns_emotional_intelligence_instance(self, monkeypatch):
        _ei_module._instance = None
        monkeypatch.setattr(_ei_module, "get_llm", lambda *a, **kw: object())

        class FakeAgent:
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr(_ei_module, "Agent", FakeAgent)
        result = get_emotional_intelligence()
        assert isinstance(result, EmotionalIntelligence)

    def test_returns_same_instance_on_second_call(self, monkeypatch):
        _ei_module._instance = None
        monkeypatch.setattr(_ei_module, "get_llm", lambda *a, **kw: object())

        class FakeAgent:
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr(_ei_module, "Agent", FakeAgent)
        a = get_emotional_intelligence()
        b = get_emotional_intelligence()
        assert a is b

    def test_reuses_existing_instance(self, monkeypatch):
        monkeypatch.setattr(_ei_module, "get_llm", lambda *a, **kw: object())

        class FakeAgent:
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr(_ei_module, "Agent", FakeAgent)
        sentinel = EmotionalIntelligence()
        _ei_module._instance = sentinel
        result = get_emotional_intelligence()
        assert result is sentinel
