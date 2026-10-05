"""Regressions for multilingual CrewAI prompts and respondent-source entities."""

import json
from unittest.mock import MagicMock

import pytest

from backend.agents import conversation_manager, language_processor, validation_agent


def _mock_framework(monkeypatch, module, output="[]"):
    agent_factory = MagicMock()
    task_factory = MagicMock()
    crew = MagicMock()
    crew.kickoff.return_value = output
    monkeypatch.setattr(module, "get_llm", MagicMock())
    monkeypatch.setattr(module, "Agent", agent_factory)
    monkeypatch.setattr(module, "Task", task_factory)
    monkeypatch.setattr(module, "Crew", MagicMock(return_value=crew))
    return agent_factory, task_factory


@pytest.mark.parametrize(
    "language, original, entity_text, translated_text",
    [
        ("tl", "Ako ay isang guro sa paaralan", "guro", "teacher"),
        ("ar-gulf", "أنا شاغل في شركة", "شاغل", "عامل"),
    ],
)
def test_normalisation_keeps_ner_entities_and_offsets_in_original_answer(
    monkeypatch, language, original, entity_text, translated_text,
):
    output = json.dumps([
        {"text": entity_text, "label": "JOB_TITLE", "language": language},
    ], ensure_ascii=False)
    _, tasks = _mock_framework(monkeypatch, language_processor, output)
    processor = language_processor.LanguageProcessor()
    monkeypatch.setattr(processor, "_detect_language", lambda text: (language, 0.99))

    result = processor.process(original)

    assert result.raw_text == original
    assert translated_text in result.normalised_text
    entity = result.entities[0]
    assert entity.text == entity_text
    assert entity.language == language
    assert result.raw_text[entity.start:entity.end] == entity_text
    description = tasks.call_args.kwargs["description"]
    assert f'Original survey message:\n"""\n{original}\n"""' in description
    assert "Normalised wording for interpretation only" in description
    assert "Do not substitute translations" in description


def test_hindi_segment_has_language_label_in_code_switched_answer(monkeypatch):
    _mock_framework(monkeypatch, language_processor)
    processor = language_processor.LanguageProcessor()

    result = processor.process("मैं अस्पताल में काम करता हूँ as a nurse")

    assert result.is_code_switched
    hindi_segments = [segment for segment in result.segments if segment.script == "devanagari"]
    assert hindi_segments
    assert all(segment.detected_language == "hi" for segment in hindi_segments)


@pytest.mark.parametrize(
    "language, output_language",
    [("ur", "Urdu"), ("hi", "Hindi"), ("tl", "Filipino/Tagalog")],
)
def test_conversation_task_output_agrees_with_selected_interview_language(
    monkeypatch, language, output_language,
):
    _, tasks = _mock_framework(monkeypatch, conversation_manager)
    manager = conversation_manager.ConversationManager()
    context = manager.new_context(session_id=1, language=language)
    context.state = conversation_manager.ConversationState.COLLECTING_INFO

    manager._build_task(context)

    prompt = tasks.call_args.kwargs
    assert output_language in prompt["expected_output"]
    assert "in English" not in prompt["expected_output"]
    assert conversation_manager._LANG_RESPONSE_INSTRUCTION[language] in prompt["description"]
    expected_question = manager._questions_and_ack_for_lang(language)[0]["employment_status"]
    assert expected_question in prompt["description"]


@pytest.mark.parametrize(
    "language, input_language",
    [
        ("en", "English"), ("ar", "Arabic"), ("ar-gulf", "Gulf Arabic"),
        ("ur", "Urdu"), ("hi", "Hindi"), ("tl", "Tagalog (Filipino)"),
    ],
)
def test_semantic_validation_identifies_actual_input_language(
    monkeypatch, language, input_language,
):
    output = json.dumps({
        "is_semantically_consistent": True, "confidence": 0.9, "issues": [],
        "explanation_en": "Consistent.", "explanation_ar": "الإجابات متسقة.",
    })
    _, tasks = _mock_framework(monkeypatch, validation_agent, output)
    agent = validation_agent.ValidationAgent()

    valid, _, issues, _, _ = agent._semantic_validate({"job_title": "nurse"}, language)

    assert valid
    assert issues == []
    assert f"The survey was conducted in {input_language}." in tasks.call_args.kwargs["description"]


@pytest.mark.parametrize("module, cls", [
    (language_processor, language_processor.LanguageProcessor),
    (conversation_manager, conversation_manager.ConversationManager),
])
def test_core_agent_role_covers_all_supported_interview_languages(monkeypatch, module, cls):
    agents, _ = _mock_framework(monkeypatch, module)
    cls()

    goal = agents.call_args.kwargs["goal"]
    for language in ("English", "Arabic", "Urdu", "Hindi", "Tagalog"):
        assert language in goal


@pytest.mark.parametrize("language, answer", [
    ("en", "My highest qualification is a bachelor's degree."),
    ("ar", "أكملت بكالوريوس في الجامعة"),
    ("ur", "میں نے بیچلر ڈگری مکمل کی ہے"),
    ("hi", "मैंने स्नातक की डिग्री पूरी की है"),
    ("tl", "Nakatapos ako ng batsilyer sa kolehiyo"),
])
def test_natural_bachelors_answer_opens_field_of_study(monkeypatch, language, answer):
    _mock_framework(monkeypatch, conversation_manager)
    manager = conversation_manager.ConversationManager()
    manager._agent_available = False
    context = manager.new_context(session_id=1, language=language)
    context.state = conversation_manager.ConversationState.COLLECTING_INFO
    context.collected_data = {"employment_status": "employed"}

    reply = manager.process_message(context, answer)

    assert context.collected_data == {"employment_status": "employed", "education_level": "bachelor"}
    next_field = next(field for field in manager._get_field_order(context.collected_data) if field not in context.collected_data)
    assert next_field == "field_of_study"
    assert manager._questions_and_ack_for_lang(language)[0]["field_of_study"] in reply


@pytest.mark.parametrize("language, answer", [
    ("en", "I have not completed a bachelor's degree."),
    ("ar", "لم أكمل بكالوريوس"),
    ("ur", "میں نے بیچلر ڈگری مکمل نہیں کی"),
    ("hi", "मैंने स्नातक की डिग्री पूरी नहीं की है"),
    ("tl", "Hindi ako nakatapos ng batsilyer"),
    ("en", "It was a bachelor's degree or master's degree."),
    ("ur", "بیچلر ڈگری یا ماسٹر ڈگری ہے"),
    ("hi", "मेरी पढ़ाई स्नातक या स्नातकोत्तर है"),
    ("tl", "Batsilyer o diploma ang aking edukasyon"),
])
def test_negated_or_conflicting_qualification_asks_for_clarification(monkeypatch, language, answer):
    _mock_framework(monkeypatch, conversation_manager)
    manager = conversation_manager.ConversationManager()
    manager._agent_available = False
    context = manager.new_context(session_id=1, language=language)
    context.state = conversation_manager.ConversationState.COLLECTING_INFO
    context.collected_data = {"employment_status": "employed"}

    manager.process_message(context, answer)

    assert context.collected_data == {"employment_status": "employed"}
    assert context.state == conversation_manager.ConversationState.CLARIFYING
    assert context.clarification_target == "education_level"


@pytest.mark.parametrize("language, answer", [
    ("en", "I have no formal education."),
    ("ur", "میری کوئی رسمی تعلیم نہیں ہے"),
    ("hi", "मुझे कोई औपचारिक शिक्षा नहीं मिली"),
    ("tl", "Walang pormal na edukasyon ang natapos ko"),
])
def test_negative_education_option_is_a_valid_natural_answer(monkeypatch, language, answer):
    _mock_framework(monkeypatch, conversation_manager)
    manager = conversation_manager.ConversationManager()
    context = manager.new_context(session_id=1, language=language)
    context.state = conversation_manager.ConversationState.COLLECTING_INFO
    context.collected_data = {"employment_status": "employed"}

    manager._extract_fields(context, answer)

    assert context.collected_data == {"employment_status": "employed", "education_level": "no_formal"}
    assert "field_of_study" not in manager._get_field_order(context.collected_data)


def test_localized_option_phrase_does_not_match_hindi_word_prefix():
    assert conversation_manager.ConversationManager._match_current_option_phrase(
        "education_level", "मैं स्नातकों के साथ काम करता हूँ",
    ) == (None, False)


def test_localized_option_phrase_only_matches_current_question():
    assert conversation_manager.ConversationManager._match_current_option_phrase(
        "employment_status", "मैंने स्नातक की डिग्री पूरी की है",
    ) == (None, False)


@pytest.mark.parametrize("language, change", [
    ("ur", "{label} تبدیل کریں"),
    ("hi", "{label} बदलें"),
    ("tl", "Baguhin ang {label}"),
])
def test_native_field_name_reaches_multilingual_correction_extractor(monkeypatch, language, change):
    _mock_framework(monkeypatch, conversation_manager)
    manager = conversation_manager.ConversationManager()
    context = manager.new_context(session_id=1, language=language)
    context.state = conversation_manager.ConversationState.VALIDATING
    context.collected_data = {"employment_status": "employed", "education_level": "secondary"}
    field_label = manager._field_labels_for_lang(language)["education_level"]
    answer = change.format(label=field_label)
    assert manager._wants_correction(answer, language)

    def apply_correction(ctx, message):
        ctx.collected_data["education_level"] = "bachelor"
        ctx.corrected_fields.add("education_level")
        return True

    extractor = MagicMock(side_effect=apply_correction)
    monkeypatch.setattr(manager, "_llm_extract_correction", extractor)

    manager._transition(context, answer, "")

    extractor.assert_called_once_with(context, answer)
    assert context.collected_data["education_level"] == "bachelor"
    assert "field_of_study" in manager._get_field_order(context.collected_data)


@pytest.mark.parametrize("language, answer", [
    ("ur", "نہیں، غلطی درست کریں"),
    ("hi", "नहीं, बदलाव करें"),
    ("tl", "Hindi, gusto kong baguhin"),
])
def test_correction_without_native_field_name_stays_guarded(language, answer):
    assert not conversation_manager.ConversationManager._mentions_known_field(answer)
