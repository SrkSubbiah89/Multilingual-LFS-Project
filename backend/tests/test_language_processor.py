"""
Tests for backend/agents/language_processor.py

Pure logic (language detection, script segmentation, entity parsing) needs no
mocking.  Full-pipeline tests (process()) patch Agent and Crew at the module
level so CrewAI's Pydantic LLM validation is bypassed entirely.
"""

from unittest.mock import MagicMock

import pytest

from backend.agents.language_processor import (
    LFS_ENTITY_LABELS,
    LanguageProcessor,
    LanguageProcessorResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_crew(monkeypatch):
    """
    Patches Agent and Crew inside language_processor.
    Returns the Crew *instance* mock so tests can control kickoff() return values.
    """
    crew_instance = MagicMock()
    crew_instance.kickoff.return_value = "[]"   # default: empty NER result
    crew_class = MagicMock(return_value=crew_instance)

    monkeypatch.setattr("backend.agents.language_processor.Agent", MagicMock())
    monkeypatch.setattr("backend.agents.language_processor.Crew", crew_class)
    monkeypatch.setattr("backend.agents.language_processor.Task", MagicMock())
    return crew_instance


@pytest.fixture
def lp(monkeypatch, mock_crew):
    """LanguageProcessor with all CrewAI components patched out."""
    monkeypatch.setattr(
        "backend.agents.language_processor.get_llm",
        lambda *a, **kw: MagicMock(),
    )
    return LanguageProcessor()


# ---------------------------------------------------------------------------
# Language detection — _detect_language
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_clear_english_returns_en(self, lp):
        lang, conf = lp._detect_language("I work as a software engineer in Riyadh")
        assert lang == "en"
        assert 0.0 < conf <= 1.0

    def test_clear_arabic_returns_ar(self, lp):
        lang, conf = lp._detect_language("أنا مهندس برمجيات في شركة تقنية بالرياض")
        assert lang == "ar"
        assert 0.0 < conf <= 1.0

    def test_langdetect_exception_falls_back_gracefully(self, lp, monkeypatch):
        from langdetect import LangDetectException

        def raise_exc(_):
            raise LangDetectException(0, "")

        monkeypatch.setattr("backend.agents.language_processor.detect_langs", raise_exc)
        lang, conf = lp._detect_language("Hello world")
        assert lang in ("en", "ar", "other")
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Script fallback — _script_fallback
# ---------------------------------------------------------------------------

class TestScriptFallback:
    def test_majority_arabic_returns_ar(self, lp):
        lang, conf = lp._script_fallback("أنا أعمل في مستشفى كبير")
        assert lang == "ar"
        assert conf > 0.5

    def test_majority_latin_returns_en(self, lp):
        lang, conf = lp._script_fallback("I am a software developer")
        assert lang == "en"

    def test_no_alphabetic_chars_returns_other(self, lp):
        lang, conf = lp._script_fallback("123 456 789")
        assert lang == "other"
        assert conf == 0.5

    def test_half_arabic_returns_ar(self, lp):
        # ar_ratio = 0.5 → qualifies as "ar"
        lang, _ = lp._script_fallback("أب ab")
        assert lang == "ar"


# ---------------------------------------------------------------------------
# Code-switching — _segment_scripts
# ---------------------------------------------------------------------------

class TestSegmentScripts:
    def test_pure_english_not_code_switched(self, lp):
        is_cs, _, _, _ = lp._segment_scripts("I work as a nurse full time")
        assert is_cs is False

    def test_pure_arabic_not_code_switched(self, lp):
        is_cs, _, _, _ = lp._segment_scripts("أعمل ممرضًا في مستشفى حكومي")
        assert is_cs is False

    def test_mixed_text_exceeding_threshold_is_code_switched(self, lp):
        is_cs, _, _, _ = lp._segment_scripts("أنا software engineer في tech company")
        assert is_cs is True

    def test_arabic_and_latin_ratios_sum_to_one(self, lp):
        _, _, ar, lat = lp._segment_scripts("أنا engineer")
        assert abs(ar + lat - 1.0) < 1e-9

    def test_arabic_ratio_correct_for_pure_arabic(self, lp):
        _, _, ar, lat = lp._segment_scripts("مرحبا")
        assert ar == pytest.approx(1.0)
        assert lat == pytest.approx(0.0)

    def test_segments_list_not_empty(self, lp):
        _, segments, _, _ = lp._segment_scripts("Hello world")
        assert len(segments) > 0

    def test_each_segment_has_valid_script(self, lp):
        _, segments, _, _ = lp._segment_scripts("أنا engineer في شركة")
        for seg in segments:
            assert seg.script in ("arabic", "latin", "other")


# ---------------------------------------------------------------------------
# Segment building — _build_segments
# ---------------------------------------------------------------------------

class TestBuildSegments:
    def test_empty_string_returns_empty_list(self, lp):
        assert lp._build_segments("") == []

    def test_pure_latin_has_no_arabic_segment(self, lp):
        segs = lp._build_segments("hello world")
        assert all(s.script != "arabic" for s in segs)

    def test_arabic_segment_has_arabic_script(self, lp):
        segs = lp._build_segments("مرحبا")
        assert any(s.script == "arabic" for s in segs)

    def test_whitespace_does_not_create_empty_segments(self, lp):
        segs = lp._build_segments("   hello   ")
        for s in segs:
            assert s.text.strip() != ""

    def test_code_switched_text_yields_multiple_scripts(self, lp):
        segs = lp._build_segments("أنا engineer")
        scripts = {s.script for s in segs}
        assert "arabic" in scripts
        assert "latin" in scripts

    def test_short_segment_has_no_detected_language(self, lp):
        segs = lp._build_segments("ab")
        for s in segs:
            if len(s.text.strip()) < 4:
                assert s.detected_language is None


# ---------------------------------------------------------------------------
# Entity parsing — _parse_entities
# ---------------------------------------------------------------------------

class TestParseEntities:
    def test_valid_json_returns_entities(self, lp):
        raw = '[{"text": "nurse", "label": "JOB_TITLE", "language": "en"}]'
        entities = lp._parse_entities(raw, "I work as a nurse")
        assert len(entities) == 1
        assert entities[0].text == "nurse"
        assert entities[0].label == "JOB_TITLE"
        assert entities[0].language == "en"

    def test_strips_markdown_code_fence(self, lp):
        raw = '```json\n[{"text": "doctor", "label": "JOB_TITLE", "language": "en"}]\n```'
        entities = lp._parse_entities(raw, "I am a doctor")
        assert len(entities) == 1
        assert entities[0].label == "JOB_TITLE"

    def test_falls_back_to_regex_on_bad_json(self, lp):
        raw = 'Here: [{"text": "teacher", "label": "JOB_TITLE", "language": "en"}] done.'
        entities = lp._parse_entities(raw, "I am a teacher")
        assert len(entities) == 1

    def test_unknown_label_is_silently_dropped(self, lp):
        raw = '[{"text": "foo", "label": "UNKNOWN_LABEL", "language": "en"}]'
        assert lp._parse_entities(raw, "foo") == []

    def test_unknown_language_normalised_to_en(self, lp):
        raw = '[{"text": "nurse", "label": "JOB_TITLE", "language": "xyz"}]'
        entities = lp._parse_entities(raw, "I am a nurse")
        assert entities[0].language == "en"

    def test_adds_character_offsets(self, lp):
        text = "I work as a software engineer in Riyadh"
        raw  = '[{"text": "software engineer", "label": "JOB_TITLE", "language": "en"}]'
        entities = lp._parse_entities(raw, text)
        assert entities[0].start == text.find("software engineer")
        assert entities[0].end   == entities[0].start + len("software engineer")

    def test_entity_not_in_text_has_no_offsets(self, lp):
        raw = '[{"text": "astronaut", "label": "JOB_TITLE", "language": "en"}]'
        entities = lp._parse_entities(raw, "I am a nurse")
        assert entities[0].start is None
        assert entities[0].end   is None

    def test_empty_array_returns_empty_list(self, lp):
        assert lp._parse_entities("[]", "anything") == []

    def test_totally_malformed_returns_empty_list(self, lp):
        assert lp._parse_entities("not json at all", "anything") == []

    def test_multiple_entities_returned(self, lp):
        raw = (
            '[{"text": "nurse", "label": "JOB_TITLE", "language": "en"},'
            ' {"text": "Riyadh", "label": "LOCATION", "language": "en"}]'
        )
        entities = lp._parse_entities(raw, "I am a nurse in Riyadh")
        assert len(entities) == 2
        labels = {e.label for e in entities}
        assert "JOB_TITLE" in labels
        assert "LOCATION"  in labels

    def test_all_lfs_labels_are_accepted(self, lp):
        for label in LFS_ENTITY_LABELS:
            raw = f'[{{"text": "test", "label": "{label}", "language": "en"}}]'
            assert len(lp._parse_entities(raw, "test")) == 1


# ---------------------------------------------------------------------------
# Full pipeline — process()
# ---------------------------------------------------------------------------

_MOCK_NER_JSON = (
    '[{"text": "software engineer", "label": "JOB_TITLE", "language": "en"},'
    ' {"text": "Riyadh", "label": "LOCATION", "language": "en"}]'
)


class TestProcess:
    def test_empty_input_returns_default_result(self, lp):
        result = lp.process("")
        assert isinstance(result, LanguageProcessorResult)
        assert result.detected_language == "other"
        assert result.confidence == 0.0
        assert result.is_code_switched is False
        assert result.entities == []
        assert result.segments == []

    def test_whitespace_only_treated_as_empty(self, lp):
        assert lp.process("   ").detected_language == "other"

    def test_returns_language_processor_result(self, lp):
        result = lp.process("I work as a software engineer in Riyadh")
        assert isinstance(result, LanguageProcessorResult)

    def test_english_text_detected_as_en(self, lp):
        result = lp.process("I work as a nurse full time in London")
        assert result.detected_language == "en"

    def test_arabic_text_detected_as_ar(self, lp):
        result = lp.process("أنا أعمل ممرضًا في مستشفى حكومي")
        assert result.detected_language == "ar"

    def test_entities_extracted_from_ner_response(self, lp, mock_crew):
        mock_crew.kickoff.return_value = _MOCK_NER_JSON
        result = lp.process("I work as a software engineer in Riyadh")
        job_titles = [e for e in result.entities if e.label == "JOB_TITLE"]
        assert len(job_titles) == 1
        assert job_titles[0].text == "software engineer"

    def test_code_switched_text_flagged(self, lp):
        result = lp.process("أنا software engineer في tech company بالرياض")
        assert result.is_code_switched is True

    def test_confidence_in_range(self, lp):
        result = lp.process("I am a doctor")
        assert 0.0 <= result.confidence <= 1.0

    def test_raw_text_preserved_in_result(self, lp):
        text = "I am an accountant"
        assert lp.process(text).raw_text == text

    def test_arabic_and_latin_ratios_sum_to_one(self, lp):
        result = lp.process("Hello مرحبا")
        assert abs(result.arabic_ratio + result.latin_ratio - 1.0) < 1e-6
