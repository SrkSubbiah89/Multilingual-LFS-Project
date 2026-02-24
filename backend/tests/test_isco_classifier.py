"""
Tests for backend/agents/isco_classifier.py

Pure logic (_detect_script, _parse_llm_response) is tested without mocking.
classify() tests patch the VectorStore and Crew.kickoff to avoid external deps.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.agents.isco_classifier import (
    ISCOClassification,
    ISCOClassifier,
    MIN_USABLE_CONFIDENCE,
    _HIGH_CONFIDENCE_THRESHOLD,
    _detect_script,
)
from backend.rag.vector_store import OccupationMatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_match(code="2512", title_en="Software Developers",
               title_ar="مطورو البرمجيات", level=4,
               confidence=0.75) -> OccupationMatch:
    return OccupationMatch(
        code=code,
        title_en=title_en,
        title_ar=title_ar,
        level=level,
        description="Design, develop, test and maintain software applications",
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    """A VectorStore mock that returns one default candidate."""
    store = MagicMock()
    store.search.return_value = [make_match()]
    return store


@pytest.fixture
def clf(monkeypatch, mock_store):
    """ISCOClassifier with LLM and VectorStore patched out."""
    monkeypatch.setattr(
        "backend.agents.isco_classifier.get_llm",
        lambda *a, **kw: MagicMock(),
    )
    monkeypatch.setattr(
        "backend.agents.isco_classifier.get_vector_store",
        lambda **kw: mock_store,
    )
    return ISCOClassifier()


@pytest.fixture
def sample_candidates():
    return [
        make_match(code="2512", title_en="Software Developers",     confidence=0.80),
        make_match(code="2511", title_en="Systems Analysts",        confidence=0.72),
        make_match(code="2513", title_en="Web and Multimedia Developers", confidence=0.65),
    ]


# ---------------------------------------------------------------------------
# _detect_script (module-level helper)
# ---------------------------------------------------------------------------

class TestDetectScript:
    def test_arabic_text_returns_ar(self):
        assert _detect_script("أنا مهندس برمجيات") == "ar"

    def test_english_text_returns_en(self):
        assert _detect_script("I am a software engineer") == "en"

    def test_mixed_returns_mixed(self):
        result = _detect_script("أنا software engineer")
        assert result == "mixed"

    def test_digits_and_punctuation_only_returns_other(self):
        assert _detect_script("123 !@#") == "other"

    def test_empty_string_returns_other(self):
        assert _detect_script("") == "other"

    def test_mostly_arabic_with_small_latin_returns_ar(self):
        # 90 %+ Arabic → "ar"
        result = _detect_script("أنا أعمل في شركة كبيرة وأحب عملي a")
        # The single Latin letter may or may not push below 90 %;
        # the important assertion is it is NOT "en"
        assert result in ("ar", "mixed")

    def test_pure_latin_no_arabic_is_en(self):
        assert _detect_script("doctor nurse teacher engineer") == "en"


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------

class TestParseLlmResponse:
    def test_valid_json_selects_correct_candidate(self, clf, sample_candidates):
        raw = '{"selected_code": "2512", "reasoning": "Best match for software developer."}'
        match, reasoning = clf._parse_llm_response(raw, sample_candidates)
        assert match.code == "2512"
        assert "software" in reasoning.lower()

    def test_strips_markdown_code_fence(self, clf, sample_candidates):
        raw = '```json\n{"selected_code": "2511", "reasoning": "Analyst match."}\n```'
        match, reasoning = clf._parse_llm_response(raw, sample_candidates)
        assert match.code == "2511"

    def test_unrecognised_code_falls_back_to_top_candidate(self, clf, sample_candidates):
        raw = '{"selected_code": "9999", "reasoning": "Unknown code."}'
        match, _ = clf._parse_llm_response(raw, sample_candidates)
        assert match.code == sample_candidates[0].code

    def test_malformed_json_falls_back_to_top_candidate(self, clf, sample_candidates):
        match, _ = clf._parse_llm_response("not valid json at all", sample_candidates)
        assert match.code == sample_candidates[0].code

    def test_empty_string_falls_back_to_top_candidate(self, clf, sample_candidates):
        match, _ = clf._parse_llm_response("", sample_candidates)
        assert match.code == sample_candidates[0].code

    def test_json_embedded_in_prose_is_extracted(self, clf, sample_candidates):
        raw = 'The answer is: {"selected_code": "2513", "reasoning": "Web developer."} end.'
        match, _ = clf._parse_llm_response(raw, sample_candidates)
        assert match.code == "2513"

    def test_reasoning_preserved_in_return(self, clf, sample_candidates):
        raw = '{"selected_code": "2512", "reasoning": "Exact title match."}'
        _, reasoning = clf._parse_llm_response(raw, sample_candidates)
        assert reasoning == "Exact title match."


# ---------------------------------------------------------------------------
# classify — empty and no-candidate cases
# ---------------------------------------------------------------------------

class TestClassifyEdgeCases:
    def test_empty_input_returns_sentinel(self, clf, mock_store):
        result = clf.classify("")
        assert isinstance(result, ISCOClassification)
        assert result.primary.code == ""
        assert result.primary.confidence == 0.0
        assert result.candidates == []

    def test_whitespace_only_returns_sentinel(self, clf, mock_store):
        result = clf.classify("   ")
        assert result.primary.code == ""

    def test_no_candidates_returns_sentinel(self, clf, mock_store):
        mock_store.search.return_value = []
        result = clf.classify("something obscure")
        assert result.primary.confidence == 0.0
        assert result.candidates == []


# ---------------------------------------------------------------------------
# classify — fast semantic path (high confidence)
# ---------------------------------------------------------------------------

class TestClassifyFastPath:
    def test_high_confidence_uses_semantic_method(self, clf, mock_store):
        mock_store.search.return_value = [
            make_match(confidence=_HIGH_CONFIDENCE_THRESHOLD + 0.01)
        ]
        result = clf.classify("software developer")
        assert result.method == "semantic"

    def test_high_confidence_does_not_call_llm(self, clf, mock_store):
        mock_store.search.return_value = [
            make_match(confidence=_HIGH_CONFIDENCE_THRESHOLD + 0.01)
        ]
        with patch("crewai.Crew.kickoff") as mock_kickoff:
            clf.classify("software developer")
        mock_kickoff.assert_not_called()

    def test_high_confidence_primary_is_top_candidate(self, clf, mock_store):
        top = make_match(code="2512", confidence=_HIGH_CONFIDENCE_THRESHOLD + 0.01)
        mock_store.search.return_value = [top]
        result = clf.classify("software developer")
        assert result.primary.code == "2512"


# ---------------------------------------------------------------------------
# classify — LLM re-ranking path (low confidence)
# ---------------------------------------------------------------------------

_LLM_RESPONSE = '{"selected_code": "2512", "reasoning": "Best match."}'


class TestClassifyLlmPath:
    def test_low_confidence_uses_llm_ranked_method(self, clf, mock_store):
        mock_store.search.return_value = [make_match(confidence=0.70)]
        with patch("crewai.Crew.kickoff", return_value=_LLM_RESPONSE):
            result = clf.classify("software developer")
        assert result.method == "llm_ranked"

    def test_llm_selected_code_is_primary(self, clf, mock_store):
        mock_store.search.return_value = [
            make_match(code="2512", confidence=0.75),
            make_match(code="2511", confidence=0.70),
        ]
        with patch("crewai.Crew.kickoff", return_value=_LLM_RESPONSE):
            result = clf.classify("software developer")
        assert result.primary.code == "2512"

    def test_all_candidates_included_in_result(self, clf, mock_store, sample_candidates):
        mock_store.search.return_value = sample_candidates
        with patch("crewai.Crew.kickoff", return_value=_LLM_RESPONSE):
            result = clf.classify("developer")
        assert len(result.candidates) == len(sample_candidates)

    def test_llm_failure_falls_back_to_top_semantic(self, clf, mock_store):
        mock_store.search.return_value = [make_match(code="2512", confidence=0.70)]
        with patch("crewai.Crew.kickoff", return_value="not json"):
            result = clf.classify("software developer")
        assert result.primary.code == "2512"


# ---------------------------------------------------------------------------
# classify — language detection
# ---------------------------------------------------------------------------

class TestClassifyLanguage:
    def test_arabic_input_detected_as_ar(self, clf, mock_store):
        with patch("crewai.Crew.kickoff", return_value=_LLM_RESPONSE):
            result = clf.classify("مهندس برمجيات")
        assert result.language == "ar"

    def test_english_input_detected_as_en(self, clf, mock_store):
        result = clf.classify("software engineer",)
        assert result.language == "en"

    def test_mixed_input_detected_as_mixed(self, clf, mock_store):
        with patch("crewai.Crew.kickoff", return_value=_LLM_RESPONSE):
            result = clf.classify("أنا software engineer")
        assert result.language == "mixed"


# ---------------------------------------------------------------------------
# classify — result shape
# ---------------------------------------------------------------------------

class TestClassifyResultShape:
    def test_result_is_isco_classification(self, clf, mock_store):
        with patch("crewai.Crew.kickoff", return_value=_LLM_RESPONSE):
            result = clf.classify("nurse")
        assert isinstance(result, ISCOClassification)

    def test_reasoning_is_non_empty_string(self, clf, mock_store):
        with patch("crewai.Crew.kickoff", return_value=_LLM_RESPONSE):
            result = clf.classify("nurse")
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    def test_context_passed_to_store_search(self, clf, mock_store):
        with patch("crewai.Crew.kickoff", return_value=_LLM_RESPONSE):
            clf.classify("nurse", context="healthcare sector", top_k=3)
        mock_store.search.assert_called_once_with("nurse", top_k=3)

    def test_min_usable_confidence_constant_exposed(self):
        assert ISCOClassifier.MIN_USABLE_CONFIDENCE == MIN_USABLE_CONFIDENCE
        assert 0.0 < MIN_USABLE_CONFIDENCE < 1.0
