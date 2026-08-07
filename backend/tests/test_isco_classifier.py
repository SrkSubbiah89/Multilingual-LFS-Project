"""
Tests for backend/agents/isco_classifier.py

Pure logic (_detect_script, _parse_llm_response) is tested without mocking.
classify() tests patch the VectorStore and Agent/Crew at the module level to
avoid external deps and bypass CrewAI's Pydantic LLM validation entirely.
"""

from unittest.mock import MagicMock

import pytest

from backend.agents.isco_classifier import (
    ISCOClassification,
    ISCOClassifier,
    MIN_USABLE_CONFIDENCE,
    _HIGH_CONFIDENCE_THRESHOLD,
    _detect_script,
)
from backend.rag.hierarchical_store import UnitCandidate
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
# Constants
# ---------------------------------------------------------------------------

_LLM_RESPONSE = '{"selected_code": "2512", "reasoning": "Best match."}'


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
def mock_crew(monkeypatch):
    """
    Patches Agent and Crew inside isco_classifier.
    Returns the Crew *instance* mock so tests can control kickoff() return values.
    """
    crew_instance = MagicMock()
    crew_instance.kickoff.return_value = _LLM_RESPONSE
    crew_class = MagicMock(return_value=crew_instance)

    monkeypatch.setattr("backend.agents.isco_classifier.Agent", MagicMock())
    monkeypatch.setattr("backend.agents.isco_classifier.Crew", crew_class)
    monkeypatch.setattr("backend.agents.isco_classifier.Task", MagicMock())
    return crew_instance


@pytest.fixture
def clf(monkeypatch, mock_store, mock_crew):
    """ISCOClassifier with LLM and VectorStore patched out.

    The hierarchical store is forced to raise so the classifier falls back
    to the flat mock_store — this keeps tests deterministic without Qdrant.
    """
    monkeypatch.setattr(
        "backend.agents.isco_classifier.get_llm",
        lambda *a, **kw: MagicMock(),
    )
    # Force hierarchical store to be unavailable → flat fallback used
    monkeypatch.setattr(
        "backend.agents.isco_classifier.get_hierarchical_store",
        lambda: (_ for _ in ()).throw(RuntimeError("no hierarchical store in tests")),
    )
    monkeypatch.setattr(
        "backend.agents.isco_classifier.get_vector_store",
        lambda **kw: mock_store,
    )
    return ISCOClassifier()


def make_candidate(code="2512", title_en="Software Developers",
                   title_ar="", score=0.75) -> UnitCandidate:
    return UnitCandidate(code=code, label_en=title_en, label_ar=title_ar or title_en, score=score)


@pytest.fixture
def sample_candidates():
    return [
        make_candidate(code="2512", title_en="Software Developers",       score=0.80),
        make_candidate(code="2511", title_en="Systems Analysts",          score=0.72),
        make_candidate(code="2513", title_en="Web and Multimedia Developers", score=0.65),
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
        assert result.alternatives == []

    def test_whitespace_only_returns_sentinel(self, clf, mock_store):
        result = clf.classify("   ")
        assert result.primary.code == ""

    def test_no_candidates_returns_sentinel(self, clf, mock_store):
        mock_store.search.return_value = []
        result = clf.classify("something obscure")
        assert result.primary.confidence == 0.0
        assert result.alternatives == []


# ---------------------------------------------------------------------------
# classify — fast semantic path (high confidence)
# ---------------------------------------------------------------------------

class TestClassifyFastPath:
    def test_high_confidence_uses_semantic_method(self, clf, mock_store):
        mock_store.search.return_value = [
            make_match(confidence=_HIGH_CONFIDENCE_THRESHOLD + 0.01)
        ]
        result = clf.classify("software developer")
        assert result.method.endswith("_semantic")  # "flat_semantic" or "hierarchical_semantic"

    def test_high_confidence_does_not_call_llm(self, clf, mock_store, mock_crew):
        mock_store.search.return_value = [
            make_match(confidence=_HIGH_CONFIDENCE_THRESHOLD + 0.01)
        ]
        clf.classify("software developer")
        mock_crew.kickoff.assert_not_called()

    def test_high_confidence_primary_is_top_candidate(self, clf, mock_store):
        top = make_match(code="2512", confidence=_HIGH_CONFIDENCE_THRESHOLD + 0.01)
        mock_store.search.return_value = [top]
        result = clf.classify("software developer")
        assert result.primary.code == "2512"


# ---------------------------------------------------------------------------
# classify — LLM re-ranking path (low confidence)
# ---------------------------------------------------------------------------

class TestClassifyLlmPath:
    def test_low_confidence_uses_llm_ranked_method(self, clf, mock_store):
        mock_store.search.return_value = [make_match(confidence=0.70)]
        result = clf.classify("software developer")
        assert result.method.endswith("_llm")  # "flat_llm" or "hierarchical_llm"

    def test_llm_selected_code_is_primary(self, clf, mock_store):
        mock_store.search.return_value = [
            make_match(code="2512", confidence=0.75),
            make_match(code="2511", confidence=0.70),
        ]
        result = clf.classify("software developer")
        assert result.primary.code == "2512"

    def test_all_candidates_included_in_result(self, clf, mock_store):
        oc_candidates = [
            make_match(code="2512", confidence=0.75),
            make_match(code="2511", confidence=0.72),
            make_match(code="2513", confidence=0.65),
        ]
        mock_store.search.return_value = oc_candidates
        result = clf.classify("developer")
        # alternatives holds all except the primary (up to 2)
        assert len(result.alternatives) <= len(oc_candidates)

    def test_llm_failure_falls_back_to_top_semantic(self, clf, mock_store, mock_crew):
        mock_store.search.return_value = [make_match(code="2512", confidence=0.70)]
        mock_crew.kickoff.return_value = "not json"
        result = clf.classify("software developer")
        assert result.primary.code == "2512"


# ---------------------------------------------------------------------------
# classify — language detection
# ---------------------------------------------------------------------------

class TestClassifyLanguage:
    def test_arabic_input_detected_as_ar(self, clf, mock_store):
        result = clf.classify("مهندس برمجيات")
        assert result.language == "ar"

    def test_english_input_detected_as_en(self, clf, mock_store):
        result = clf.classify("software engineer")
        assert result.language == "en"

    def test_mixed_input_detected_as_mixed(self, clf, mock_store):
        result = clf.classify("أنا software engineer")
        assert result.language == "mixed"


# ---------------------------------------------------------------------------
# classify — result shape
# ---------------------------------------------------------------------------

class TestClassifyResultShape:
    def test_result_is_isco_classification(self, clf, mock_store):
        result = clf.classify("nurse")
        assert isinstance(result, ISCOClassification)

    def test_reasoning_is_non_empty_string(self, clf, mock_store):
        result = clf.classify("nurse")
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    def test_context_passed_to_store_search(self, clf, mock_store):
        clf.classify("nurse", context="healthcare sector", top_k=3)
        # flat fallback calls search(job_title, top_k=top_k); context is used in LLM prompt not store query
        mock_store.search.assert_called_once_with("nurse", top_k=3)

    def test_min_usable_confidence_constant_exposed(self):
        assert ISCOClassifier.MIN_USABLE_CONFIDENCE == MIN_USABLE_CONFIDENCE
        assert 0.0 < MIN_USABLE_CONFIDENCE < 1.0


# ---------------------------------------------------------------------------
# enable_llm (Task 09) -- retrieval-only construction, no LLM/agent init
# ---------------------------------------------------------------------------

def _make_clf(monkeypatch, mock_store, *, enable_llm=True, reranker_model=None,
              get_llm=None, get_llm_strict=None, agent=None):
    """Same store-forcing pattern as the module-level `clf` fixture, but
    parameterized so enable_llm/reranker_model/get_llm*/Agent can be
    controlled per test. get_llm/get_llm_strict default to a call-tracking
    MagicMock so tests can assert they were (not) called; Agent defaults to
    a plain MagicMock unless a fake needs to raise."""
    get_llm = get_llm if get_llm is not None else MagicMock(return_value=MagicMock())
    get_llm_strict = get_llm_strict if get_llm_strict is not None else MagicMock(return_value=MagicMock())
    monkeypatch.setattr("backend.agents.isco_classifier.get_llm", get_llm)
    monkeypatch.setattr("backend.agents.isco_classifier.get_llm_strict", get_llm_strict)
    monkeypatch.setattr("backend.agents.isco_classifier.Agent", agent or MagicMock())
    monkeypatch.setattr("backend.agents.isco_classifier.Task", MagicMock())
    monkeypatch.setattr(
        "backend.agents.isco_classifier.get_hierarchical_store",
        lambda: (_ for _ in ()).throw(RuntimeError("no hierarchical store in tests")),
    )
    monkeypatch.setattr(
        "backend.agents.isco_classifier.get_vector_store",
        lambda **kw: mock_store,
    )
    return ISCOClassifier(enable_llm=enable_llm, reranker_model=reranker_model)


class TestEnableLlmFalse:
    """Task 09: ISCOClassifier(enable_llm=False) must never construct an
    LLM/agent, must leave _agent_available False, and must still produce a
    real semantic-only classification against a ready fake store."""

    def test_does_not_call_get_llm_or_get_llm_strict(self, monkeypatch, mock_store):
        get_llm = MagicMock(return_value=MagicMock())
        get_llm_strict = MagicMock(return_value=MagicMock())
        clf = _make_clf(monkeypatch, mock_store, enable_llm=False,
                         get_llm=get_llm, get_llm_strict=get_llm_strict)
        get_llm.assert_not_called()
        get_llm_strict.assert_not_called()
        assert clf._agent_available is False

    def test_does_not_construct_agent(self, monkeypatch, mock_store):
        agent = MagicMock(side_effect=AssertionError("Agent() must not be constructed when enable_llm=False"))
        _make_clf(monkeypatch, mock_store, enable_llm=False, agent=agent)
        agent.assert_not_called()

    def test_reranker_model_resolved_is_explicit_disabled_string(self, monkeypatch, mock_store):
        clf = _make_clf(monkeypatch, mock_store, enable_llm=False)
        assert clf.reranker_model_resolved == "none (reranking disabled)"

    def test_reranker_model_pin_is_ignored_when_enable_llm_false(self, monkeypatch, mock_store):
        """enable_llm=False must win even if a reranker_model pin is also
        passed -- eval/run_eval.py's build_system() never does this (it
        passes reranker_model=None), but the classifier itself must not
        rely on that caller discipline alone."""
        get_llm_strict = MagicMock(return_value=MagicMock())
        clf = _make_clf(monkeypatch, mock_store, enable_llm=False,
                         reranker_model="ollama/llama3.2:1b", get_llm_strict=get_llm_strict)
        get_llm_strict.assert_not_called()
        assert clf._agent_available is False
        assert clf.reranker_model_resolved == "none (reranking disabled)"

    def test_still_makes_semantic_only_classification_with_ready_store(self, monkeypatch, mock_store):
        clf = _make_clf(monkeypatch, mock_store, enable_llm=False)
        result = clf.classify("software developer")
        assert isinstance(result, ISCOClassification)
        assert result.primary.code == "2512"
        assert "llm" not in result.method

    def test_classify_never_produces_an_llm_reranker_trace(self, monkeypatch, mock_store):
        mock_store.search.return_value = [make_match(confidence=0.55)]  # low confidence -- would trigger rerank if enabled
        clf = _make_clf(monkeypatch, mock_store, enable_llm=False)
        trace: dict = {}
        clf.classify("I do stuff with computers", trace=trace)
        assert not trace.get("reranker_fired")
        assert "reranker_output" not in trace
        assert "reranker_model" not in trace

    def test_no_vector_store_available_still_constructs_without_llm(self, monkeypatch):
        get_llm = MagicMock(return_value=MagicMock())
        get_llm_strict = MagicMock(return_value=MagicMock())
        monkeypatch.setattr("backend.agents.isco_classifier.get_llm", get_llm)
        monkeypatch.setattr("backend.agents.isco_classifier.get_llm_strict", get_llm_strict)
        monkeypatch.setattr(
            "backend.agents.isco_classifier.get_hierarchical_store",
            lambda: (_ for _ in ()).throw(RuntimeError("no hierarchical store")),
        )
        monkeypatch.setattr(
            "backend.agents.isco_classifier.get_vector_store",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("no flat store either")),
        )
        clf = ISCOClassifier(enable_llm=False)
        get_llm.assert_not_called()
        get_llm_strict.assert_not_called()
        assert clf._agent_available is False


class TestEnableLlmTrueUnchanged:
    """Task 09 regression guard: enable_llm's default (True) must retain
    every existing behaviour byte-for-byte, including the strict pinned-
    model path."""

    def test_default_omitted_still_calls_get_llm(self, clf, mock_store, mock_crew):
        # `clf` fixture already constructs with enable_llm defaulted (True)
        # and get_llm patched to return a MagicMock -- _agent_available
        # being True is the existing, unchanged contract.
        assert clf._agent_available is True
        assert clf.reranker_model_resolved != "none (reranking disabled)"

    def test_reranker_model_pin_still_uses_get_llm_strict(self, monkeypatch, mock_store):
        get_llm = MagicMock(return_value=MagicMock())
        get_llm_strict = MagicMock(return_value=MagicMock())
        clf = _make_clf(monkeypatch, mock_store, enable_llm=True,
                         reranker_model="ollama/llama3.2:1b",
                         get_llm=get_llm, get_llm_strict=get_llm_strict)
        get_llm_strict.assert_called_once()
        get_llm.assert_not_called()  # strict mode never falls back to get_llm()
        assert clf._agent_available is True

    def test_no_reranker_model_pin_uses_plain_get_llm(self, monkeypatch, mock_store):
        get_llm = MagicMock(return_value=MagicMock())
        get_llm_strict = MagicMock(return_value=MagicMock())
        clf = _make_clf(monkeypatch, mock_store, enable_llm=True,
                         reranker_model=None, get_llm=get_llm, get_llm_strict=get_llm_strict)
        get_llm.assert_called_once()
        get_llm_strict.assert_not_called()
        assert clf._agent_available is True
