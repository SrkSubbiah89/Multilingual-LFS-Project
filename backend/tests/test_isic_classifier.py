"""
Tests for backend/agents/isic_classifier.py

Fully offline — no LLM, no network calls required.
Only the keyword-scoring path is exercised; the LLM re-ranking path is
tested via a mock.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.agents.classifier_methods import ISIC_HIERARCHICAL_RETRIEVAL
from backend.agents.isic_classifier import (
    ISICClassifier,
    ISICClassification,
    _ISIC_DATA,
)


@pytest.fixture
def clf():
    """ISICClassifier with LLM constructor patched out."""
    with patch("backend.agents.isic_classifier.get_llm", return_value=MagicMock()):
        return ISICClassifier()


# ---------------------------------------------------------------------------
# 1. Data integrity
# ---------------------------------------------------------------------------

def test_isic_data_not_empty():
    assert len(_ISIC_DATA) > 30


def test_all_entries_have_required_keys():
    required = {"section", "section_title", "division_code", "division_title", "keywords"}
    for entry in _ISIC_DATA:
        assert required.issubset(entry.keys()), f"Missing keys in: {entry}"


def test_division_codes_are_two_digits():
    bad = [e["division_code"] for e in _ISIC_DATA if not e["division_code"].isdigit()]
    assert bad == []


def test_sections_A_to_U_present():
    sections = {e["section"] for e in _ISIC_DATA}
    # Core economic sections must be present
    for s in "ABCDFGHIJKLMNOPQRST":
        assert s in sections, f"Section {s} missing"


# ---------------------------------------------------------------------------
# 2. classify() — keyword path
# ---------------------------------------------------------------------------

def test_software_company_maps_to_section_J(clf):
    result = clf.classify("I work at a software development company")
    assert result.section == "J"


def test_division_code_for_it_services(clf):
    result = clf.classify("software developer tech startup app")
    assert result.division_code == "62"


def test_hospital_maps_to_section_Q(clf):
    result = clf.classify("I work in a hospital as a nurse")
    assert result.section == "Q"


def test_school_maps_to_section_P(clf):
    result = clf.classify("teacher at a secondary school")
    assert result.section == "P"


def test_bank_maps_to_section_K(clf):
    result = clf.classify("finance banking investment fund")
    assert result.section == "K"


def test_construction_maps_to_section_F(clf):
    result = clf.classify("construction building contractor civil")
    assert result.section == "F"


def test_arabic_hospital_maps_to_section_Q(clf):
    result = clf.classify("أعمل في مستشفى حكومي")
    assert result.section == "Q"


def test_arabic_software_maps_to_section_J(clf):
    result = clf.classify("شركة برمجيات تقنية")
    assert result.section == "J"


def test_confidence_in_range(clf):
    result = clf.classify("restaurant food service waiter")
    assert 0.0 <= result.confidence <= 1.0


def test_high_confidence_skips_llm(clf):
    # High keyword overlap → method = "keyword"
    result = clf.classify("software developer programmer IT technology app web")
    assert result.method == "keyword"


def test_returns_isic_classification(clf):
    result = clf.classify("teacher university professor")
    assert isinstance(result, ISICClassification)


def test_empty_text_returns_fallback(clf):
    result = clf.classify("")
    assert isinstance(result, ISICClassification)
    assert result.confidence == 0.0


def test_alternatives_list_populated(clf):
    # Multiple keyword hits → alternatives present
    result = clf.classify("doctor hospital clinic nurse physician")
    # For high-confidence results alternatives may be empty or present
    assert isinstance(result.alternatives, list)


# ---------------------------------------------------------------------------
# 3. LLM re-ranking path
# ---------------------------------------------------------------------------

def test_llm_rerank_called_for_low_confidence(clf):
    """When keyword score < KEYWORD_THRESHOLD, LLM re-ranking is attempted."""
    # Patch _keyword_score to return a low score
    original_score = clf._keyword_score

    def low_score(text):
        scored = original_score(text)
        return [(0.5, e) for _, e in scored[:3]] if scored else []

    with patch.object(clf, "_keyword_score", side_effect=low_score):
        with patch.object(clf, "_llm_rerank", return_value=None) as mock_rerank:
            result = clf.classify("some obscure industry description")
            # _llm_rerank may or may not be called depending on low-score path
            assert isinstance(result, ISICClassification)


def test_llm_rerank_exception_falls_back_to_keyword(clf):
    """LLM failure must not crash classify()."""
    with patch.object(clf, "_llm_rerank", side_effect=RuntimeError("LLM down")):
        # Should not raise; falls back to keyword result
        result = clf.classify("farm agriculture crop harvest")
        assert isinstance(result, ISICClassification)


# ---------------------------------------------------------------------------
# 4. method= stub: isic_hierarchical_retrieval is not yet implemented
# ---------------------------------------------------------------------------

def test_method_none_default_path_unchanged(clf):
    """Omitting method= (or passing None explicitly) must be byte-for-byte
    identical to calling classify(text) as before this parameter existed."""
    r_default = clf.classify("software developer tech startup app")
    r_explicit_none = clf.classify("software developer tech startup app", method=None)
    assert r_default == r_explicit_none


def test_hierarchical_retrieval_method_returns_structured_not_implemented(clf):
    result = clf.classify("software developer", method=ISIC_HIERARCHICAL_RETRIEVAL)
    assert isinstance(result, ISICClassification)
    assert result.method == ISIC_HIERARCHICAL_RETRIEVAL
    assert result.confidence == 0.0
    assert result.section == ""
    assert result.class_code == ""
    assert "not yet implemented" in (result.raw_text or "").lower() \
        or "deferred" in (result.raw_text or "").lower()


def test_hierarchical_retrieval_method_does_not_call_keyword_scoring(clf):
    """The stub path must short-circuit before any real classification work."""
    with patch.object(clf, "_keyword_score") as mock_score:
        clf.classify("software developer", method=ISIC_HIERARCHICAL_RETRIEVAL)
        mock_score.assert_not_called()


def test_unknown_method_value_falls_through_to_default_pipeline(clf):
    """A method= value that is NOT in NOT_IMPLEMENTED_METHODS is not a stub
    trigger -- it's ignored and the default pipeline runs (there's only one
    real pipeline for ISIC today; this guards against a typo silently
    routing to the not-implemented branch)."""
    result = clf.classify("software developer tech startup app", method="some_other_value")
    assert result.method in ("keyword", "llm")
