"""
Tests for backend/agents/isced_classifier.py

Fully offline — keyword-only classifier, no LLM required.
"""

import pytest

from backend.agents.classifier_methods import ISCEDF_HIERARCHICAL_RETRIEVAL
from backend.agents.isced_classifier import (
    ISCEDClassifier,
    ISCEDClassification,
    _ISCED_LEVELS,
)


@pytest.fixture
def clf():
    return ISCEDClassifier()


# ---------------------------------------------------------------------------
# 1. Data integrity
# ---------------------------------------------------------------------------

def test_isced_levels_0_to_8():
    levels = [e["level"] for e in _ISCED_LEVELS]
    assert levels == list(range(9))


def test_each_level_has_required_keys():
    required = {"level", "level_title", "keywords"}
    for entry in _ISCED_LEVELS:
        assert required.issubset(entry.keys())


# ---------------------------------------------------------------------------
# 2. classify()
# ---------------------------------------------------------------------------

def test_phd_returns_level_8(clf):
    result = clf.classify("PhD in Computer Science")
    assert result.level == 8


def test_masters_returns_level_7(clf):
    result = clf.classify("Master of Business Administration MBA")
    assert result.level == 7


def test_bachelor_returns_level_6(clf):
    result = clf.classify("Bachelor of Science BSc university")
    assert result.level == 6


def test_secondary_returns_level_3(clf):
    result = clf.classify("high school secondary school grade 11")
    assert result.level == 3


def test_primary_returns_level_1(clf):
    result = clf.classify("primary school elementary grade 5")
    assert result.level == 1


def test_arabic_secondary_returns_level_3(clf):
    result = clf.classify("ثانوية عامة")
    assert result.level == 3


def test_arabic_university_returns_level_6(clf):
    result = clf.classify("بكالوريوس جامعي")
    assert result.level == 6


def test_arabic_phd_returns_level_8(clf):
    result = clf.classify("دكتوراه")
    assert result.level == 8


def test_confidence_in_range(clf):
    result = clf.classify("Bachelor of Engineering")
    assert 0.0 <= result.confidence <= 1.0


def test_method_is_keyword(clf):
    result = clf.classify("Masters degree postgraduate")
    assert result.method == "keyword"


def test_returns_isced_classification(clf):
    result = clf.classify("university degree")
    assert isinstance(result, ISCEDClassification)


def test_empty_string_returns_default(clf):
    result = clf.classify("")
    assert isinstance(result, ISCEDClassification)
    assert result.confidence == 0.0


def test_whitespace_returns_default(clf):
    result = clf.classify("   ")
    assert isinstance(result, ISCEDClassification)
    assert result.confidence == 0.0


def test_level_title_is_string(clf):
    result = clf.classify("doctor doctorate PhD thesis")
    assert isinstance(result.level_title, str)
    assert len(result.level_title) > 0


def test_raw_text_preserved(clf):
    text = "Bachelor of Science in Engineering"
    result = clf.classify(text)
    assert result.raw_text == text


# ---------------------------------------------------------------------------
# 3. method= stub: iscedf_hierarchical_retrieval is not yet implemented
# ---------------------------------------------------------------------------

def test_method_none_default_path_unchanged(clf):
    """Omitting method= (or passing None explicitly) must be byte-for-byte
    identical to calling classify(text) as before this parameter existed."""
    r_default = clf.classify("Bachelor of Science BSc university")
    r_explicit_none = clf.classify("Bachelor of Science BSc university", method=None)
    assert r_default == r_explicit_none


def test_hierarchical_retrieval_method_returns_structured_not_implemented(clf):
    result = clf.classify("Bachelor of Science", method=ISCEDF_HIERARCHICAL_RETRIEVAL)
    assert isinstance(result, ISCEDClassification)
    assert result.method == ISCEDF_HIERARCHICAL_RETRIEVAL
    assert result.confidence == 0.0
    assert result.level == -1
    assert result.broad_code == ""
    assert result.detailed_code == ""
    assert "not yet implemented" in (result.raw_text or "").lower() \
        or "deferred" in (result.raw_text or "").lower()


def test_hierarchical_retrieval_method_does_not_call_scoring(clf):
    """The stub path must short-circuit before any real classification work."""
    with pytest.MonkeyPatch.context() as mp:
        called = {"level": False, "field": False}

        def fake_score_level(text):
            called["level"] = True
            return _ISCED_LEVELS[3], 0.3

        mp.setattr(clf, "_score_level", fake_score_level)
        clf.classify("Bachelor of Science", method=ISCEDF_HIERARCHICAL_RETRIEVAL)
        assert called["level"] is False


def test_unknown_method_value_falls_through_to_default_pipeline(clf):
    """A method= value that is NOT in NOT_IMPLEMENTED_METHODS is not a stub
    trigger -- it's ignored and the default pipeline runs."""
    result = clf.classify("Bachelor of Science BSc university", method="some_other_value")
    assert result.method == "keyword"
    assert result.level == 6
