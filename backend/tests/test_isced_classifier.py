"""
Tests for backend/agents/isced_classifier.py

Fully offline — keyword-only classifier, no LLM required.
"""

import pytest

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
