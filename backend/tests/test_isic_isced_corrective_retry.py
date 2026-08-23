"""
Tests for ISICClassifier/ISCEDClassifier's enable_corrective_retry
(2026-08-24), ported from ISCOClassifier's corrective retry to close the
"same logic across ISCO-08/ISIC/ISCED" gap. Same gap-based accept rule,
same evidence trail (see ISCOClassifier._maybe_corrective_retry's
docstring): retrieve -> compare top1/top2 candidate-score gap -> accept
the retry only if its gap is strictly wider than the original's.

Fully offline -- no live LLM/Qdrant/network. Reformulation and scoring
are mocked at the same boundaries the reranker-parity tests already use.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.agents.isced_classifier import ISCEDClassifier
from backend.agents.isic_classifier import ISICClassifier


# ---------------------------------------------------------------------------
# ISICClassifier
# ---------------------------------------------------------------------------

class TestISICCorrectiveRetryDisabledByDefault:
    def test_default_false_never_calls_reformulation(self):
        with patch("backend.agents.isic_classifier.get_llm", return_value=MagicMock()):
            clf = ISICClassifier()
        with patch.object(clf, "_llm_reformulate_query") as reformulate:
            # Genuinely ambiguous input (real tie -- see test_isic_isced_reranker_model.py)
            clf.classify("management consultancy and advisory services")
            reformulate.assert_not_called()


class TestISICCorrectiveRetryEnabled:
    def _make_clf(self):
        with patch("backend.agents.isic_classifier.get_llm", return_value=MagicMock()):
            return ISICClassifier(enable_corrective_retry=True)

    def test_unambiguous_input_never_triggers_retry(self):
        clf = self._make_clf()
        with patch.object(clf, "_llm_reformulate_query") as reformulate:
            result = clf.classify("I sell things in a shop")  # clear winner, large gap
            reformulate.assert_not_called()
        assert result.method == "keyword"

    def test_ambiguous_input_with_wider_retry_gap_is_accepted(self):
        clf = self._make_clf()
        with patch.object(clf, "_llm_reformulate_query", return_value="management consultancy services specifically"), \
             patch.object(clf, "_keyword_score") as keyword_score:
            # First call: original ambiguous tie. Second call (retry text): clear winner.
            keyword_score.side_effect = [
                [(0.6667, {"class_code": "6619", "class_title": "Other financial services",
                           "division_code": "66", "section": "K", "section_title": "Finance",
                           "division_title": "Aux financial", "group_code": "661", "group_title": "Aux"}),
                 (0.6667, {"class_code": "7020", "class_title": "Management consultancy",
                           "division_code": "70", "section": "M", "section_title": "Prof",
                           "division_title": "Head offices", "group_code": "702", "group_title": "Mgmt"})],
                [(0.9, {"class_code": "7020", "class_title": "Management consultancy",
                        "division_code": "70", "section": "M", "section_title": "Prof",
                        "division_title": "Head offices", "group_code": "702", "group_title": "Mgmt"}),
                 (0.3, {"class_code": "6619", "class_title": "Other financial services",
                        "division_code": "66", "section": "K", "section_title": "Finance",
                        "division_title": "Aux financial", "group_code": "661", "group_title": "Aux"})],
            ]
            with patch.object(clf, "_llm_rerank", return_value=None):
                result = clf.classify("management consultancy and advisory services")

        assert result.method == "keyword_corrective"
        assert result.class_code == "7020"

    def test_retry_with_narrower_gap_is_rejected_original_stands(self):
        clf = self._make_clf()
        with patch.object(clf, "_llm_reformulate_query", return_value="still vague text"), \
             patch.object(clf, "_keyword_score") as keyword_score:
            tied = [(0.5, {"class_code": "AAAA", "class_title": "A", "division_code": "1",
                           "section": "X", "section_title": "X", "division_title": "X",
                           "group_code": "1", "group_title": "X"}),
                    (0.5, {"class_code": "BBBB", "class_title": "B", "division_code": "2",
                           "section": "Y", "section_title": "Y", "division_title": "Y",
                           "group_code": "2", "group_title": "Y"})]
            keyword_score.side_effect = [tied, tied]  # retry no better than original
            with patch.object(clf, "_llm_rerank", return_value=None):
                result = clf.classify("ambiguous text")

        assert "corrective" not in result.method

    def test_reformulation_returning_none_leaves_original_result(self):
        clf = self._make_clf()
        with patch.object(clf, "_llm_reformulate_query", return_value=None), \
             patch.object(clf, "_llm_rerank", return_value=None):
            result = clf.classify("management consultancy and advisory services")
        assert "corrective" not in result.method


# ---------------------------------------------------------------------------
# ISCEDClassifier
# ---------------------------------------------------------------------------

class TestISCEDCorrectiveRetryGating:
    def test_default_false_never_calls_reformulation(self):
        with patch("backend.agents.isced_classifier.get_llm_strict", return_value=MagicMock()):
            clf = ISCEDClassifier(reranker_model="gemini/gemini-3.6-flash")
        with patch.object(clf, "_llm_reformulate_field_query") as reformulate:
            clf.classify("medical and dental studies")  # real 4-way tie
            reformulate.assert_not_called()

    def test_enabled_but_no_reranker_configured_never_calls_reformulation(self):
        """enable_corrective_retry=True with no reranker_model means
        self._llm is None -- corrective retry must never fire without an
        LLM, same gate as the reranker step itself."""
        clf = ISCEDClassifier(enable_corrective_retry=True)
        assert clf._llm is None
        with patch.object(clf, "_llm_reformulate_field_query") as reformulate:
            clf.classify("medical and dental studies")
            reformulate.assert_not_called()


class TestISCEDCorrectiveRetryEnabled:
    _AMBIGUOUS = [
        (0.55, {"broad_code": "06", "broad_title": "ICT", "narrow_code": "061", "narrow_title": "ICT",
                "detailed_code": "0613", "detailed_title": "Software development"}),
        (0.50, {"broad_code": "05", "broad_title": "Natural sciences", "narrow_code": "054", "narrow_title": "Maths",
                "detailed_code": "0541", "detailed_title": "Mathematics"}),
    ]
    _CLEAR = [
        (0.95, {"broad_code": "06", "broad_title": "ICT", "narrow_code": "061", "narrow_title": "ICT",
                "detailed_code": "0613", "detailed_title": "Software development"}),
        (0.10, {"broad_code": "05", "broad_title": "Natural sciences", "narrow_code": "054", "narrow_title": "Maths",
                "detailed_code": "0541", "detailed_title": "Mathematics"}),
    ]

    def _make_clf(self):
        with patch("backend.agents.isced_classifier.get_llm_strict", return_value=MagicMock()):
            return ISCEDClassifier(reranker_model="gemini/gemini-3.6-flash", enable_corrective_retry=True)

    def test_ambiguous_field_with_wider_retry_gap_is_accepted(self):
        clf = self._make_clf()
        with patch.object(clf, "_score_field_candidates", side_effect=[self._AMBIGUOUS, self._CLEAR]), \
             patch.object(clf, "_llm_rerank_field", return_value=None), \
             patch.object(clf, "_llm_reformulate_field_query", return_value="software engineering specifically"):
            result = clf.classify("some ambiguous education description")

        assert "corrective" in result.method
        assert result.detailed_code == "0613"

    def test_level_dimension_never_affected_by_corrective_retry(self):
        clf = self._make_clf()
        with patch.object(clf, "_score_field_candidates", side_effect=[self._AMBIGUOUS, self._CLEAR]), \
             patch.object(clf, "_llm_rerank_field", return_value=None), \
             patch.object(clf, "_llm_reformulate_field_query", return_value="software engineering specifically"):
            result = clf.classify("Bachelor of Science, some ambiguous education description")
        assert result.level == 6  # bachelor's -- unaffected by the field-only retry

    def test_retry_with_narrower_gap_is_rejected(self):
        clf = self._make_clf()
        with patch.object(clf, "_score_field_candidates", side_effect=[self._AMBIGUOUS, self._AMBIGUOUS]), \
             patch.object(clf, "_llm_rerank_field", return_value=None), \
             patch.object(clf, "_llm_reformulate_field_query", return_value="still ambiguous"):
            result = clf.classify("some ambiguous education description")
        assert "corrective" not in result.method
