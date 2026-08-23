"""
Tests for the reranker_model parameter added to ISICClassifier and
ISCEDClassifier (2026-08-23), mirroring ISCOClassifier's existing
reranker_model contract (see TestRerankerModelBehaviourPreserved in
test_isco_classifier.py): default (None) preserves prior behaviour
byte-for-byte; an explicit model string routes through get_llm_strict()
with no silent fallback -- a missing/unreachable pinned model raises
RuntimeError out of __init__ rather than substituting a different model.

ISCEDClassifier additionally gains a real LLM re-ranking step (previously
it had none at all -- "fully offline" was a hard invariant). These tests
confirm that invariant is now conditional: still true when reranker_model
is omitted, no longer true only when a caller explicitly opts in.

Fully offline -- no live LLM, Qdrant, or network calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.agents.isced_classifier import ISCEDClassifier
from backend.agents.isic_classifier import ISICClassifier


# ---------------------------------------------------------------------------
# ISICClassifier.reranker_model
# ---------------------------------------------------------------------------

class TestISICRerankerModel:
    def test_default_omitted_uses_plain_get_llm(self):
        get_llm = MagicMock(return_value=MagicMock())
        get_llm_strict = MagicMock(return_value=MagicMock())
        with patch("backend.agents.isic_classifier.get_llm", get_llm), \
             patch("backend.agents.isic_classifier.get_llm_strict", get_llm_strict):
            clf = ISICClassifier()
        get_llm.assert_called_once()
        get_llm_strict.assert_not_called()
        assert clf.reranker_model_resolved != ""

    def test_reranker_model_pin_uses_get_llm_strict_not_get_llm(self):
        get_llm = MagicMock(return_value=MagicMock())
        get_llm_strict = MagicMock(return_value=MagicMock())
        with patch("backend.agents.isic_classifier.get_llm", get_llm), \
             patch("backend.agents.isic_classifier.get_llm_strict", get_llm_strict):
            clf = ISICClassifier(reranker_model="gemini/gemini-3.6-flash")
        get_llm_strict.assert_called_once_with("gemini/gemini-3.6-flash", temperature=0.3)
        get_llm.assert_not_called()

    def test_reranker_model_resolved_reflects_pinned_model(self):
        fake_llm = MagicMock()
        fake_llm.model = "gemini/gemini-3.6-flash"
        with patch("backend.agents.isic_classifier.get_llm_strict", return_value=fake_llm):
            clf = ISICClassifier(reranker_model="gemini/gemini-3.6-flash")
        assert clf.reranker_model_resolved == "gemini/gemini-3.6-flash"

    def test_unreachable_pinned_model_raises_and_never_falls_back(self):
        """Fail-closed: a RuntimeError from get_llm_strict must propagate out
        of __init__, not be swallowed into a silent substitution."""
        with patch(
            "backend.agents.isic_classifier.get_llm_strict",
            side_effect=RuntimeError("GROQ_API_KEY is not set"),
        ):
            with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
                ISICClassifier(reranker_model="groq/openai/gpt-oss-120b")


# ---------------------------------------------------------------------------
# ISCEDClassifier.reranker_model
# ---------------------------------------------------------------------------

class TestISCEDRerankerModelConstruction:
    def test_default_omitted_never_touches_llm_client(self):
        get_llm_strict = MagicMock(return_value=MagicMock())
        with patch("backend.agents.isced_classifier.get_llm_strict", get_llm_strict):
            clf = ISCEDClassifier()
        get_llm_strict.assert_not_called()
        assert clf._llm is None
        assert clf.reranker_model_resolved == "none (no reranker configured)"

    def test_reranker_model_pin_uses_get_llm_strict(self):
        get_llm_strict = MagicMock(return_value=MagicMock())
        with patch("backend.agents.isced_classifier.get_llm_strict", get_llm_strict):
            ISCEDClassifier(reranker_model="gemini/gemini-3.6-flash")
        get_llm_strict.assert_called_once_with("gemini/gemini-3.6-flash", temperature=0.3)

    def test_reranker_model_resolved_reflects_pinned_model(self):
        fake_llm = MagicMock()
        fake_llm.model = "gemini/gemini-3.6-flash"
        with patch("backend.agents.isced_classifier.get_llm_strict", return_value=fake_llm):
            clf = ISCEDClassifier(reranker_model="gemini/gemini-3.6-flash")
        assert clf.reranker_model_resolved == "gemini/gemini-3.6-flash"

    def test_unreachable_pinned_model_raises_and_never_falls_back(self):
        with patch(
            "backend.agents.isced_classifier.get_llm_strict",
            side_effect=RuntimeError("GEMINI_API_KEY is not set"),
        ):
            with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
                ISCEDClassifier(reranker_model="gemini/gemini-3.6-flash")


class TestISCEDDefaultBehaviourUnchanged:
    """reranker_model omitted must reproduce the pre-existing, fully-offline
    keyword/rule pipeline exactly -- same as calling classify(text) before
    this parameter existed."""

    def test_unambiguous_field_stays_keyword_method(self):
        clf = ISCEDClassifier()
        result = clf.classify("Bachelor of Engineering in Computer Science, software development")
        assert result.method == "keyword"
        assert result.detailed_code == "0613"

    def test_ambiguous_field_still_never_calls_llm_when_no_reranker_configured(self):
        clf = ISCEDClassifier()
        with patch.object(clf, "_llm_rerank_field") as rerank:
            result = clf.classify("I studied at university")  # weak/ambiguous field signal
            rerank.assert_not_called()
        assert result.method == "keyword"

    def test_level_dimension_never_affected_by_reranker_model(self):
        """ISCED 2011 level must come from _score_level() regardless of
        whether a reranker is configured -- it's never part of the LLM path."""
        clf_plain = ISCEDClassifier()
        with patch("backend.agents.isced_classifier.get_llm_strict", return_value=MagicMock()):
            clf_with_reranker = ISCEDClassifier(reranker_model="gemini/gemini-3.6-flash")
        text = "Bachelor of Engineering in Computer Science"
        assert clf_plain.classify(text).level == clf_with_reranker.classify(text).level


class TestISCEDLlmRerankFieldPath:
    """Exercises the new opt-in LLM re-ranking step directly, mocking only
    the CrewAI Crew.kickoff() call boundary -- same isolation level as
    ISICClassifier's existing _llm_rerank tests."""

    def _make_clf(self):
        with patch("backend.agents.isced_classifier.get_llm_strict", return_value=MagicMock()):
            return ISCEDClassifier(reranker_model="gemini/gemini-3.6-flash")

    # Two candidates in a genuine near-tie -- gap (0.55-0.50=0.05) is below
    # _MIN_CANDIDATE_GAP (0.15), so this must be treated as ambiguous
    # regardless of real keyword-table content (mocked at the
    # _score_field_candidates boundary to stay independent of it).
    _AMBIGUOUS_CANDIDATES = [
        (0.55, {"broad_code": "06", "broad_title": "ICT", "narrow_code": "061", "narrow_title": "ICT",
                "detailed_code": "0613", "detailed_title": "Software development"}),
        (0.50, {"broad_code": "05", "broad_title": "Natural sciences", "narrow_code": "054", "narrow_title": "Maths",
                "detailed_code": "0541", "detailed_title": "Mathematics"}),
    ]

    def test_ambiguous_field_triggers_llm_rerank_and_changes_method(self):
        clf = self._make_clf()
        with patch.object(clf, "_score_field_candidates", return_value=self._AMBIGUOUS_CANDIDATES), \
             patch.object(
                 clf, "_llm_rerank_field",
                 return_value=({
                     "broad_code": "06", "broad_title": "Information and Communication Technologies",
                     "narrow_code": "061", "narrow_title": "Information and communication technologies",
                     "detailed_code": "0613", "detailed_title": "Software and applications development and analysis",
                 }, 0.9),
             ) as rerank:
            result = clf.classify("some ambiguous education description")
        rerank.assert_called_once()
        assert result.method == "llm"
        assert result.detailed_code == "0613"

    def test_unambiguous_field_skips_llm_even_when_reranker_configured(self):
        clf = self._make_clf()
        with patch.object(clf, "_llm_rerank_field") as rerank:
            result = clf.classify("Bachelor of Engineering in Computer Science, software development")
            rerank.assert_not_called()
        assert result.method == "keyword"

    def test_llm_rerank_failure_falls_back_to_keyword_result(self):
        clf = self._make_clf()
        with patch.object(clf, "_score_field_candidates", return_value=self._AMBIGUOUS_CANDIDATES), \
             patch.object(clf, "_llm_rerank_field", return_value=None):
            result = clf.classify("some ambiguous education description")
        assert result.method == "keyword"

    def test_parse_llm_field_response_matches_by_detailed_code(self):
        clf = self._make_clf()
        candidates = [
            {"broad_code": "06", "broad_title": "ICT", "narrow_code": "061", "narrow_title": "ICT",
             "detailed_code": "0613", "detailed_title": "Software development"},
            {"broad_code": "05", "broad_title": "Natural sciences", "narrow_code": "054", "narrow_title": "Maths",
             "detailed_code": "0541", "detailed_title": "Mathematics"},
        ]
        entry, conf = clf._parse_llm_field_response(
            '{"detailed_code": "0613", "confidence": 0.88, "reasoning": "software engineering fits best"}',
            candidates,
        )
        assert entry["detailed_code"] == "0613"
        assert conf == 0.88

    def test_parse_llm_field_response_unrecognised_code_falls_back_with_deflated_confidence(self):
        clf = self._make_clf()
        candidates = [
            {"broad_code": "06", "broad_title": "ICT", "narrow_code": "061", "narrow_title": "ICT",
             "detailed_code": "0613", "detailed_title": "Software development"},
        ]
        entry, conf = clf._parse_llm_field_response(
            '{"detailed_code": "9999", "confidence": 0.9}', candidates,
        )
        assert entry["detailed_code"] == "0613"
        assert conf == pytest.approx(0.63)

    def test_parse_llm_field_response_unparseable_json_returns_none(self):
        clf = self._make_clf()
        assert clf._parse_llm_field_response("not json at all", [{"detailed_code": "0613"}]) is None
