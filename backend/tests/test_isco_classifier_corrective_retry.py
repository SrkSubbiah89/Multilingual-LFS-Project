"""
Tests for ISCOClassifier's experimental, opt-in corrective-retry feature
(enable_corrective_retry=, backend/agents/isco_classifier.py).

Thesis RAG-comparison work: when the normal (retrieval + optional rerank)
result is still below HITL_THRESHOLD, one additional retrieval attempt is
made with an LLM-reformulated query. The retry result replaces the
original ONLY if it strictly improves confidence; the feature is fully
opt-in (default False) and must not change behaviour for any existing
caller that omits it.
"""

from unittest.mock import MagicMock

import pytest

from backend.agents.isco_classifier import ISCOClassifier
from backend.rag.hierarchical_store import HierarchicalResult, UnitCandidate


def make_result(code, confidence, label_en="Some Occupation"):
    cand = UnitCandidate(code=code, label_en=label_en, label_ar=label_en, score=confidence)
    return HierarchicalResult(
        code=code,
        label_en=label_en,
        label_ar=label_en,
        confidence=confidence,
        stage_confidences={"stage1": confidence, "stage2": confidence, "stage3": confidence, "stage4": confidence},
        hierarchy_path=[code[0], code[:2], code[:3], code],
        top_candidates=[cand],
        hitl_required=confidence < 0.70,
        fallback_used=False,
    )


def make_result_with_candidates(candidate_scores: list[tuple[str, float]], label_en="Some Occupation"):
    """candidate_scores: ordered [(code, score), ...], best first -- lets a
    test control the top1/top2 gap directly."""
    top_code, top_score = candidate_scores[0]
    cands = [UnitCandidate(code=c, label_en=label_en, label_ar=label_en, score=s) for c, s in candidate_scores]
    return HierarchicalResult(
        code=top_code,
        label_en=label_en,
        label_ar=label_en,
        confidence=top_score,
        stage_confidences={"stage1": top_score, "stage2": top_score, "stage3": top_score, "stage4": top_score},
        hierarchy_path=[top_code[0], top_code[:2], top_code[:3], top_code],
        top_candidates=cands,
        hitl_required=top_score < 0.70,
        fallback_used=False,
    )


@pytest.fixture
def mock_hier_store():
    """A HierarchicalISCOStore mock whose .search() return value can be
    changed between calls via side_effect, to simulate a weak first result
    followed by a stronger (or weaker) corrective-retry result."""
    store = MagicMock()
    return store


@pytest.fixture
def mock_crew_sequence(monkeypatch):
    """Patches Agent/Task/Crew so each Crew().kickoff() call returns (or
    raises, if the queued item is an Exception instance/class -- native
    unittest.mock side_effect-list behaviour) the next value from a
    configurable queue. Lets a test control the rerank-selection response
    and the reformulation response separately."""
    responses: list = []
    crew_instance = MagicMock()
    crew_instance.kickoff.side_effect = responses
    crew_class = MagicMock(return_value=crew_instance)

    monkeypatch.setattr("backend.agents.isco_classifier.Agent", MagicMock())
    monkeypatch.setattr("backend.agents.isco_classifier.Crew", crew_class)
    monkeypatch.setattr("backend.agents.isco_classifier.Task", MagicMock())
    return responses


@pytest.fixture
def clf_factory(monkeypatch, mock_hier_store, mock_crew_sequence):
    """Returns a function that builds an ISCOClassifier with a genuinely
    "available" mocked hierarchical store (so _classify_hierarchical runs,
    not the legacy flat fallback) and a controllable LLM response queue."""
    monkeypatch.setattr("backend.agents.isco_classifier.get_llm", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("backend.agents.isco_classifier.get_hierarchical_store", lambda: mock_hier_store)

    def _build(**kwargs):
        return ISCOClassifier(**kwargs)
    return _build


class TestCorrectiveRetryDisabledByDefault:
    def test_default_false_never_calls_reformulation(self, clf_factory, mock_hier_store, mock_crew_sequence):
        weak = make_result("9999", 0.40)
        mock_hier_store.search.return_value = weak
        mock_crew_sequence.append('{"selected_code": "9999", "reasoning": "top pick"}')

        clf = clf_factory()  # enable_corrective_retry omitted -> False
        result = clf.classify("some vague title")

        assert result.primary.code == "9999"
        assert "corrective" not in result.method
        # Only one Crew().kickoff() call (the normal rerank) -- no reformulation attempt.
        assert mock_hier_store.search.call_count == 1


class TestCorrectiveRetryEnabled:
    def test_high_confidence_never_triggers_retry(self, clf_factory, mock_hier_store, mock_crew_sequence):
        strong = make_result("2512", 0.95)  # >= _HIGH_CONFIDENCE_THRESHOLD, skips rerank entirely
        mock_hier_store.search.return_value = strong

        clf = clf_factory(enable_corrective_retry=True)
        result = clf.classify("software developer")

        assert result.primary.code == "2512"
        assert "corrective" not in result.method
        assert mock_hier_store.search.call_count == 1  # no second (retry) search call

    def test_weak_result_triggers_retry_and_improvement_wins(self, clf_factory, mock_hier_store, mock_crew_sequence):
        weak = make_result("9999", 0.40, label_en="Vague Match")
        strong = make_result("2512", 0.85, label_en="Software Developers")
        mock_hier_store.search.side_effect = [weak, strong]

        mock_crew_sequence.append('{"selected_code": "9999", "reasoning": "best of a weak set"}')
        mock_crew_sequence.append('{"reformulated_query": "computer programmer software engineer"}')
        mock_crew_sequence.append('{"selected_code": "2512", "reasoning": "clear match after reformulation"}')

        clf = clf_factory(enable_corrective_retry=True)
        trace: dict = {}
        result = clf.classify("does computer stuff", trace=trace)

        assert result.primary.code == "2512"
        assert result.primary.confidence == 0.85
        assert "corrective" in result.method
        assert "[Corrective retry]" in result.reasoning
        assert mock_hier_store.search.call_count == 2
        assert trace["corrective_retry_attempted"] is True
        assert trace["corrective_retry_used"] is True
        assert trace["corrective_retry_query"] == "computer programmer software engineer"

    def test_weak_result_retry_no_improvement_keeps_original(self, clf_factory, mock_hier_store, mock_crew_sequence):
        weak = make_result("9999", 0.40, label_en="Vague Match")
        still_weak = make_result("8888", 0.35, label_en="Also Vague")
        mock_hier_store.search.side_effect = [weak, still_weak]

        mock_crew_sequence.append('{"selected_code": "9999", "reasoning": "best of a weak set"}')
        mock_crew_sequence.append('{"reformulated_query": "some other phrase"}')
        mock_crew_sequence.append('{"selected_code": "8888", "reasoning": "still not great"}')

        clf = clf_factory(enable_corrective_retry=True)
        trace: dict = {}
        result = clf.classify("does computer stuff", trace=trace)

        # Retry did not improve confidence (0.35 < 0.40) -- original kept.
        assert result.primary.code == "9999"
        assert result.primary.confidence == 0.40
        assert "corrective" not in result.method
        assert trace["corrective_retry_used"] is False

    def test_reformulation_llm_failure_keeps_original(self, clf_factory, mock_hier_store, mock_crew_sequence):
        weak = make_result("9999", 0.40)
        mock_hier_store.search.return_value = weak

        mock_crew_sequence.append('{"selected_code": "9999", "reasoning": "best of a weak set"}')
        mock_crew_sequence.append(RuntimeError("boom"))  # kickoff() raises on the reformulation attempt

        clf = clf_factory(enable_corrective_retry=True)
        trace: dict = {}
        result = clf.classify("does computer stuff", trace=trace)

        # _llm_reformulate_query's own try/except must catch the raised
        # exception and return None -- never propagate, never crash classify().
        assert result.primary.code == "9999"
        assert "corrective" not in result.method
        assert mock_hier_store.search.call_count == 1  # no retry search attempted
        assert trace["corrective_retry_attempted"] is True
        assert trace["corrective_retry_used"] is False


# ---------------------------------------------------------------------------
# Gap-aware confidence (use_gap_aware_confidence=)
# ---------------------------------------------------------------------------

class TestGapAwareConfidenceDisabledByDefault:
    def test_thin_gap_high_confidence_stays_not_hitl_when_disabled(self, clf_factory, mock_hier_store, mock_crew_sequence):
        # Confidence is high (0.85, skips reranking entirely) but the top-2
        # candidates are nearly tied (gap 0.003 < _MIN_TRUSTED_CANDIDATE_GAP).
        # Without use_gap_aware_confidence, this must NOT be flagged HITL --
        # byte-identical to pre-existing behaviour.
        thin_gap = make_result_with_candidates([("2512", 0.85), ("2511", 0.847)])
        mock_hier_store.search.return_value = thin_gap
        mock_crew_sequence.append('{"selected_code": "2512", "reasoning": "top pick"}')

        clf = clf_factory()  # both new flags omitted -> False
        trace: dict = {}
        result = clf.classify("ambiguous title", trace=trace)

        assert result.hitl_required is False
        assert trace["gap_ambiguous"] is True  # computed regardless...
        # ...but not allowed to change reported hitl_required when the flag is off.


class TestGapAwareConfidenceEnabled:
    def test_thin_gap_high_confidence_becomes_hitl_when_enabled(self, clf_factory, mock_hier_store, mock_crew_sequence):
        thin_gap = make_result_with_candidates([("2512", 0.85), ("2511", 0.847)])
        mock_hier_store.search.return_value = thin_gap
        # 0.85 < _HIGH_CONFIDENCE_THRESHOLD -> normal reranking runs once.
        mock_crew_sequence.append('{"selected_code": "2512", "reasoning": "top pick"}')

        clf = clf_factory(use_gap_aware_confidence=True)
        trace: dict = {}
        result = clf.classify("ambiguous title", trace=trace)

        assert result.primary.code == "2512"  # prediction itself is unchanged
        assert result.hitl_required is True   # but now correctly flagged for review
        assert trace["top_candidate_gap"] == pytest.approx(0.003, abs=1e-6)
        assert trace["gap_ambiguous"] is True

    def test_wide_gap_high_confidence_stays_not_hitl(self, clf_factory, mock_hier_store, mock_crew_sequence):
        wide_gap = make_result_with_candidates([("2512", 0.85), ("2511", 0.70)])
        mock_hier_store.search.return_value = wide_gap
        mock_crew_sequence.append('{"selected_code": "2512", "reasoning": "top pick"}')

        clf = clf_factory(use_gap_aware_confidence=True)
        result = clf.classify("clear title")

        assert result.hitl_required is False  # gap 0.15 is well above the threshold

    def test_single_candidate_no_gap_never_flagged_ambiguous(self, clf_factory, mock_hier_store, mock_crew_sequence):
        # Only one candidate at all -- _top_candidate_gap returns None, so
        # gap-awareness must never fabricate an ambiguity signal from nothing.
        single = make_result("2512", 0.85)
        mock_hier_store.search.return_value = single
        mock_crew_sequence.append('{"selected_code": "2512", "reasoning": "top pick"}')

        clf = clf_factory(use_gap_aware_confidence=True)
        trace: dict = {}
        result = clf.classify("some title", trace=trace)

        assert trace["top_candidate_gap"] is None
        assert trace["gap_ambiguous"] is False
        assert result.hitl_required is False


class TestAmbiguityTriggersCorrectiveRetry:
    def test_thin_gap_alone_triggers_retry_even_with_high_confidence(self, clf_factory, mock_hier_store, mock_crew_sequence):
        # This is the whole point of the recalibration: a thin gap must be
        # able to trigger a corrective retry EVEN when raw confidence is
        # comfortably above HITL_THRESHOLD (0.85 here) and
        # use_gap_aware_confidence is off -- corrective retry always
        # consults the gap signal once enabled, per its own docstring.
        thin_gap = make_result_with_candidates([("2512", 0.85), ("2511", 0.847)])
        stronger = make_result_with_candidates([("2513", 0.90), ("2511", 0.60)])
        mock_hier_store.search.side_effect = [thin_gap, stronger]

        # 0.85 < _HIGH_CONFIDENCE_THRESHOLD (0.92), so normal reranking runs
        # first -- three Crew calls total: initial rerank, reformulation,
        # retry rerank.
        mock_crew_sequence.append('{"selected_code": "2512", "reasoning": "top pick, thin gap"}')
        mock_crew_sequence.append('{"reformulated_query": "better phrase"}')
        mock_crew_sequence.append('{"selected_code": "2513", "reasoning": "clear after reformulation"}')

        clf = clf_factory(enable_corrective_retry=True)  # gap-aware-confidence NOT set
        trace: dict = {}
        result = clf.classify("ambiguous but high-confidence title", trace=trace)

        assert mock_hier_store.search.call_count == 2  # retry actually fired
        assert result.primary.code == "2513"
        assert "corrective" in result.method
        assert trace["corrective_retry_used"] is True


class TestGapBasedAcceptanceRule:
    """
    Regression tests for the 2026-08-23 fix: the retry accept/reject
    decision now prefers the candidate-GAP comparison over raw confidence,
    because real evidence (a 63-case run) showed raw confidence does not
    track correctness while the gap does. Before this fix, a retry with
    higher raw confidence but a WORSE (thinner) gap would have been wrongly
    accepted -- these tests prove that no longer happens.
    """

    def test_higher_confidence_but_worse_gap_is_rejected(self, clf_factory, mock_hier_store, mock_crew_sequence):
        # Original: confidence 0.80, wide gap (0.15) -- genuinely trustworthy.
        # Retry: confidence 0.83 (numerically higher!) but a razor-thin gap
        # (0.002) -- the OLD rule (confidence-only) would have accepted this
        # and thrown away the more trustworthy original. The NEW rule must not.
        original = make_result_with_candidates([("2512", 0.80), ("2511", 0.65)])  # gap 0.15
        retry_weaker_gap = make_result_with_candidates([("9999", 0.83), ("8888", 0.828)])  # gap 0.002

        # Force the retry path directly via _maybe_corrective_retry to isolate
        # the acceptance rule itself from the hitl/ambiguity trigger logic.
        mock_hier_store.search.return_value = original
        clf = clf_factory(enable_corrective_retry=True)

        from backend.agents.isco_classifier import ISCOMatch, _top_candidate_gap
        current_match = ISCOMatch(code="2512", title_en="Software Developers", title_ar="", confidence=0.80)
        current_gap = _top_candidate_gap(original.top_candidates)  # 0.15

        mock_hier_store.search.return_value = retry_weaker_gap
        mock_crew_sequence.append('{"reformulated_query": "alt phrase"}')
        mock_crew_sequence.append('{"selected_code": "9999", "reasoning": "picked despite thin gap"}')

        result = clf._maybe_corrective_retry(
            job_title="ambiguous title", context="", lang="en",
            current_match=current_match, top_k=5, current_gap=current_gap,
        )

        # Rejected: retry's gap (0.002) is worse than the original's (0.15),
        # even though its raw confidence (0.83) is numerically higher.
        assert result is None

    def test_lower_confidence_but_better_gap_is_accepted(self, clf_factory, mock_hier_store, mock_crew_sequence):
        # Mirror case: retry has LOWER raw confidence than the original, but
        # a much wider (more trustworthy) gap. The fixed rule should accept
        # it -- the old confidence-only rule would have rejected it.
        original = make_result_with_candidates([("2512", 0.85), ("2511", 0.845)])  # gap 0.005 (thin)

        clf = clf_factory(enable_corrective_retry=True)
        from backend.agents.isco_classifier import ISCOMatch, _top_candidate_gap
        current_match = ISCOMatch(code="2512", title_en="Software Developers", title_ar="", confidence=0.85)
        current_gap = _top_candidate_gap(original.top_candidates)  # 0.005

        retry_better_gap = make_result_with_candidates([("9999", 0.70), ("8888", 0.40)])  # gap 0.30, lower raw score
        mock_hier_store.search.return_value = retry_better_gap
        mock_crew_sequence.append('{"reformulated_query": "alt phrase"}')
        mock_crew_sequence.append('{"selected_code": "9999", "reasoning": "clearly separated despite lower score"}')

        result = clf._maybe_corrective_retry(
            job_title="ambiguous title", context="", lang="en",
            current_match=current_match, top_k=5, current_gap=current_gap,
        )

        assert result is not None
        retry_match, _ = result
        assert retry_match.code == "9999"  # accepted despite lower raw confidence

    def test_no_gap_available_falls_back_to_confidence_rule(self, clf_factory, mock_hier_store, mock_crew_sequence):
        # Single-candidate results on both sides -- no gap computable for
        # either -- must fall back to the original confidence comparison,
        # not silently reject or crash.
        clf = clf_factory(enable_corrective_retry=True)
        from backend.agents.isco_classifier import ISCOMatch
        current_match = ISCOMatch(code="9999", title_en="Vague", title_ar="", confidence=0.40)

        retry_single = make_result("2512", 0.80)  # only 1 candidate -> no gap
        mock_hier_store.search.return_value = retry_single
        mock_crew_sequence.append('{"reformulated_query": "alt phrase"}')
        mock_crew_sequence.append('{"selected_code": "2512", "reasoning": "clear win"}')

        result = clf._maybe_corrective_retry(
            job_title="ambiguous title", context="", lang="en",
            current_match=current_match, top_k=5, current_gap=None,
        )

        assert result is not None
        retry_match, _ = result
        assert retry_match.confidence > current_match.confidence  # fallback rule applied correctly
