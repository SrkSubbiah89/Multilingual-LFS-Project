"""
Tests for backend/agents/isco_reranker_strict.py -- the future B3-Reliability
reranker. Not used by B0/B1/B2; these tests validate it in isolation using
the same Agent/Crew mocking pattern as test_isco_classifier.py.
"""

from unittest.mock import MagicMock

import pytest

from backend.agents.isco_reranker_strict import StrictCandidate, StrictReranker


def make_candidates():
    return [
        StrictCandidate(code="2512", label_en="Software Developers", score=0.80),
        StrictCandidate(code="2511", label_en="Systems Analysts", score=0.72),
    ]


@pytest.fixture
def mock_crew(monkeypatch):
    crew_instance = MagicMock()
    crew_class = MagicMock(return_value=crew_instance)
    monkeypatch.setattr("backend.agents.isco_reranker_strict.Agent", MagicMock())
    monkeypatch.setattr("backend.agents.isco_reranker_strict.Crew", crew_class)
    monkeypatch.setattr("backend.agents.isco_reranker_strict.Task", MagicMock())
    return crew_instance


def make_reranker(max_retries=1):
    return StrictReranker(agent=MagicMock(), max_retries=max_retries)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_select_valid_output_first_attempt(mock_crew):
    mock_crew.kickoff.return_value = (
        '{"selected_isco_code": "2512", "confidence_or_score": 0.9, "reason": "Best match."}'
    )
    reranker = make_reranker()
    decision = reranker.select("software developer", make_candidates(), lang="en")

    assert decision.abstained is False
    assert decision.selected_code == "2512"
    assert decision.confidence == pytest.approx(0.9)
    assert decision.reason == "Best match."
    assert decision.attempts == 1
    assert decision.invalid_output is False


def test_select_strips_markdown_fences():
    pass  # covered implicitly by _parse tests below; kept as a marker for readers


def test_parse_handles_markdown_fences():
    reranker = make_reranker()
    code_map = {"2512": StrictCandidate(code="2512", label_en="x")}
    raw = '```json\n{"selected_isco_code": "2512", "confidence_or_score": 0.5, "reason": "ok"}\n```'
    code, conf, reason, ok = StrictReranker._parse(raw, code_map)
    assert ok is True
    assert code == "2512"


# ---------------------------------------------------------------------------
# Invalid-code rejection
# ---------------------------------------------------------------------------

def test_select_unrecognised_code_is_invalid_not_accepted(mock_crew):
    mock_crew.kickoff.return_value = (
        '{"selected_isco_code": "9999", "confidence_or_score": 0.9, "reason": "not a real candidate"}'
    )
    reranker = make_reranker(max_retries=0)  # single attempt only, for a clean assertion
    decision = reranker.select("software developer", make_candidates(), lang="en")

    assert decision.selected_code is None
    assert decision.abstained is True
    assert decision.invalid_output is True


def test_select_malformed_json_is_invalid(mock_crew):
    mock_crew.kickoff.return_value = "this is not json at all"
    reranker = make_reranker(max_retries=0)
    decision = reranker.select("software developer", make_candidates(), lang="en")

    assert decision.selected_code is None
    assert decision.abstained is True
    assert decision.invalid_output is True


# ---------------------------------------------------------------------------
# One-retry-only behaviour
# ---------------------------------------------------------------------------

def test_select_retries_exactly_once_on_invalid_output(mock_crew):
    mock_crew.kickoff.return_value = "not json"
    reranker = make_reranker(max_retries=1)
    decision = reranker.select("software developer", make_candidates(), lang="en")

    assert mock_crew.kickoff.call_count == 2  # initial attempt + exactly one retry
    assert decision.attempts == 2
    assert decision.abstained is True


def test_select_succeeds_on_retry_after_initial_failure(mock_crew):
    mock_crew.kickoff.side_effect = [
        "garbage, not json",
        '{"selected_isco_code": "2511", "confidence_or_score": 0.6, "reason": "second try works"}',
    ]
    reranker = make_reranker(max_retries=1)
    decision = reranker.select("software developer", make_candidates(), lang="en")

    assert mock_crew.kickoff.call_count == 2
    assert decision.abstained is False
    assert decision.selected_code == "2511"
    assert decision.attempts == 2


def test_select_does_not_retry_more_than_configured(mock_crew):
    mock_crew.kickoff.return_value = "not json"
    reranker = make_reranker(max_retries=3)
    decision = reranker.select("software developer", make_candidates(), lang="en")

    assert mock_crew.kickoff.call_count == 4  # initial + 3 retries, exactly
    assert decision.abstained is True


# ---------------------------------------------------------------------------
# No silent fallback after invalid output
# ---------------------------------------------------------------------------

def test_select_never_defaults_to_top_candidate_on_failure(mock_crew):
    """The critical B0/B1-vs-B3 behavioural difference: B0/B1's
    _parse_llm_response() falls back to candidates[0] on failure. This
    reranker must NEVER do that -- selected_code must be None, not the
    top candidate's code, when every attempt fails."""
    mock_crew.kickoff.return_value = "not json"
    reranker = make_reranker(max_retries=1)
    candidates = make_candidates()
    decision = reranker.select("software developer", candidates, lang="en")

    assert decision.selected_code is None
    assert decision.selected_code != candidates[0].code
    assert decision.abstained is True
    assert decision.abstain_reason  # non-empty, explains why


def test_select_empty_candidate_list_abstains_without_calling_llm(mock_crew):
    reranker = make_reranker()
    decision = reranker.select("software developer", [], lang="en")

    assert decision.abstained is True
    assert decision.selected_code is None
    mock_crew.kickoff.assert_not_called()


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------

def test_select_exception_during_call_is_recorded_and_retried(mock_crew):
    mock_crew.kickoff.side_effect = TimeoutError("Connection timed out after 120.0 seconds")
    reranker = make_reranker(max_retries=1)
    decision = reranker.select("software developer", make_candidates(), lang="en")

    assert mock_crew.kickoff.call_count == 2
    assert decision.abstained is True
    assert decision.timed_out is True
    assert decision.selected_code is None


def test_select_confidence_missing_defaults_to_none(mock_crew):
    mock_crew.kickoff.return_value = '{"selected_isco_code": "2512", "reason": "no confidence field"}'
    reranker = make_reranker()
    decision = reranker.select("software developer", make_candidates(), lang="en")

    assert decision.selected_code == "2512"
    assert decision.confidence is None
