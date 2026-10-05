"""
Tests for backend/agents/query_planner.py (Item 2 of the 2026-09-12
multi-agent RAG work: multi-step agentic retrieval).

Follows test_isco_classifier.py's own documented convention: patch
Agent/Task/Crew at the module level to bypass CrewAI's Pydantic LLM
validation entirely.
"""

from unittest.mock import MagicMock

import pytest

from backend.agents.query_planner import QueryPlanner


@pytest.fixture
def planner(monkeypatch):
    monkeypatch.setattr("backend.agents.query_planner.get_llm", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("backend.agents.query_planner.Agent", MagicMock())
    return QueryPlanner()


def _set_kickoff_response(monkeypatch, response):
    crew_instance = MagicMock()
    if isinstance(response, Exception):
        crew_instance.kickoff.side_effect = response
    else:
        crew_instance.kickoff.return_value = response
    monkeypatch.setattr("backend.agents.query_planner.Crew", MagicMock(return_value=crew_instance))
    monkeypatch.setattr("backend.agents.query_planner.Task", MagicMock())


class TestDecompose:
    @pytest.mark.parametrize("response,received", [("one phrase", True), (RuntimeError("timeout"), False)])
    def test_usage_observer_receives_success_and_failed_calls(self, planner, monkeypatch, response, received):
        _set_kickoff_response(monkeypatch, response)
        observer = MagicMock()
        planner.decompose("original", "occupation", usage_observer=observer)
        observer.assert_called_once()
        crew, response_received, error = observer.call_args.args
        assert crew is not None
        assert response_received is received
        assert error == (None if received else "RuntimeError: timeout")

    def test_usage_observer_failure_does_not_change_decomposition(self, planner, monkeypatch):
        _set_kickoff_response(monkeypatch, "phrase one\nphrase two")
        result = planner.decompose("original", "occupation", usage_observer=MagicMock(side_effect=RuntimeError("bad telemetry")))
        assert result == ["phrase one", "phrase two"]

    def test_parses_multiple_lines(self, planner, monkeypatch):
        _set_kickoff_response(monkeypatch, "subsistence farmer\ntaxi driver")
        result = planner.decompose("farmer who also drives a taxi part-time", "occupation")
        assert result == ["subsistence farmer", "taxi driver"]

    def test_strips_numbering_and_bullets(self, planner, monkeypatch):
        _set_kickoff_response(monkeypatch, "1. subsistence farmer\n- taxi driver\n* street vendor")
        result = planner.decompose("multi job description", "occupation", max_subqueries=3)
        assert result == ["subsistence farmer", "taxi driver", "street vendor"]

    def test_caps_at_max_subqueries(self, planner, monkeypatch):
        _set_kickoff_response(monkeypatch, "a\nb\nc\nd\ne")
        result = planner.decompose("text", "occupation", max_subqueries=2)
        assert result == ["a", "b"]

    def test_empty_response_falls_back_to_original_text(self, planner, monkeypatch):
        _set_kickoff_response(monkeypatch, "")
        result = planner.decompose("original text", "occupation")
        assert result == ["original text"]

    def test_kickoff_failure_falls_back_to_original_text(self, planner, monkeypatch):
        _set_kickoff_response(monkeypatch, RuntimeError("boom"))
        result = planner.decompose("original text", "occupation")
        assert result == ["original text"]

    def test_blank_text_returns_blank(self, planner):
        assert planner.decompose("   ", "occupation") == ["   ".strip()]

    def test_single_line_response_is_single_item_list(self, planner, monkeypatch):
        _set_kickoff_response(monkeypatch, "software developer")
        result = planner.decompose("computer stuff", "occupation")
        assert result == ["software developer"]


class TestReconcile:
    def test_single_candidate_returned_as_is(self):
        assert QueryPlanner.reconcile([("2512", 0.9)]) == ("2512", 0.9)

    def test_no_repeats_highest_score_wins(self):
        result = QueryPlanner.reconcile([("2512", 0.9), ("6111", 0.4)])
        assert result == ("2512", 0.9)

    def test_repeated_candidate_beats_single_higher_outlier(self):
        # 6111 appears twice (as top1 for two different sub-queries) with
        # lower individual scores; 2512 appears once with a higher score.
        # The frequency tiebreak means 6111 wins.
        result = QueryPlanner.reconcile([("2512", 0.95), ("6111", 0.60), ("6111", 0.55)])
        assert result == ("6111", 0.60)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            QueryPlanner.reconcile([])
