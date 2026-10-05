"""
Tests for backend/agents/hierarchical_classification_crew.py (Item 3 of
the 2026-09-12 multi-agent RAG work: real CrewAI hierarchical delegation).

A real manager-LLM delegation decision can't be pinned to one fixed
kickoff.return_value string the way a sequential single-task crew can (the
whole point is the manager decides at runtime) -- so these tests assert
what CAN be honestly asserted hermetically: the Crew is actually
constructed with process=Process.hierarchical and a real manager_llm, the
worker Agents are allow_delegation=False, and the fallback path (crew
failure) really does call all three underlying classifiers directly. Real
delegation behaviour against a live LLM is a separate, documented manual
smoke check (see the module's own docstring / CLAUDE.md), not something a
hermetic unit test can honestly claim to cover.
"""

from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest

from backend.agents.hierarchical_classification_crew import (
    CoordinatedResult,
    HierarchicalClassificationCoordinator,
)


@pytest.fixture
def mock_classifiers():
    isco = MagicMock()
    isco.classify.return_value = MagicMock(primary=MagicMock(code="2512", title_en="Software Developers"))
    isic = MagicMock()
    isic.classify.return_value = MagicMock(section="J", class_code="6201", class_title="Computer programming")
    isced = MagicMock()
    isced.classify.return_value = MagicMock(level=6, level_title="Bachelor's")
    return isco, isic, isced


@pytest.fixture
def coordinator(mock_classifiers, monkeypatch):
    monkeypatch.setattr(
        "backend.agents.hierarchical_classification_crew.get_llm", lambda *a, **kw: MagicMock()
    )
    isco, isic, isced = mock_classifiers
    return HierarchicalClassificationCoordinator(isco, isic, isced)


class TestCrewConfiguration:
    """Config-shape assertions -- the honest thing to test for a
    real delegation decision, per this file's own docstring."""

    def test_crew_uses_hierarchical_process_with_manager_llm(self, coordinator, monkeypatch):
        captured = {}

        class _FakeCrew:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def kickoff(self):
                captured["agents"][0].tools[0]("software developer")
                return '{"isco_code": "2512", "isic_section": "J", "isced_level": 6}'

        monkeypatch.setattr("crewai.Crew", _FakeCrew)
        monkeypatch.setattr("crewai.Agent", lambda **kwargs: SimpleNamespace(**kwargs))
        monkeypatch.setattr("crewai.Task", lambda **kwargs: SimpleNamespace(**kwargs))
        monkeypatch.setattr("crewai.tools.tool", lambda *a, **kw: (lambda fn: fn))

        result = coordinator.classify_all(job_title="software developer")

        assert captured.get("process") is not None
        from crewai import Process
        assert captured["process"] == Process.hierarchical
        assert captured.get("manager_llm") is not None
        assert not hasattr(captured["tasks"][0], "agent") or captured["tasks"][0].agent is None
        assert result.fallback_used is False

    def test_worker_agents_never_allow_delegation(self, coordinator, monkeypatch):
        agent_calls = []

        class _FakeAgent:
            def __init__(self, **kwargs):
                agent_calls.append(kwargs)
                self.__dict__.update(kwargs)

        class _FakeCrew:
            def __init__(self, **kwargs):
                self.agents = kwargs["agents"]

            def kickoff(self):
                self.agents[0].tools[0]("software developer")
                return '{"isco_code": "2512", "isic_section": null, "isced_level": null}'

        monkeypatch.setattr("crewai.Crew", _FakeCrew)
        monkeypatch.setattr("crewai.Agent", _FakeAgent)
        monkeypatch.setattr("crewai.Task", MagicMock())
        monkeypatch.setattr("crewai.tools.tool", lambda *a, **kw: (lambda fn: fn))

        coordinator.classify_all(job_title="software developer")

        worker_calls = [kwargs for kwargs in agent_calls if kwargs["role"] != "Classification Manager"]
        assert len(worker_calls) == 3
        for kwargs in worker_calls:
            assert kwargs.get("allow_delegation") is False
            assert kwargs["max_iter"] == 4
        manager_call = next(kwargs for kwargs in agent_calls if kwargs["role"] == "Classification Manager")
        assert manager_call["allow_delegation"] is True
        assert manager_call["max_iter"] == 8


class TestFallbackPath:
    def test_crew_failure_falls_back_to_sequential_direct_calls(
        self, coordinator, mock_classifiers, monkeypatch
    ):
        isco, isic, isced = mock_classifiers

        def _boom(**kwargs):
            raise RuntimeError("manager LLM unreachable")

        monkeypatch.setattr("crewai.Crew", _boom)
        monkeypatch.setattr("crewai.Agent", MagicMock())
        monkeypatch.setattr("crewai.Task", MagicMock())
        monkeypatch.setattr("crewai.tools.tool", lambda *a, **kw: (lambda fn: fn))

        result = coordinator.classify_all(
            job_title="software developer", industry_text="tech company", education_text="bachelor's degree",
        )

        assert result.fallback_used is True
        assert "fallback" in result.fallback_reason
        assert result.isco_code == "2512"
        assert result.isic_section == "J"
        assert result.isced_level == 6
        isco.classify.assert_called_once_with("software developer")
        isic.classify.assert_called_once_with("tech company")
        isced.classify.assert_called_once_with("bachelor's degree")

    def test_fallback_skips_empty_dimensions(self, coordinator, mock_classifiers, monkeypatch):
        isco, isic, isced = mock_classifiers

        def _boom(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr("crewai.Crew", _boom)
        monkeypatch.setattr("crewai.Agent", MagicMock())
        monkeypatch.setattr("crewai.Task", MagicMock())
        monkeypatch.setattr("crewai.tools.tool", lambda *a, **kw: (lambda fn: fn))

        result = coordinator.classify_all(job_title="software developer")

        assert result.isco_code == "2512"
        assert result.isic_section is None
        assert result.isced_level is None
        isic.classify.assert_not_called()
        isced.classify.assert_not_called()

    def test_malformed_crew_output_falls_back(self, coordinator, mock_classifiers, monkeypatch):
        isco, isic, isced = mock_classifiers

        class _FakeCrew:
            def __init__(self, **kwargs):
                pass

            def kickoff(self):
                return "not valid json at all"

        monkeypatch.setattr("crewai.Crew", _FakeCrew)
        monkeypatch.setattr("crewai.Agent", MagicMock())
        monkeypatch.setattr("crewai.Task", MagicMock())
        monkeypatch.setattr("crewai.tools.tool", lambda *a, **kw: (lambda fn: fn))

        result = coordinator.classify_all(job_title="software developer")

        assert result.isco_code == "2512"
        assert result.fallback_used is True
        isco.classify.assert_called_once_with("software developer")


class TestParseCrewResult:
    def test_parses_clean_json(self):
        result = HierarchicalClassificationCoordinator._parse_crew_result(
            '{"isco_code": "2512", "isic_section": "J", "isced_level": 6}'
        )
        assert result == CoordinatedResult(isco_code="2512", isic_section="J", isced_level=6, fallback_used=False)

    def test_parses_json_with_markdown_fences(self):
        result = HierarchicalClassificationCoordinator._parse_crew_result(
            '```json\n{"isco_code": "2512", "isic_section": null, "isced_level": null}\n```'
        )
        assert result.isco_code == "2512"
        assert result.isic_section is None
        assert result.isced_level is None

    def test_null_string_treated_as_none(self):
        result = HierarchicalClassificationCoordinator._parse_crew_result(
            '{"isco_code": "null", "isic_section": "J", "isced_level": null}'
        )
        assert result.isco_code is None
        assert result.isic_section == "J"

    def test_malformed_value_rejected_not_passed_through(self):
        # Real, live-caught case (2026-09-12): a local-model manager LLM
        # produced exactly this malformed shape in a manual smoke check --
        # valid JSON, but not a real ISCO code or ISIC section. Must be
        # treated as "no answer," never silently passed through as if it
        # were a real result.
        result = HierarchicalClassificationCoordinator._parse_crew_result(
            '{"isco_code": "5310, null", "isic_section": "11, null", "isced_level": null}'
        )
        assert result.isco_code is None
        assert result.isic_section is None
        assert result.isced_level is None
        assert result.fallback_used is False  # parsed fine, just no valid values -- not a crew failure

    def test_isced_level_out_of_range_rejected(self):
        result = HierarchicalClassificationCoordinator._parse_crew_result(
            '{"isco_code": "2512", "isic_section": "J", "isced_level": 15}'
        )
        assert result.isced_level is None

    def test_lowercase_isic_section_normalised(self):
        result = HierarchicalClassificationCoordinator._parse_crew_result(
            '{"isco_code": "2512", "isic_section": "j", "isced_level": 6}'
        )
        assert result.isic_section == "J"

    @pytest.mark.parametrize("value", [6.9, True, float("nan"), float("inf")])
    def test_fractional_boolean_and_nonfinite_levels_rejected(self, value):
        from backend.agents.hierarchical_classification_crew import _valid_isced_level
        assert _valid_isced_level(value) is None

    @pytest.mark.parametrize("raw", ["not JSON", "[]", "null", '"2512"', "{broken}"])
    def test_invalid_json_or_nonobject_rejected(self, raw):
        with pytest.raises(ValueError):
            HierarchicalClassificationCoordinator._parse_crew_result(raw)


def _tool_framework(monkeypatch, kickoff):
    monkeypatch.setattr("crewai.Agent", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr("crewai.Task", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr("crewai.tools.tool", lambda *args, **kwargs: (lambda fn: fn))

    class FakeCrew:
        def __init__(self, **kwargs):
            self.workers = kwargs["agents"]

        def kickoff(self):
            return kickoff(self.workers)

    monkeypatch.setattr("crewai.Crew", FakeCrew)


def test_valid_shaped_code_without_tool_evidence_falls_back(coordinator, mock_classifiers, monkeypatch):
    _tool_framework(monkeypatch, lambda workers: '{"isco_code": "9999"}')
    result = coordinator.classify_all(job_title="Developer")
    assert result.isco_code == "2512" and result.fallback_used is True
    mock_classifiers[0].classify.assert_called_once_with("Developer")


def test_manager_cannot_replace_actual_tool_code(coordinator, mock_classifiers, monkeypatch):
    def kickoff(workers):
        workers[0].tools[0]("Developer")
        return '{"isco_code": "9999"}'
    _tool_framework(monkeypatch, kickoff)
    result = coordinator.classify_all(job_title="Developer")
    assert result.isco_code == "2512" and result.fallback_used is True
    mock_classifiers[0].classify.assert_called_once_with("Developer")


def test_partial_delegation_falls_back_only_unfinished_dimensions(coordinator, mock_classifiers, monkeypatch):
    def kickoff(workers):
        workers[0].tools[0]("Developer")
        return '{"isco_code": "2512", "isic_section": null, "isced_level": null}'
    _tool_framework(monkeypatch, kickoff)
    result = coordinator.classify_all("Developer", "IT company", "Bachelor")
    assert (result.isco_code, result.isic_section, result.isced_level) == ("2512", "J", 6)
    assert result.fallback_used is True
    for classifier in mock_classifiers:
        classifier.classify.assert_called_once()


def test_worker_cannot_substitute_survey_text(coordinator, mock_classifiers, monkeypatch):
    _tool_framework(monkeypatch, lambda workers: workers[0].tools[0]("CEO"))
    result = coordinator.classify_all(job_title="Developer")
    assert result.isco_code == "2512" and result.fallback_used is True
    mock_classifiers[0].classify.assert_called_once_with("Developer")


def test_accepted_manager_output_requires_all_tool_evidence(coordinator, mock_classifiers, monkeypatch):
    def kickoff(workers):
        workers[0].tools[0]("Developer")
        workers[1].tools[0]("IT company")
        workers[2].tools[0]("Bachelor")
        return '{"isco_code": "2512", "isic_section": "J", "isced_level": 6}'
    _tool_framework(monkeypatch, kickoff)
    result = coordinator.classify_all("Developer", "IT company", "Bachelor")
    assert result.fallback_used is False
    assert (result.isco_code, result.isic_section, result.isced_level) == ("2512", "J", 6)


def test_fallback_classifier_failure_retains_other_dimensions(coordinator, mock_classifiers, monkeypatch):
    _tool_framework(monkeypatch, lambda workers: "bad JSON")
    mock_classifiers[0].classify.side_effect = RuntimeError("store unavailable")
    result = coordinator.classify_all("Developer", "IT company", "Bachelor")
    assert result.isco_code is None
    assert (result.isic_section, result.isced_level) == ("J", 6)
    assert result.fallback_used is True
    assert "RuntimeError" in result.fallback_reason


def test_empty_inputs_do_not_construct_a_crew(coordinator, monkeypatch):
    monkeypatch.setattr("crewai.Crew", MagicMock(side_effect=AssertionError("unexpected Crew")))
    assert coordinator.classify_all() == CoordinatedResult()
