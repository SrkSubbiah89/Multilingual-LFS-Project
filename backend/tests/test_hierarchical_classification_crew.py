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
                return '{"isco_code": "2512", "isic_section": "J", "isced_level": 6}'

        monkeypatch.setattr("crewai.Crew", _FakeCrew)
        monkeypatch.setattr("crewai.Agent", MagicMock())
        monkeypatch.setattr("crewai.Task", MagicMock())
        monkeypatch.setattr("crewai.tools.tool", lambda *a, **kw: (lambda fn: fn))

        result = coordinator.classify_all(job_title="software developer")

        assert captured.get("process") is not None
        from crewai import Process
        assert captured["process"] == Process.hierarchical
        assert captured.get("manager_llm") is not None
        assert result.fallback_used is False

    def test_worker_agents_never_allow_delegation(self, coordinator, monkeypatch):
        agent_calls = []

        class _FakeAgent:
            def __init__(self, **kwargs):
                agent_calls.append(kwargs)

        class _FakeCrew:
            def __init__(self, **kwargs):
                pass

            def kickoff(self):
                return '{"isco_code": "2512", "isic_section": null, "isced_level": null}'

        monkeypatch.setattr("crewai.Crew", _FakeCrew)
        monkeypatch.setattr("crewai.Agent", _FakeAgent)
        monkeypatch.setattr("crewai.Task", MagicMock())
        monkeypatch.setattr("crewai.tools.tool", lambda *a, **kw: (lambda fn: fn))

        coordinator.classify_all(job_title="software developer")

        assert len(agent_calls) == 3  # occupation, industry, education
        for kwargs in agent_calls:
            assert kwargs.get("allow_delegation") is False


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

        # _parse_crew_result degrades to all-None fields on unparsable
        # output -- NOT a raised exception, but also not a real result;
        # the coordinator's own contract only guarantees no crash, not
        # that malformed LLM output magically becomes a fallback trigger.
        assert result.isco_code is None
        assert result.fallback_used is False


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
