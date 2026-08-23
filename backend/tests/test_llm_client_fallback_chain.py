"""
Tests for get_llm(TaskType.GENERAL)'s fallback chain (2026-08-24):
Ollama -> Claude -> Gemini -> Groq -> OpenRouter, stopping at the first
provider that's actually configured and reachable.

get_llm_strict() itself is untouched by this feature -- confirmed
indirectly here (each cloud hop delegates to it) and directly in
test_llm_client_get_llm_strict.py. This file only tests the NEW chaining
behaviour in get_llm().

Fully offline -- no live Ollama/network/API calls. Ollama reachability
and each provider's env var / API call are mocked at the same boundaries
test_llm_client_get_llm_strict.py already uses.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.llm.llm_client import MODEL_CRITICAL, MODEL_GENERAL, TaskType, get_llm


def _clear_all_provider_keys(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _isolate_from_real_dotenv(monkeypatch):
    """The real .env sets LLM_FALLBACK_EXCLUDE=anthropic (this project's
    Anthropic account has zero credit) via load_dotenv() at module import
    time. Every test in this file exercises the chain logic itself, so it
    must not silently inherit that -- clear it here and let individual
    tests opt back in explicitly (see TestFallbackExclude)."""
    monkeypatch.delenv("LLM_FALLBACK_EXCLUDE", raising=False)


class TestOllamaFirst:
    def test_ollama_reachable_uses_ollama_no_fallback_attempted(self, monkeypatch):
        monkeypatch.setattr("backend.llm.llm_client._ollama_is_running", lambda: True)
        get_llm_strict = MagicMock()
        monkeypatch.setattr("backend.llm.llm_client.get_llm_strict", get_llm_strict)

        trace = {}
        llm = get_llm(TaskType.GENERAL, trace=trace)

        assert llm.model == MODEL_GENERAL
        get_llm_strict.assert_not_called()
        assert trace["resolved_provider"] == "ollama"
        assert trace["attempted_providers"] == ["ollama"]
        assert trace["failure_reasons"] == {}


class TestFallsThroughChainInOrder:
    def test_ollama_down_falls_back_to_anthropic(self, monkeypatch):
        monkeypatch.setattr("backend.llm.llm_client._ollama_is_running", lambda: False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        fake_llm = MagicMock(model=MODEL_CRITICAL)
        get_llm_strict = MagicMock(return_value=fake_llm)
        monkeypatch.setattr("backend.llm.llm_client.get_llm_strict", get_llm_strict)

        trace = {}
        llm = get_llm(TaskType.GENERAL, trace=trace)

        assert llm is fake_llm
        get_llm_strict.assert_called_once_with(MODEL_CRITICAL, temperature=0.3)
        assert trace["resolved_provider"] == "anthropic"
        assert trace["attempted_providers"] == ["ollama", "anthropic"]
        assert "ollama" in trace["failure_reasons"]

    def test_ollama_and_anthropic_unavailable_falls_back_to_gemini(self, monkeypatch):
        monkeypatch.setattr("backend.llm.llm_client._ollama_is_running", lambda: False)
        fake_llm = MagicMock(model="gemini/gemini-3.6-flash")

        def strict_side_effect(model, temperature):
            if model == MODEL_CRITICAL:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            return fake_llm

        get_llm_strict = MagicMock(side_effect=strict_side_effect)
        monkeypatch.setattr("backend.llm.llm_client.get_llm_strict", get_llm_strict)

        trace = {}
        llm = get_llm(TaskType.GENERAL, trace=trace)

        assert llm is fake_llm
        assert trace["resolved_provider"] == "gemini"
        assert trace["attempted_providers"] == ["ollama", "anthropic", "gemini"]
        assert set(trace["failure_reasons"]) == {"ollama", "anthropic"}

    def test_falls_all_the_way_to_openrouter(self, monkeypatch):
        monkeypatch.setattr("backend.llm.llm_client._ollama_is_running", lambda: False)
        fake_llm = MagicMock(model="openrouter/nvidia/nemotron-3-nano-30b-a3b:free")

        def strict_side_effect(model, temperature):
            if "openrouter" in model:
                return fake_llm
            raise RuntimeError(f"{model} unavailable")

        get_llm_strict = MagicMock(side_effect=strict_side_effect)
        monkeypatch.setattr("backend.llm.llm_client.get_llm_strict", get_llm_strict)

        trace = {}
        llm = get_llm(TaskType.GENERAL, trace=trace)

        assert llm is fake_llm
        assert trace["resolved_provider"] == "openrouter"
        assert trace["attempted_providers"] == ["ollama", "anthropic", "gemini", "groq", "openrouter"]
        assert set(trace["failure_reasons"]) == {"ollama", "anthropic", "gemini", "groq"}

    def test_every_provider_unavailable_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr("backend.llm.llm_client._ollama_is_running", lambda: False)
        get_llm_strict = MagicMock(side_effect=RuntimeError("not configured"))
        monkeypatch.setattr("backend.llm.llm_client.get_llm_strict", get_llm_strict)

        trace = {}
        with pytest.raises(RuntimeError, match="every provider in the fallback chain"):
            get_llm(TaskType.GENERAL, trace=trace)

        assert trace["attempted_providers"] == ["ollama", "anthropic", "gemini", "groq", "openrouter"]
        assert len(trace["failure_reasons"]) == 5


class TestFallbackExclude:
    def test_excluded_provider_is_skipped_without_being_attempted(self, monkeypatch):
        monkeypatch.setattr("backend.llm.llm_client._ollama_is_running", lambda: False)
        monkeypatch.setenv("LLM_FALLBACK_EXCLUDE", "anthropic")
        fake_llm = MagicMock(model="gemini/gemini-3.6-flash")

        def strict_side_effect(model, temperature):
            if model == MODEL_CRITICAL:
                raise AssertionError("anthropic must never be reached when excluded")
            return fake_llm

        get_llm_strict = MagicMock(side_effect=strict_side_effect)
        monkeypatch.setattr("backend.llm.llm_client.get_llm_strict", get_llm_strict)

        trace = {}
        llm = get_llm(TaskType.GENERAL, trace=trace)

        assert llm is fake_llm
        assert trace["resolved_provider"] == "gemini"
        assert "anthropic" not in trace["attempted_providers"]
        assert trace["failure_reasons"]["anthropic"] == "excluded via LLM_FALLBACK_EXCLUDE"

    def test_exclude_list_is_comma_separated_and_case_insensitive(self, monkeypatch):
        monkeypatch.setattr("backend.llm.llm_client._ollama_is_running", lambda: False)
        monkeypatch.setenv("LLM_FALLBACK_EXCLUDE", "Anthropic, GEMINI")
        fake_llm = MagicMock(model="groq/openai/gpt-oss-120b")

        def strict_side_effect(model, temperature):
            if "groq" in model:
                return fake_llm
            raise AssertionError(f"{model} must never be reached when excluded")

        get_llm_strict = MagicMock(side_effect=strict_side_effect)
        monkeypatch.setattr("backend.llm.llm_client.get_llm_strict", get_llm_strict)

        trace = {}
        llm = get_llm(TaskType.GENERAL, trace=trace)
        assert llm is fake_llm
        assert trace["resolved_provider"] == "groq"


class TestTraceOptional:
    def test_omitting_trace_still_works(self, monkeypatch):
        monkeypatch.setattr("backend.llm.llm_client._ollama_is_running", lambda: True)
        llm = get_llm(TaskType.GENERAL)   # no trace= passed -- existing call signature
        assert llm.model == MODEL_GENERAL


class TestCriticalTaskUnaffected:
    def test_critical_still_requires_anthropic_no_fallback(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(EnvironmentError):
            get_llm(TaskType.CRITICAL)

    def test_critical_never_touches_fallback_chain(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        get_llm_strict = MagicMock()
        monkeypatch.setattr("backend.llm.llm_client.get_llm_strict", get_llm_strict)
        get_llm(TaskType.CRITICAL)
        get_llm_strict.assert_not_called()   # CRITICAL uses _get_claude_llm directly
