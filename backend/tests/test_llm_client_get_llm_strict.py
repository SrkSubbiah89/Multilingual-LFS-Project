"""
Tests for backend/llm/llm_client.py::get_llm_strict().

get_llm_strict() has no prior dedicated test coverage. It's the pinned-
model factory the eval harness (and, as of 2026-08-23, the ISCOClassifier
reranker_model= parameter) uses to guarantee a specific model answers
every call -- never a silent provider/model substitution, unlike get_llm()
(which does fall back). Covers all three recognised provider prefixes
(ollama/, anthropic/, groq/) plus the fail-closed contract: a missing key
or unreachable/unpulled model raises immediately, never substitutes.
"""

from unittest.mock import MagicMock

import pytest

from backend.llm.llm_client import get_llm_strict


class TestUnrecognisedProvider:
    def test_unknown_prefix_raises_value_error(self):
        with pytest.raises(ValueError, match="unrecognised provider prefix"):
            get_llm_strict("cohere/command-r", temperature=0.0)

    def test_bare_model_name_no_prefix_raises_value_error(self):
        with pytest.raises(ValueError, match="unrecognised provider prefix"):
            get_llm_strict("llama3.2", temperature=0.0)


class TestOllamaProvider:
    def test_ollama_unreachable_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr("backend.llm.llm_client._ollama_is_running", lambda: False)
        with pytest.raises(RuntimeError, match="not reachable"):
            get_llm_strict("ollama/llama3.2", temperature=0.0)

    def test_ollama_reachable_but_model_not_pulled_raises(self, monkeypatch):
        monkeypatch.setattr("backend.llm.llm_client._ollama_is_running", lambda: True)

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b'{"models": [{"name": "qwen2.5:3b"}]}'

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: FakeResponse())
        with pytest.raises(RuntimeError, match="not pulled"):
            get_llm_strict("ollama/llama3.2", temperature=0.0)

    def test_ollama_reachable_and_pulled_constructs_llm(self, monkeypatch):
        monkeypatch.setattr("backend.llm.llm_client._ollama_is_running", lambda: True)

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b'{"models": [{"name": "llama3.2:latest"}]}'

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: FakeResponse())
        fake_llm = MagicMock()
        monkeypatch.setattr("backend.llm.llm_client.LLM", MagicMock(return_value=fake_llm))

        result = get_llm_strict("ollama/llama3.2", temperature=0.3)
        assert result is fake_llm


class TestAnthropicProvider:
    def test_missing_api_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            get_llm_strict("anthropic/claude-3-5-sonnet-20241022", temperature=0.0)

    def test_present_api_key_constructs_llm(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-test-key")
        fake_llm = MagicMock()
        llm_class = MagicMock(return_value=fake_llm)
        monkeypatch.setattr("backend.llm.llm_client.LLM", llm_class)

        result = get_llm_strict("anthropic/claude-3-5-sonnet-20241022", temperature=0.0)
        assert result is fake_llm
        _, kwargs = llm_class.call_args
        assert kwargs["api_key"] == "sk-ant-fake-test-key"
        assert kwargs["model"] == "anthropic/claude-3-5-sonnet-20241022"


class TestGroqProvider:
    """New 2026-08-23: Groq support, added specifically because this
    project's local machine cannot reliably run larger Ollama models
    (confirmed: aya:latest, 8B, hit 120s timeouts on ~76% of reranking
    calls under real memory pressure) and Anthropic credit is unavailable.
    Groq hosts open models on its own infrastructure over a free-tier API,
    removing the local memory-pressure failure mode entirely."""

    def test_missing_api_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            get_llm_strict("groq/llama-3.3-70b-versatile", temperature=0.0)

    def test_present_api_key_constructs_llm(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_fake_test_key")
        fake_llm = MagicMock()
        llm_class = MagicMock(return_value=fake_llm)
        monkeypatch.setattr("backend.llm.llm_client.LLM", llm_class)

        result = get_llm_strict("groq/llama-3.3-70b-versatile", temperature=0.0)
        assert result is fake_llm
        _, kwargs = llm_class.call_args
        assert kwargs["api_key"] == "gsk_fake_test_key"
        assert kwargs["model"] == "groq/llama-3.3-70b-versatile"

    def test_never_silently_falls_back_to_ollama_or_anthropic(self, monkeypatch):
        # Fail-closed contract: a missing GROQ_API_KEY must raise, never
        # silently try Ollama or Anthropic instead.
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-present-but-irrelevant")
        ollama_check = MagicMock(return_value=True)
        monkeypatch.setattr("backend.llm.llm_client._ollama_is_running", ollama_check)

        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            get_llm_strict("groq/llama-3.3-70b-versatile", temperature=0.0)
        ollama_check.assert_not_called()


class TestGeminiProvider:
    """New 2026-08-23: recommended PRIMARY free-tier option for this
    project's RAG-comparison work (Google AI Studio's Gemini Flash --
    strong at document understanding, long prompts, structured JSON
    extraction)."""

    def test_missing_api_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            get_llm_strict("gemini/gemini-1.5-flash", temperature=0.0)

    def test_present_api_key_constructs_llm(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake_gemini_test_key")
        fake_llm = MagicMock()
        llm_class = MagicMock(return_value=fake_llm)
        monkeypatch.setattr("backend.llm.llm_client.LLM", llm_class)

        result = get_llm_strict("gemini/gemini-1.5-flash", temperature=0.0)
        assert result is fake_llm
        _, kwargs = llm_class.call_args
        assert kwargs["api_key"] == "fake_gemini_test_key"
        assert kwargs["model"] == "gemini/gemini-1.5-flash"


class TestOpenRouterProvider:
    """New 2026-08-23: model-testing fallback (permanent free catalog,
    tighter rate limits -- for benchmarking several free open models
    without rewriting this client, not a full evaluation run)."""

    def test_missing_api_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            get_llm_strict("openrouter/meta-llama/llama-3.1-8b-instruct:free", temperature=0.0)

    def test_present_api_key_constructs_llm(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake_openrouter_test_key")
        fake_llm = MagicMock()
        llm_class = MagicMock(return_value=fake_llm)
        monkeypatch.setattr("backend.llm.llm_client.LLM", llm_class)

        result = get_llm_strict("openrouter/meta-llama/llama-3.1-8b-instruct:free", temperature=0.0)
        assert result is fake_llm
        _, kwargs = llm_class.call_args
        assert kwargs["api_key"] == "fake_openrouter_test_key"
