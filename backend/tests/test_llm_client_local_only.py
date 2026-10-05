"""Local-only app inference stays on the configured installed Ollama model."""

import json
from unittest.mock import MagicMock

import pytest

from backend.llm import llm_client


@pytest.fixture(autouse=True)
def isolated_mode(monkeypatch):
    monkeypatch.delenv("LFS_LOCAL_ONLY", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_EXCLUDE", raising=False)


def installed_local_model(monkeypatch):
    monkeypatch.setattr(llm_client, "MODEL_GENERAL", "ollama/qwen2.5:3b")
    monkeypatch.setattr(llm_client, "_OLLAMA_BASE_URL", "http://127.0.0.1:11435")
    monkeypatch.setattr(llm_client, "_ollama_is_running", lambda: True)

    class TagsResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"models": [{"name": "qwen2.5:3b"}]}).encode()

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", MagicMock(return_value=TagsResponse()))
    local_llm = MagicMock(model="ollama/qwen2.5:3b")
    factory = MagicMock(return_value=local_llm)
    monkeypatch.setattr(llm_client, "LLM", factory)
    cloud = MagicMock(side_effect=AssertionError("Cloud provider factory must not run"))
    fallback = MagicMock(side_effect=AssertionError("Provider fallback must not run"))
    monkeypatch.setattr(llm_client, "_get_claude_llm", cloud)
    monkeypatch.setattr(llm_client, "_get_general_llm_with_fallback", fallback)
    return local_llm, factory, cloud, fallback


@pytest.mark.parametrize("task, expected_temperature", [
    (llm_client.TaskType.GENERAL, 0.3), (llm_client.TaskType.CRITICAL, 0.0),
])
def test_local_only_selects_installed_ollama_for_both_task_types(monkeypatch, task, expected_temperature):
    monkeypatch.setenv("LFS_LOCAL_ONLY", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "configured-test-only-cloud-key")
    local_llm, factory, cloud, fallback = installed_local_model(monkeypatch)
    trace = {}

    result = llm_client.get_llm(task, trace=trace)

    assert result is local_llm
    factory.assert_called_once_with(
        model="ollama/qwen2.5:3b", temperature=expected_temperature,
        base_url="http://127.0.0.1:11435", timeout=llm_client._OLLAMA_INFERENCE_TIMEOUT,
    )
    cloud.assert_not_called()
    fallback.assert_not_called()
    assert trace == {
        "local_only": True, "resolved_provider": "ollama", "resolved_model": "ollama/qwen2.5:3b",
        "attempted_providers": ["ollama"], "failure_reasons": {},
    }


@pytest.mark.parametrize("task", ["general", "critical"])
def test_local_only_preserves_explicit_temperature_override(monkeypatch, task):
    monkeypatch.setenv("LFS_LOCAL_ONLY", "true")
    _, factory, _, _ = installed_local_model(monkeypatch)

    llm_client.get_llm(task, temperature=0.15)

    assert factory.call_args.kwargs["temperature"] == 0.15


@pytest.mark.parametrize("task", ["general", "critical"])
def test_unavailable_ollama_refuses_cloud_and_clears_stale_trace(monkeypatch, task):
    monkeypatch.setenv("LFS_LOCAL_ONLY", "true")
    _, factory, cloud, fallback = installed_local_model(monkeypatch)
    monkeypatch.setattr(llm_client, "_ollama_is_running", lambda: False)
    trace = {"resolved_provider": "anthropic", "resolved_model": llm_client.MODEL_CRITICAL}

    with pytest.raises(RuntimeError, match="LFS_LOCAL_ONLY=true.*Cloud fallback is disabled"):
        llm_client.get_llm(task, trace=trace)

    factory.assert_not_called()
    cloud.assert_not_called()
    fallback.assert_not_called()
    assert trace["attempted_providers"] == ["ollama"]
    assert trace["local_only"] is True
    assert "resolved_provider" not in trace
    assert "resolved_model" not in trace
    assert "not reachable" in trace["failure_reasons"]["ollama"]


def test_uninstalled_configured_model_fails_without_cloud(monkeypatch):
    monkeypatch.setenv("LFS_LOCAL_ONLY", "true")
    _, factory, cloud, fallback = installed_local_model(monkeypatch)
    monkeypatch.setattr(llm_client, "MODEL_GENERAL", "ollama/model-not-installed:latest")

    with pytest.raises(RuntimeError, match="not pulled"):
        llm_client.get_llm("critical")

    factory.assert_not_called()
    cloud.assert_not_called()
    fallback.assert_not_called()


@pytest.mark.parametrize("flag", ["1", "yes", "TRUE", " true "])
def test_supported_local_only_flag_values(monkeypatch, flag):
    monkeypatch.setenv("LFS_LOCAL_ONLY", flag)
    local_llm, _, _, _ = installed_local_model(monkeypatch)
    assert llm_client.get_llm("critical") is local_llm


@pytest.mark.parametrize("flag", [None, "false", "0", "no"])
def test_default_and_disabled_modes_preserve_existing_provider_selection(monkeypatch, flag):
    if flag is not None:
        monkeypatch.setenv("LFS_LOCAL_ONLY", flag)
    general = MagicMock()
    critical = MagicMock()
    general_factory = MagicMock(return_value=general)
    critical_factory = MagicMock(return_value=critical)
    monkeypatch.setattr(llm_client, "_get_general_llm_with_fallback", general_factory)
    monkeypatch.setattr(llm_client, "_get_claude_llm", critical_factory)

    assert llm_client.get_llm("general") is general
    assert llm_client.get_llm("critical") is critical
    general_factory.assert_called_once_with(0.3, None)
    critical_factory.assert_called_once_with(temperature=0.0)


def test_local_only_does_not_substitute_explicitly_pinned_strict_model(monkeypatch):
    monkeypatch.setenv("LFS_LOCAL_ONLY", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "pinned-evaluation-test-only-key")
    factory = MagicMock()
    monkeypatch.setattr(llm_client, "LLM", factory)

    llm_client.get_llm_strict(llm_client.MODEL_CRITICAL, temperature=0)

    factory.assert_called_once_with(
        model=llm_client.MODEL_CRITICAL, temperature=0, api_key="pinned-evaluation-test-only-key",
    )
