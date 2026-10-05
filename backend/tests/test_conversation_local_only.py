"""Direct correction helpers honor local-only inference without altering answers."""

import json
from unittest.mock import MagicMock

import pytest

from backend.agents import conversation_manager


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setenv("LFS_LOCAL_ONLY", "true")
    for key in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(key, "configured-dummy-cloud-key-for-isolated-test")
    # Pure correction logic needs no CrewAI agent or model initialization.
    return object.__new__(conversation_manager.ConversationManager)


def validating_context(manager):
    context = manager.new_context(session_id=1, language="en")
    context.state = conversation_manager.ConversationState.VALIDATING
    context.collected_data = {
        "employment_status": "employed", "education_level": "secondary",
        "job_title": "nurse", "main_skills": "patient care",
    }
    return context


@pytest.mark.parametrize("helper", ["_call_groq_json", "_call_anthropic_json", "_call_gemini_json"])
def test_direct_cloud_helpers_do_not_construct_requests_with_nonempty_keys(monkeypatch, manager, helper):
    request = MagicMock(side_effect=AssertionError("Cloud request must not be created"))
    connection = MagicMock(side_effect=AssertionError("Cloud connection must not open"))
    monkeypatch.setattr(conversation_manager.urllib.request, "Request", request)
    monkeypatch.setattr(conversation_manager.urllib.request, "urlopen", connection)

    assert getattr(manager, helper)("Correct the job title") is None

    request.assert_not_called()
    connection.assert_not_called()


@pytest.mark.parametrize("configured_provider", ["ollama", "groq"])
def test_local_correction_failure_does_not_attempt_cloud_or_change_answers(monkeypatch, manager, configured_provider):
    monkeypatch.setenv("CORRECTION_LLM_PROVIDER", configured_provider)
    local = MagicMock(return_value=None)
    cloud = MagicMock(side_effect=AssertionError("Local-only must not invoke cloud fallback"))
    monkeypatch.setattr(manager, "_call_ollama_json", local)
    for helper in ("_call_groq_json", "_call_anthropic_json", "_call_gemini_json"):
        monkeypatch.setattr(manager, helper, cloud)
    context = validating_context(manager)
    original_answers = dict(context.collected_data)

    manager._transition(context, "change main skills please", "")

    local.assert_called_once()
    cloud.assert_not_called()
    assert context.collected_data == original_answers
    assert context.corrected_fields == set()
    assert context.correction_no_target
    assert not context.correction_applied
    assert context.state == conversation_manager.ConversationState.VALIDATING


def test_local_correction_success_preserves_existing_canonical_validation(monkeypatch, manager):
    monkeypatch.setenv("CORRECTION_LLM_PROVIDER", "groq")
    monkeypatch.setattr(manager, "_call_ollama_json", MagicMock(return_value=json.dumps({
        "corrections": [{"field": "education_level", "value": "Bachelor's degree"}],
    })))
    cloud = MagicMock(side_effect=AssertionError("Local-only must not invoke cloud fallback"))
    for helper in ("_call_groq_json", "_call_anthropic_json", "_call_gemini_json"):
        monkeypatch.setattr(manager, helper, cloud)
    context = validating_context(manager)

    assert manager._llm_extract_correction(context, "update education please")

    assert context.collected_data["education_level"] == "bachelor"
    assert context.corrected_fields == {"education_level"}
    cloud.assert_not_called()


def test_default_correction_mode_keeps_existing_cloud_fallback(monkeypatch, manager):
    monkeypatch.delenv("LFS_LOCAL_ONLY")
    monkeypatch.setenv("CORRECTION_LLM_PROVIDER", "ollama")
    monkeypatch.setattr(manager, "_call_ollama_json", MagicMock(return_value=None))
    cloud = MagicMock(return_value=json.dumps({
        "corrections": [{"field": "main_skills", "value": "first aid"}],
    }))
    monkeypatch.setattr(manager, "_call_anthropic_json", cloud)
    context = validating_context(manager)

    assert manager._llm_extract_correction(context, "update main skills please")

    cloud.assert_called_once()
    assert context.collected_data["main_skills"] == "first aid"
