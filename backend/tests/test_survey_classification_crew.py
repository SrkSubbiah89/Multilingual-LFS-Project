"""Classification provenance and real CrewAI collaboration, without providers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.agents.survey_classification_crew import SurveyClassificationCrew


@pytest.fixture
def classifiers():
    isco, isic, isced = MagicMock(), MagicMock(), MagicMock()
    isco.classify.return_value = SimpleNamespace(
        primary=SimpleNamespace(code="2512", confidence=0.41), method="rag",
        hitl_required=True, hierarchy_path=["2", "25", "251", "2512"])
    isic.classify.return_value = SimpleNamespace(
        section="J", class_code="6201", confidence=0.65, method="keyword")
    isced.classify.return_value = SimpleNamespace(
        level=6, detailed_code="0613", confidence=0.7, method="rule")
    return isco, isic, isced


def fake_framework(monkeypatch, behavior=None):
    captured = {}
    monkeypatch.setattr("crewai.Agent", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr("crewai.Task", lambda **kwargs: SimpleNamespace(**kwargs))

    class FakeCrew:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def kickoff(self):
            if behavior:
                return behavior(captured)
            for agent in captured["agents"]:
                agent.tools[0].run()
            return '{"isco_code": "9999", "confidence": 1.0}'

    monkeypatch.setattr("crewai.Crew", FakeCrew)
    return captured


def test_specialists_share_evidence_and_preserve_original_objects(classifiers, monkeypatch):
    captured = fake_framework(monkeypatch)
    crew = SurveyClassificationCrew(*classifiers, llm=object())
    result = crew.classify("Software developer", "IT company", "Bachelor Computer Science",
                           language="hi", isco_context="programming duties",
                           use_llm=False, isic_cross_hints={"isco_code": "2512"})
    assert result.isco is classifiers[0].classify.return_value
    assert result.isic is classifiers[1].classify.return_value
    assert result.isced is classifiers[2].classify.return_value
    assert result.isco.primary.confidence == 0.41
    assert result.isco.hitl_required is True
    classifiers[0].classify.assert_called_once_with(
        "Software developer", context="programming duties", language="hi", use_llm=False)
    classifiers[1].classify.assert_called_once_with("IT company", cross_hints={"isco_code": "2512"})
    classifiers[2].classify.assert_called_once_with("Bachelor Computer Science")
    assert result.execution["mode"] == "crewai_sequential"
    assert result.execution["cooperation_verified"] is True
    assert len(captured["agents"]) == 4
    assert captured["tasks"][-1].context == captured["tasks"][:-1]
    for agent in captured["agents"]:
        assert agent.max_iter == 4 and agent.max_retry_limit == 0
        assert agent.max_execution_time == 180 and agent.allow_delegation is False
    assert all(task.guardrail_max_retries == 2 for task in captured["tasks"])


def test_single_dimension_still_has_two_collaborating_agents(classifiers, monkeypatch):
    captured = fake_framework(monkeypatch)
    result = SurveyClassificationCrew(*classifiers, llm=object()).classify(job_title="Developer")
    assert len(captured["agents"]) == 2
    assert result.execution["requested_dimensions"] == ["isco"]
    classifiers[1].classify.assert_not_called()
    classifiers[2].classify.assert_not_called()


def test_empty_request_creates_no_crew_or_llm(classifiers, monkeypatch):
    monkeypatch.setattr("crewai.Crew", MagicMock(side_effect=AssertionError("Unexpected crew")))
    monkeypatch.setattr("backend.agents.survey_classification_crew.get_llm",
                        MagicMock(side_effect=AssertionError("Unexpected LLM")))
    result = SurveyClassificationCrew(*classifiers).classify(job_title=" ")
    assert result.execution["mode"] == "no_classification"
    for classifier in classifiers:
        classifier.classify.assert_not_called()


def test_repeat_tool_calls_are_memoized(classifiers, monkeypatch):
    def repeated(captured):
        captured["agents"][0].tools[0].run()
        captured["agents"][0].tools[0].run()
        captured["agents"][-1].tools[0].run()
    fake_framework(monkeypatch, repeated)
    result = SurveyClassificationCrew(*classifiers, llm=object()).classify(job_title="Developer")
    classifiers[0].classify.assert_called_once()
    assert result.execution["tool_executed_dimensions"] == ["isco"]


def test_crew_failure_preserves_completed_tool_and_falls_back_only_missing(classifiers, monkeypatch):
    def partial(captured):
        captured["agents"][0].tools[0].run()
        raise TimeoutError("late worker failure")
    fake_framework(monkeypatch, partial)
    result = SurveyClassificationCrew(*classifiers, llm=object()).classify(
        "Developer", "IT company", "Bachelor")
    for classifier in classifiers:
        classifier.classify.assert_called_once()
    assert result.isco is classifiers[0].classify.return_value
    assert result.execution["fallback_dimensions"] == ["isic", "isced"]
    assert result.execution["failure_reasons"]["crew"] == "TimeoutError"
    assert result.execution["mode"] == "crewai_fallback"


def test_hallucinated_final_answer_does_not_replace_missing_tool(classifiers, monkeypatch):
    fake_framework(monkeypatch, lambda captured: '{"isco_code": "9999", "confidence": 1.0}')
    result = SurveyClassificationCrew(*classifiers, llm=object()).classify(job_title="Developer")
    assert result.isco.primary.code == "2512"
    assert result.isco.primary.confidence == 0.41
    assert result.execution["fallback_dimensions"] == ["isco"]
    assert result.execution["cooperation_verified"] is False
    assert result.execution["crew_completed"] is False


def test_classifier_failure_is_explicit_and_not_retried(classifiers, monkeypatch):
    classifiers[0].classify.side_effect = RuntimeError("URL may contain private input")
    fake_framework(monkeypatch)
    result = SurveyClassificationCrew(*classifiers, llm=object()).classify(
        "Developer", "IT company", "Bachelor")
    classifiers[0].classify.assert_called_once()
    assert result.isco is None
    assert result.isic is classifiers[1].classify.return_value
    assert result.isced is classifiers[2].classify.return_value
    assert result.execution["failed_dimensions"] == ["isco"]
    assert result.execution["failure_reasons"]["isco"] == "RuntimeError"
    assert "private" not in str(result.execution)


def test_direct_mode_never_resolves_llm(classifiers, monkeypatch):
    monkeypatch.setattr("backend.agents.survey_classification_crew.get_llm",
                        MagicMock(side_effect=AssertionError("Unexpected LLM")))
    result = SurveyClassificationCrew(*classifiers).classify(
        "Developer", "IT company", "Bachelor", enable_crew=False)
    assert result.execution["mode"] == "direct"
    assert result.execution["crew_attempted"] is False
    for classifier in classifiers:
        classifier.classify.assert_called_once()


def test_coordinated_hints_come_from_captured_isco_result(classifiers, monkeypatch):
    fake_framework(monkeypatch)
    result = SurveyClassificationCrew(*classifiers, llm=object()).classify(
        "Developer", "IT company", isic_cross_hints="from_isco")
    classifiers[1].classify.assert_called_once_with("IT company", cross_hints={"isco_code": "2512"})
    assert result.execution["cooperation_verified"] is True


def test_coordinated_hints_are_absent_without_isco_evidence(classifiers, monkeypatch):
    fake_framework(monkeypatch)
    SurveyClassificationCrew(*classifiers, llm=object()).classify(
        industry_text="IT company", isic_cross_hints="from_isco")
    classifiers[1].classify.assert_called_once_with("IT company", cross_hints=None)


def test_provider_resolution_failure_retains_direct_results(classifiers, monkeypatch):
    monkeypatch.setattr("backend.agents.survey_classification_crew.get_llm",
                        MagicMock(side_effect=RuntimeError("No configured providers")))
    result = SurveyClassificationCrew(*classifiers).classify("Developer", "IT company", "Bachelor")
    assert result.execution["mode"] == "crewai_fallback"
    assert result.execution["failure_reasons"]["crew"] == "RuntimeError"
    assert result.execution["fallback_dimensions"] == ["isco", "isic", "isced"]
    for classifier in classifiers:
        classifier.classify.assert_called_once()


def test_tool_cannot_substitute_respondent_input(classifiers, monkeypatch):
    def substitute(captured):
        captured["agents"][0].tools[0].run(text="CEO")
    fake_framework(monkeypatch, substitute)
    result = SurveyClassificationCrew(*classifiers, llm=object()).classify(job_title="Developer")
    classifiers[0].classify.assert_called_once_with(
        "Developer", context="", language="en", use_llm=True)
    assert result.isco.primary.code == "2512"


def test_real_crewai_runtime_invokes_specialist_and_auditor(classifiers):
    # This executes actual installed Crew/Agent/Task/tool objects and their
    # handoff. Only the provider is replaced with a scripted, offline BaseLLM.
    from crewai.llms.base_llm import BaseLLM

    class OfflineToolLLM(BaseLLM):
        def __init__(self):
            super().__init__(model="offline-tool-test", temperature=0)
            self.roles = []

        def call(self, messages, tools=None, callbacks=None, available_functions=None,
                 from_task=None, from_agent=None, response_model=None):
            self.roles.append(from_agent.role)
            return ("Thought: I will invoke the bound evidence tool.\n"
                    f"Action: {from_agent.tools[0].name}\nAction Input: {{}}")

        def supports_function_calling(self):
            return False

    llm = OfflineToolLLM()
    result = SurveyClassificationCrew(*classifiers, llm=llm).classify(
        "Developer", "IT company", "Bachelor")
    assert result.execution["mode"] == "crewai_sequential"
    assert result.execution["cooperation_verified"] is True
    assert llm.roles == ["Occupation Classification Specialist", "Industry Classification Specialist",
                         "Education Classification Specialist", "Classification Evidence Auditor"]
    for classifier in classifiers:
        classifier.classify.assert_called_once()


def test_real_crewai_guardrails_retry_skipped_tools_before_handoff(classifiers):
    from crewai.llms.base_llm import BaseLLM

    class SkipThenToolLLM(BaseLLM):
        def __init__(self):
            super().__init__(model="offline-retry-test", temperature=0)
            self.calls = {}

        def call(self, messages, tools=None, callbacks=None, available_functions=None,
                 from_task=None, from_agent=None, response_model=None):
            role = from_agent.role
            self.calls[role] = self.calls.get(role, 0) + 1
            if self.calls[role] == 1:
                return 'Final Answer: {"code":"9999","confidence":1.0}'
            return ("Thought: I must obtain actual evidence.\n"
                    f"Action: {from_agent.tools[0].name}\nAction Input: {{}}")

        def supports_function_calling(self):
            return False

    llm = SkipThenToolLLM()
    result = SurveyClassificationCrew(*classifiers, llm=llm).classify(
        "Developer", "IT company", "Bachelor")
    assert result.execution["crew_completed"] is True
    assert result.execution["cooperation_verified"] is True
    assert result.execution["guardrail_rejections"] == {"isco": 1, "isic": 1, "isced": 1, "audit": 1}
    assert set(llm.calls.values()) == {2}
    assert result.isco.primary.code == "2512" and result.isco.primary.confidence == 0.41
    for classifier in classifiers:
        classifier.classify.assert_called_once()


@pytest.mark.parametrize("skipped_role", ["Occupation Classification Specialist", "Classification Evidence Auditor"])
def test_real_crewai_guardrails_stop_persistent_tool_skipping(classifiers, skipped_role):
    from crewai.llms.base_llm import BaseLLM

    class SkipRequiredToolLLM(BaseLLM):
        def __init__(self):
            super().__init__(model="offline-guard-failure-test", temperature=0)
            self.calls = {}

        def call(self, messages, tools=None, callbacks=None, available_functions=None,
                 from_task=None, from_agent=None, response_model=None):
            role = from_agent.role
            self.calls[role] = self.calls.get(role, 0) + 1
            if role == skipped_role:
                return 'Final Answer: {"code":"9999","confidence":1.0}'
            return f"Thought: Use actual evidence.\nAction: {from_agent.tools[0].name}\nAction Input: {{}}"

        def supports_function_calling(self):
            return False

    llm = SkipRequiredToolLLM()
    result = SurveyClassificationCrew(*classifiers, llm=llm).classify(
        "Developer", "IT company", "Bachelor")
    assert llm.calls[skipped_role] == 3  # Initial answer plus two bounded retries.
    assert result.execution["crew_completed"] is False
    assert result.execution["cooperation_verified"] is False
    assert result.execution["mode"] == "crewai_fallback"
    if skipped_role == "Classification Evidence Auditor":
        assert result.execution["fallback_dimensions"] == []
        assert result.execution["tool_executed_dimensions"] == ["isco", "isic", "isced"]
    else:
        assert result.execution["fallback_dimensions"] == ["isco", "isic", "isced"]
    for classifier in classifiers:
        classifier.classify.assert_called_once()


def test_guardrails_supply_captured_evidence_to_next_task(classifiers, monkeypatch):
    captured = fake_framework(monkeypatch)
    result = SurveyClassificationCrew(*classifiers, llm=object()).classify(job_title="Developer")
    accepted, evidence = captured["tasks"][0].guardrail(SimpleNamespace(raw='{"code":"9999"}'))
    import json
    assert accepted is True
    assert json.loads(evidence)["code"] == "2512"
    assert json.loads(evidence)["confidence"] == 0.41
    assert result.execution["crew_completed"] is True


def test_native_ollama_transport_executes_actual_crewai_tools(classifiers, monkeypatch):
    import io
    import json
    requests = []

    def native_response(request, timeout):
        body = json.loads(request.data)
        requests.append((request.full_url, body, timeout))
        name = body["tools"][0]["function"]["name"]
        return io.BytesIO(json.dumps({
            "message": {"role": "assistant", "tool_calls": [
                {"function": {"name": name, "arguments": {}}}]},
            "prompt_eval_count": 20, "eval_count": 4,
        }).encode())

    monkeypatch.setattr("backend.agents.survey_classification_crew.urllib.request.urlopen", native_response)
    selected = SimpleNamespace(model="ollama/qwen2.5:3b", base_url="http://localhost:11434")
    result = SurveyClassificationCrew(*classifiers, llm=selected).classify("Developer", "IT company", "Bachelor")
    assert result.execution["cooperation_verified"] is True
    assert result.execution["llm_transport"] == "ollama_native_chat"
    assert result.execution["llm_model"] == "ollama/qwen2.5:3b"
    assert len(requests) == 4
    for url, body, timeout in requests:
        assert url == "http://localhost:11434/api/chat"
        assert body["model"] == "qwen2.5:3b" and body["stream"] is False
        assert body["options"] == {"temperature": 0.0, "num_predict": 128, "num_ctx": 2048}
        assert 0 < timeout <= 30
    for classifier in classifiers:
        classifier.classify.assert_called_once()


def test_native_ollama_timeout_is_not_retried(classifiers, monkeypatch):
    failed_request = MagicMock(side_effect=TimeoutError("Local inference timeout"))
    monkeypatch.setattr("backend.agents.survey_classification_crew.urllib.request.urlopen", failed_request)
    selected = SimpleNamespace(model="ollama/qwen2.5:3b", base_url="http://localhost:11434")
    result = SurveyClassificationCrew(*classifiers, llm=selected).classify("Developer", "IT company", "Bachelor")
    assert failed_request.call_count == 1
    assert result.execution["crew_completed"] is False
    assert result.execution["cooperation_verified"] is False
    assert result.execution["mode"] == "crewai_fallback"
    assert result.execution["fallback_dimensions"] == ["isco", "isic", "isced"]
    for classifier in classifiers:
        classifier.classify.assert_called_once()


def test_native_ollama_deadline_prevents_another_network_request(monkeypatch):
    from backend.agents.survey_classification_crew import _OllamaToolLLM
    now = [0.0]
    monkeypatch.setattr("backend.agents.survey_classification_crew.time.monotonic", lambda: now[0])
    adapter = _OllamaToolLLM("ollama/qwen2.5:3b", "http://localhost:11434", budget=10)
    request = MagicMock(side_effect=AssertionError("Deadline must prevent network access"))
    monkeypatch.setattr("backend.agents.survey_classification_crew.urllib.request.urlopen", request)
    now[0] = 11.0
    with pytest.raises(TimeoutError):
        adapter.call("Classify using your tool")
    request.assert_not_called()


def test_native_ollama_request_uses_remaining_deadline(monkeypatch):
    import io
    from backend.agents.survey_classification_crew import _OllamaToolLLM
    now = [0.0]
    monkeypatch.setattr("backend.agents.survey_classification_crew.time.monotonic", lambda: now[0])
    adapter = _OllamaToolLLM("ollama/qwen2.5:3b", "http://localhost:11434", budget=10)
    request = MagicMock(return_value=io.BytesIO(b'{"message":{"content":"Not a tool call"}}'))
    monkeypatch.setattr("backend.agents.survey_classification_crew.urllib.request.urlopen", request)
    now[0] = 8.0
    assert adapter.call("Classify using your tool") == "Not a tool call"
    assert request.call_args.kwargs["timeout"] == 2.0


def test_native_ollama_invalid_response_disables_further_network(monkeypatch):
    import io
    from backend.agents.survey_classification_crew import _OllamaToolLLM
    adapter = _OllamaToolLLM("ollama/qwen2.5:3b", "http://localhost:11434")
    request = MagicMock(return_value=io.BytesIO(b'{"error":"model unavailable"}'))
    monkeypatch.setattr("backend.agents.survey_classification_crew.urllib.request.urlopen", request)
    with pytest.raises(ValueError):
        adapter.call("Classify using your tool")
    with pytest.raises(TimeoutError):
        adapter.call("Retry")
    assert request.call_count == 1


def test_native_ollama_converts_executor_tool_history_to_native_schema(monkeypatch):
    import io
    import json
    from backend.agents.survey_classification_crew import _OllamaToolLLM
    request = MagicMock(return_value=io.BytesIO(b'{"message":{"content":"Done"}}'))
    monkeypatch.setattr("backend.agents.survey_classification_crew.urllib.request.urlopen", request)
    adapter = _OllamaToolLLM("ollama/qwen2.5:3b", "http://localhost:11434")
    adapter.call([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_test", "function": {"name": "retrieve_evidence", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_test", "content": "original evidence"},
    ])
    body = json.loads(request.call_args.args[0].data)
    assert body["messages"][0] == {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "retrieve_evidence", "arguments": {}}}]}
    assert body["messages"][1] == {"role": "tool", "content": "original evidence", "tool_name": "retrieve_evidence"}
