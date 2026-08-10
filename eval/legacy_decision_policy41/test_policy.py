"""
Tests for eval/legacy_decision_policy41/policy.py (Task 41).

Fully hermetic: only fake candidates and injected fake reranker callables
are used anywhere in this file. No test reads WISCO, loads/downloads an
embedding model, connects to Qdrant, calls an LLM/provider, constructs a
CrewAI agent, or imports any current evaluation/classifier module.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.legacy_decision_policy41.policy import (  # noqa: E402
    ALLOWED_METHODS,
    HISTORICAL_CANDIDATE_COUNT,
    HISTORICAL_FALLBACK_REASONING,
    HISTORICAL_THRESHOLD,
    METHOD_LLM_RANKED,
    METHOD_SEMANTIC,
    PolicyCandidate,
    build_prompt_text,
    classify_with_policy,
)


def _candidate(code="2512", conf=0.5, title_en=None, title_ar="مطور برمجيات",
                level=4, description="Develops software applications."):
    return PolicyCandidate(
        code=code,
        title_en=title_en or f"Occupation {code}",
        title_ar=title_ar,
        level=level,
        confidence=conf,
        description=description,
    )


def _five(confidences):
    assert len(confidences) == 5
    return [
        _candidate(code=f"{1000 + i}", conf=c) for i, c in enumerate(confidences)
    ]


def _never_called(*_args, **_kwargs):
    raise AssertionError("reranker must not be invoked on the fast path")


# ---------------------------------------------------------------------------
# 1-2. candidate count
# ---------------------------------------------------------------------------

def test_exactly_five_candidates_required_and_accepted():
    candidates = _five([0.95, 0.5, 0.4, 0.3, 0.2])
    result = classify_with_policy("software engineer", candidates, reranker=_never_called)
    assert result.method == METHOD_SEMANTIC


@pytest.mark.parametrize("n", [0, 1, 4, 6, 10])
def test_wrong_candidate_count_fails_closed(n):
    candidates = _five([0.95, 0.5, 0.4, 0.3, 0.2])[:n] if n <= 5 else _five([0.95, 0.5, 0.4, 0.3, 0.2]) + [_candidate(code="9999")] * (n - 5)
    with pytest.raises(ValueError):
        classify_with_policy("x", candidates, reranker=_never_called)


# ---------------------------------------------------------------------------
# 3-5. threshold boundary behavior
# ---------------------------------------------------------------------------

def test_confidence_exactly_threshold_takes_semantic_path_zero_reranker_calls():
    candidates = _five([HISTORICAL_THRESHOLD, 0.5, 0.4, 0.3, 0.2])
    result = classify_with_policy("x", candidates, reranker=_never_called)
    assert result.method == METHOD_SEMANTIC
    assert result.reranker_invocations == 0
    assert result.primary is candidates[0]


def test_confidence_above_threshold_takes_semantic_path_zero_reranker_calls():
    candidates = _five([0.99, 0.5, 0.4, 0.3, 0.2])
    result = classify_with_policy("x", candidates, reranker=_never_called)
    assert result.method == METHOD_SEMANTIC
    assert result.reranker_invocations == 0


def test_confidence_below_threshold_invokes_reranker_exactly_once():
    candidates = _five([0.91, 0.5, 0.4, 0.3, 0.2])
    calls = []

    def reranker(prompt_text):
        calls.append(prompt_text)
        return '{"selected_code": "1000", "reasoning": "best fit"}'

    result = classify_with_policy("x", candidates, reranker=reranker)
    assert len(calls) == 1
    assert result.reranker_invocations == 1
    assert result.method == METHOD_LLM_RANKED


# ---------------------------------------------------------------------------
# 6-10. reranker response handling
# ---------------------------------------------------------------------------

def test_valid_json_response_produces_llm_ranked_with_selected_candidate():
    candidates = _five([0.5, 0.4, 0.3, 0.2, 0.1])
    result = classify_with_policy(
        "x", candidates,
        reranker=lambda p: '{"selected_code": "1002", "reasoning": "matches best"}',
    )
    assert result.method == METHOD_LLM_RANKED
    assert result.primary.code == "1002"
    assert result.reasoning == "matches best"


def test_invalid_json_falls_back_to_semantic_top():
    candidates = _five([0.5, 0.4, 0.3, 0.2, 0.1])
    result = classify_with_policy("x", candidates, reranker=lambda p: "not json at all {{{")
    assert result.method == METHOD_LLM_RANKED
    assert result.primary is candidates[0]
    assert result.reasoning == HISTORICAL_FALLBACK_REASONING


def test_missing_selected_code_falls_back_to_semantic_top():
    candidates = _five([0.5, 0.4, 0.3, 0.2, 0.1])
    result = classify_with_policy("x", candidates, reranker=lambda p: '{"reasoning": "no code given"}')
    assert result.primary is candidates[0]
    assert result.reasoning == HISTORICAL_FALLBACK_REASONING


def test_out_of_candidate_code_falls_back_to_semantic_top():
    candidates = _five([0.5, 0.4, 0.3, 0.2, 0.1])
    result = classify_with_policy(
        "x", candidates,
        reranker=lambda p: '{"selected_code": "9999", "reasoning": "unrelated code"}',
    )
    assert result.primary is candidates[0]
    assert result.reasoning == HISTORICAL_FALLBACK_REASONING


def test_reranker_exception_falls_back_to_semantic_top():
    candidates = _five([0.5, 0.4, 0.3, 0.2, 0.1])

    def broken_reranker(prompt_text):
        raise RuntimeError("simulated reranker failure")

    result = classify_with_policy("x", candidates, reranker=broken_reranker)
    assert result.primary is candidates[0]
    assert result.reasoning == HISTORICAL_FALLBACK_REASONING
    assert result.reranker_invocations == 1
    assert result.method == METHOD_LLM_RANKED


# ---------------------------------------------------------------------------
# 11-12. prompt content and candidate order
# ---------------------------------------------------------------------------

def test_prompt_includes_all_required_fields_for_all_five_candidates():
    candidates = [
        _candidate(code=f"300{i}", conf=0.5 - i * 0.01, title_en=f"Title EN {i}",
                   title_ar=f"عنوان {i}", level=i + 1, description=f"Description {i}")
        for i in range(5)
    ]
    prompt = build_prompt_text("plumber", "construction sector", "en", candidates)
    assert '"plumber"' in prompt
    assert "construction sector" in prompt
    assert "written in English" in prompt
    for c in candidates:
        assert c.code in prompt
        assert c.title_en in prompt
        assert c.title_ar in prompt
        assert f"Level {c.level}" in prompt
        assert c.description in prompt
        assert f"{c.confidence:.2%}" in prompt
    assert '"selected_code"' in prompt and '"reasoning"' in prompt


def test_candidate_order_preserved_in_prompt_and_result():
    candidates = _five([0.5, 0.45, 0.4, 0.35, 0.3])
    prompt = build_prompt_text("x", "", "en", candidates)
    positions = [prompt.index(c.code) for c in candidates]
    assert positions == sorted(positions)

    result = classify_with_policy(
        "x", candidates,
        reranker=lambda p: '{"selected_code": "1003", "reasoning": "ok"}',
    )
    assert result.primary.code == "1003"
    assert [c.code for c in candidates] == ["1000", "1001", "1002", "1003", "1004"]


# ---------------------------------------------------------------------------
# 13. only semantic/llm_ranked labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("confidences,reranker", [
    ([0.95, 0.5, 0.4, 0.3, 0.2], _never_called),
    ([0.5, 0.4, 0.3, 0.2, 0.1], lambda p: '{"selected_code": "1000", "reasoning": "x"}'),
    ([0.5, 0.4, 0.3, 0.2, 0.1], lambda p: "garbage"),
    ([0.5, 0.4, 0.3, 0.2, 0.1], lambda p: (_ for _ in ()).throw(RuntimeError())),
])
def test_only_semantic_and_llm_ranked_labels_possible(confidences, reranker):
    candidates = _five(confidences)
    result = classify_with_policy("x", candidates, reranker=reranker)
    assert result.method in ALLOWED_METHODS


# ---------------------------------------------------------------------------
# 14. no prohibited import in the compatibility-policy module
# ---------------------------------------------------------------------------

_PROHIBITED_IMPORT_TOKENS = (
    "qdrant", "crewai", "backend.rag", "backend.llm", "backend.agents",
    "sentence_transformers", "anthropic", "openai", "torch",
)


def test_policy_module_has_no_prohibited_imports():
    policy_source = Path(__file__).resolve().parent.joinpath("policy.py").read_text(encoding="utf-8")
    import_lines = [
        line.strip() for line in policy_source.splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    assert import_lines, "expected at least the stdlib imports"
    lowered = "\n".join(import_lines).lower()
    for token in _PROHIBITED_IMPORT_TOKENS:
        assert token not in lowered, f"prohibited import token {token!r} found in policy.py imports"


# ---------------------------------------------------------------------------
# 16. no network-capable reranker / no network call from this module itself
# ---------------------------------------------------------------------------

def test_no_network_call_occurs_during_classification(monkeypatch):
    def _blocked_socket(*args, **kwargs):
        raise AssertionError("no socket should be opened by classify_with_policy or its fake reranker")

    monkeypatch.setattr(socket, "socket", _blocked_socket)

    candidates = _five([0.95, 0.5, 0.4, 0.3, 0.2])
    fast_result = classify_with_policy("x", candidates, reranker=_never_called)
    assert fast_result.method == METHOD_SEMANTIC

    low_candidates = _five([0.5, 0.4, 0.3, 0.2, 0.1])
    slow_result = classify_with_policy(
        "x", low_candidates,
        reranker=lambda p: '{"selected_code": "1000", "reasoning": "purely in-memory"}',
    )
    assert slow_result.method == METHOD_LLM_RANKED
