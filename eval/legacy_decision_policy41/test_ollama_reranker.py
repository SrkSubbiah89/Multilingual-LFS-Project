"""
Tests for eval/legacy_decision_policy41/ollama_reranker.py (Task 43,
Ollama variant).

Fully hermetic: every "httpx.Client" here is a fake object -- no real
HTTP call to localhost:11434 or anywhere else is ever made in this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.legacy_decision_policy41 import ollama_reranker as orr  # noqa: E402


class _FakeHttpResponse:
    def __init__(self, json_body=None, status_code=200):
        self._json_body = json_body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://localhost:11434/api/chat")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self):
        return self._json_body


class FakeOllamaClient:
    """`responses` is a list of either a dict (successful JSON body) or an
    Exception to raise, consumed in order, one per `.post()` call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, json):
        self.calls.append({"url": url, "json": json})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, int):  # HTTP status code shorthand
            return _FakeHttpResponse(status_code=item)
        return _FakeHttpResponse(json_body=item)


def _success_body(text):
    return {"message": {"content": text}}


# ---------------------------------------------------------------------------
# make_ollama_reranker
# ---------------------------------------------------------------------------

def test_reranker_posts_correct_model_and_temperature():
    client = FakeOllamaClient([_success_body('{"selected_code": "1000", "reasoning": "ok"}')])
    reranker = orr.make_ollama_reranker(model="llama3.2:latest", client=client)
    raw = reranker("some prompt")
    assert raw == '{"selected_code": "1000", "reasoning": "ok"}'
    call = client.calls[0]
    assert call["json"]["model"] == "llama3.2:latest"
    assert call["json"]["options"]["temperature"] == 0.0
    assert call["json"]["messages"] == [{"role": "user", "content": "some prompt"}]
    assert call["json"]["stream"] is False


def test_reranker_retries_on_connection_error_then_succeeds():
    client = FakeOllamaClient([
        httpx.ConnectError("refused"),
        _success_body('{"selected_code": "2000", "reasoning": "recovered"}'),
    ])
    reranker = orr.make_ollama_reranker(client=client)
    raw = reranker("x")
    assert raw == '{"selected_code": "2000", "reasoning": "recovered"}'
    assert len(client.calls) == 2


def test_reranker_gives_up_after_max_attempts_and_marks_fatal():
    client = FakeOllamaClient([httpx.TimeoutException("slow")] * orr._MAX_ATTEMPTS)
    tracker = {}
    reranker = orr.make_ollama_reranker(client=client, fatal_tracker=tracker)
    with pytest.raises(RuntimeError, match="failed after"):
        reranker("x")
    assert len(client.calls) == orr._MAX_ATTEMPTS
    assert tracker.get("error") is not None


def test_reranker_marks_fatal_immediately_on_model_not_found():
    client = FakeOllamaClient([404])
    tracker = {}
    reranker = orr.make_ollama_reranker(client=client, fatal_tracker=tracker)
    with pytest.raises(httpx.HTTPStatusError):
        reranker("x")
    assert len(client.calls) == 1  # no retry on a 404
    assert tracker.get("error") is not None


def test_reranker_without_fatal_tracker_still_raises_on_persistent_failure():
    client = FakeOllamaClient([httpx.ConnectError("refused")] * orr._MAX_ATTEMPTS)
    reranker = orr.make_ollama_reranker(client=client)  # no tracker
    with pytest.raises(RuntimeError):
        reranker("x")


# ---------------------------------------------------------------------------
# Integration with run_dev_with_live_reranker (reranker-agnostic reuse)
# ---------------------------------------------------------------------------

def test_ollama_reranker_works_with_run_dev_with_live_reranker(tmp_path):
    from eval.legacy_decision_policy41.live_reranker import run_dev_with_live_reranker
    from eval.legacy_decision_policy41.flat_retrieval_adapter import DevRow
    from eval.legacy_decision_policy41.test_flat_retrieval_adapter import FakeStore, _five_hits

    hits = _five_hits(base_score=0.5)  # below threshold
    store = FakeStore({"plumber": hits})
    client = FakeOllamaClient([_success_body('{"selected_code": "1002", "reasoning": "chosen"}')])
    reranker = orr.make_ollama_reranker(client=client)
    row = DevRow(case_id="C1", input_text="plumber", input_language="en", gold_isco_4digit="1002")

    report = run_dev_with_live_reranker(store, [row], reranker)
    assert report.n_llm_ranked == 1
    assert report.n_correct == 1
    assert report.rows[0].method == "llm_ranked"
