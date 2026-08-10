"""
eval/legacy_decision_policy41/ollama_reranker.py

Task 43 (Ollama variant): a free, local reranker callable for Task 41's
decision policy, used because no Anthropic credit is currently available
(see CLAUDE_TASK_43_LIVE_RERANKER_WISCO_RUN_FINAL_REPORT.md).

EXPLICIT, MAJOR DISCLOSED DEVIATION FROM HISTORICAL FIDELITY: the
historical policy's identity-check-locked model is
`anthropic/claude-3-5-sonnet-20241022` (`policy.HISTORICAL_MODEL`). This
module does NOT call that model -- it calls a local Ollama model
(`llama3.2:latest` by default, the same default this project's own
`backend/llm/llm_client.py` already uses for `TaskType.GENERAL`) instead.
This is NOT a fidelity-preserving substitution; it is a further, lower-
fidelity fallback than even Task 43's Anthropic path, used only for cost
reasons. Every prompt, threshold, candidate count, and JSON/fallback
contract in `policy.py` stays completely unmodified -- only WHICH model
answers the reranker prompt changes. Any report citing results produced
with this module must say "Ollama llama3.2, not the historical Claude 3.5
Sonnet" plainly, not "the historical policy" unqualified.

Learned directly from the Task 43 Anthropic-credit incident: a reranker
failure must never be allowed to silently look like a normal parse-
fallback. This module uses the SAME `fatal_tracker` contract
`live_reranker.py` established -- `run_dev_with_live_reranker` is fully
reranker-agnostic and works with this module's callable unchanged.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import httpx

from eval.legacy_decision_policy41.policy import HISTORICAL_TEMPERATURE

DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
_REQUEST_TIMEOUT_SECONDS = 60
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0
_RETRYABLE_EXCEPTIONS = (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError)


def make_ollama_reranker(
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
    fatal_tracker: Optional[dict] = None,
    client: Optional[httpx.Client] = None,
):
    """
    Returns a `reranker(prompt_text: str) -> str` callable, suitable for
    `policy.classify_with_policy(..., reranker=...)`, that calls a local
    Ollama server instead of Anthropic.

    Retries up to `_MAX_ATTEMPTS` times on transient connection/timeout
    errors. Any other failure (model not found, malformed server
    response, or persistent connection failure after all retries) is
    fatal: if *fatal_tracker* is supplied, the exception is recorded into
    it (same contract as `live_reranker.make_anthropic_reranker`) before
    being re-raised, so `run_dev_with_live_reranker` can abort the whole
    run immediately instead of silently falling back row after row.
    """
    http_client = client or httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS)

    def reranker(prompt_text: str) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = http_client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt_text}],
                        "stream": False,
                        "options": {"temperature": HISTORICAL_TEMPERATURE},
                    },
                )
                response.raise_for_status()
                return response.json()["message"]["content"]
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
            except Exception as exc:
                if fatal_tracker is not None:
                    fatal_tracker["error"] = exc
                raise
        final_exc = RuntimeError(f"Ollama call to {base_url} failed after {_MAX_ATTEMPTS} attempts")
        final_exc.__cause__ = last_exc
        if fatal_tracker is not None:
            fatal_tracker["error"] = final_exc
        raise final_exc

    return reranker


def check_ollama_model_available(base_url: str = DEFAULT_OLLAMA_BASE_URL, model: str = DEFAULT_OLLAMA_MODEL) -> bool:
    """Cheap, read-only pre-flight check (GET /api/tags) -- confirms the
    Ollama server is reachable and *model* is actually pulled, before any
    reranker call is attempted."""
    response = httpx.get(f"{base_url}/api/tags", timeout=5)
    response.raise_for_status()
    names = {m["name"] for m in response.json().get("models", [])}
    return model in names
