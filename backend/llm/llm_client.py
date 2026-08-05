"""
backend/llm/llm_client.py

Shared LLM client factory for the LFS project.

Task types
----------
"general"  → Ollama / Llama 3.2 (local, free)
             Fast, privacy-preserving; used by conversational agents
             (ConversationManager, LanguageProcessor NER, EmotionalIntelligence).
             Requires Ollama running locally (default: http://localhost:11434).
             Falls back to Claude 3.5 Sonnet if Ollama is unreachable and
             ANTHROPIC_API_KEY is available.

"critical" → Claude 3.5 Sonnet (Anthropic)
             High-accuracy; used for tasks where correctness is essential:
             employment-status classification, answer validation,
             audit reporting, HITL quality assessment.

Usage
-----
from backend.llm.llm_client import get_llm, TaskType

llm = get_llm(TaskType.GENERAL)          # Llama 3.2 via Ollama, temp 0.3
llm = get_llm(TaskType.CRITICAL)         # Claude 3.5 Sonnet, temp 0.0
llm = get_llm("general", temperature=0)  # override temperature

Environment variables
---------------------
OLLAMA_BASE_URL   Base URL of the Ollama server (default: http://localhost:11434)
OLLAMA_MODEL      Override the Ollama model name   (default: llama3.2)
ANTHROPIC_API_KEY Required for TaskType.CRITICAL; also used as GENERAL fallback
                  when Ollama is unreachable
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from enum import Enum

from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task type enum
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    """Semantic task category that determines which model is selected."""
    GENERAL  = "general"   # Llama 3.2 via Ollama — conversational agents, NER
    CRITICAL = "critical"  # Claude 3.5 Sonnet   — classification, validation


# ---------------------------------------------------------------------------
# Model identifiers (LiteLLM routing strings used by CrewAI)
# ---------------------------------------------------------------------------

# Ollama: model name as registered in `ollama list`; override via env var
_OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2")
MODEL_GENERAL  = f"ollama/{_OLLAMA_MODEL_NAME}"
MODEL_CRITICAL = "anthropic/claude-3-5-sonnet-20241022"

# Ollama server base URL (no auth required for local Ollama)
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Default temperatures per task type
_TEMP_GENERAL  = 0.3   # slight variation keeps conversation natural
_TEMP_CRITICAL = 0.0   # fully deterministic for classification / validation

# Timeout (seconds) for the Ollama health-check probe
_OLLAMA_HEALTH_TIMEOUT = 2

# Timeout (seconds) for a single Ollama inference call.
# Set high enough for slow local hardware; the health-check cache prevents
# cascading socket exhaustion if Ollama goes down between requests.
_OLLAMA_INFERENCE_TIMEOUT = 120

# After a failed health-check, skip re-probing for this many seconds so we
# don't spam connection attempts on every request when Ollama is down.
_OLLAMA_DOWN_COOLDOWN = 30

# Module-level health-check cache: (result: bool, checked_at: float)
_ollama_cache: tuple[bool, float] = (False, 0.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ollama_is_running() -> bool:
    """
    Return True if the Ollama server responds to a quick health-check.

    Result is cached for _OLLAMA_DOWN_COOLDOWN seconds after a *failure* so
    that a flapping or absent Ollama doesn't open a new socket on every call.
    A successful check is re-verified on the next call (no stale "up" cache).
    """
    global _ollama_cache
    cached_result, checked_at = _ollama_cache

    # If last check was a failure and the cooldown hasn't expired, skip probe
    if not cached_result and (time.monotonic() - checked_at) < _OLLAMA_DOWN_COOLDOWN:
        return False

    try:
        with urllib.request.urlopen(
            f"{_OLLAMA_BASE_URL}/api/tags",
            timeout=_OLLAMA_HEALTH_TIMEOUT,
        ):
            _ollama_cache = (True, time.monotonic())
            return True
    except (urllib.error.URLError, OSError, TimeoutError):
        _ollama_cache = (False, time.monotonic())
        return False


def _get_claude_llm(temperature: float) -> LLM:
    """Return a Claude 3.5 Sonnet LLM, raising EnvironmentError if no API key."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file (see .env.example)."
        )
    return LLM(
        model=MODEL_CRITICAL,
        temperature=temperature,
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_llm(
    task_type: TaskType | str = TaskType.GENERAL,
    temperature: float | None = None,
) -> LLM:
    """
    Return a configured CrewAI LLM for the given task type.

    Parameters
    ----------
    task_type : TaskType | str
        "general"  → Llama 3.2 via Ollama (no API key required).
                     Falls back to Claude 3.5 Sonnet if Ollama is unreachable
                     and ANTHROPIC_API_KEY is available.
        "critical" → Claude 3.5 Sonnet (ANTHROPIC_API_KEY required).
    temperature : float | None
        Override the default temperature for this task type.
        If None, uses the task-appropriate default (0.3 general / 0.0 critical).

    Returns
    -------
    crewai.LLM
        Ready-to-use LLM instance for a CrewAI Agent or Task.

    Raises
    ------
    ValueError
        If task_type is not a recognised TaskType value.
    EnvironmentError
        If ANTHROPIC_API_KEY is not set when using TaskType.CRITICAL.
    RuntimeError
        If TaskType.GENERAL is requested, Ollama is not reachable, and
        ANTHROPIC_API_KEY is also missing (no fallback available).
    """
    try:
        task = TaskType(task_type)
    except ValueError:
        valid = [t.value for t in TaskType]
        raise ValueError(
            f"Unknown task_type {task_type!r}. Valid options: {valid}"
        )

    if task == TaskType.GENERAL:
        temp = temperature if temperature is not None else _TEMP_GENERAL

        # Ollama is the primary provider for GENERAL tasks (local, free).
        # Only fall back to Claude when Ollama is not reachable.
        if _ollama_is_running():
            return LLM(
                model=MODEL_GENERAL,
                temperature=temp,
                base_url=_OLLAMA_BASE_URL,
                timeout=_OLLAMA_INFERENCE_TIMEOUT,
            )

        # Ollama is down — fall back to Claude if API key is available
        _logger.warning(
            "Ollama is not reachable at %s (model: %s). "
            "Falling back to Claude 3.5 Sonnet for GENERAL tasks. "
            "Start Ollama to restore local inference.",
            _OLLAMA_BASE_URL,
            _OLLAMA_MODEL_NAME,
        )
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"Ollama is not running at {_OLLAMA_BASE_URL} and "
                "ANTHROPIC_API_KEY is not set — no LLM available for "
                "GENERAL tasks. Either start Ollama (`ollama serve`) or "
                "add ANTHROPIC_API_KEY to your .env file."
            )
        return LLM(
            model=MODEL_CRITICAL,
            temperature=temp,
            api_key=api_key,
        )

    # CRITICAL — Claude 3.5 Sonnet via Anthropic
    return _get_claude_llm(
        temperature=temperature if temperature is not None else _TEMP_CRITICAL,
    )


def get_llm_strict(model: str, temperature: float) -> LLM:
    """
    Return an LLM pinned to exactly *model*, with NO provider fallback.

    get_llm() will silently substitute Claude for GENERAL tasks when Ollama
    is unreachable -- correct behaviour for a conversational agent, wrong
    behaviour for an evaluation run where every case must be answered by
    the same model or the run's per-case latency/cost/accuracy numbers
    describe a system that was never actually run end-to-end. This raises
    RuntimeError immediately instead, so callers (the eval harness) abort
    before any case runs rather than discovering the substitution later
    from a config-hash string that doesn't distinguish which model
    actually answered.

    Parameters
    ----------
    model : str
        Fully-qualified LiteLLM routing string, e.g. "ollama/llama3.2:1b"
        or "anthropic/claude-3-5-sonnet-20241022". No other providers are
        recognised.
    temperature : float
        Passed straight through to the LLM constructor.

    Raises
    ------
    RuntimeError
        If the pinned provider/model is not reachable or not configured.
    ValueError
        If *model* does not start with a recognised provider prefix.
    """
    if model.startswith("ollama/"):
        if not _ollama_is_running():
            raise RuntimeError(
                f"Pinned reranker model {model!r} requires Ollama at "
                f"{_OLLAMA_BASE_URL}, but it is not reachable. Aborting "
                "rather than silently substituting a different model."
            )
        ollama_model_name = model.split("/", 1)[1]
        try:
            with urllib.request.urlopen(
                f"{_OLLAMA_BASE_URL}/api/tags", timeout=_OLLAMA_HEALTH_TIMEOUT
            ) as resp:
                tags = json.loads(resp.read())
            pulled = {m.get("name", "") for m in tags.get("models", [])}
        except Exception as exc:
            raise RuntimeError(
                f"Could not verify pinned reranker model {model!r} is pulled "
                f"in Ollama: {exc}"
            ) from exc
        if ollama_model_name not in pulled and not any(
            p == ollama_model_name or p.startswith(ollama_model_name + ":")
            for p in pulled
        ):
            raise RuntimeError(
                f"Pinned reranker model {model!r} is not pulled in Ollama "
                f"(available: {sorted(pulled)}). Run `ollama pull "
                f"{ollama_model_name}` or choose a different model."
            )
        return LLM(
            model=model,
            temperature=temperature,
            base_url=_OLLAMA_BASE_URL,
            timeout=_OLLAMA_INFERENCE_TIMEOUT,
        )

    if model.startswith("anthropic/"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"Pinned reranker model {model!r} requires ANTHROPIC_API_KEY, "
                "which is not set. Aborting rather than silently substituting "
                "a different model."
            )
        return LLM(model=model, temperature=temperature, api_key=api_key)

    raise ValueError(
        f"get_llm_strict: unrecognised provider prefix in {model!r} "
        "(expected 'ollama/...' or 'anthropic/...')"
    )
