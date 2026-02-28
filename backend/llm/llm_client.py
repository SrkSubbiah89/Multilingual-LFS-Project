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

import logging
import os
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ollama_is_running() -> bool:
    """
    Return True if the Ollama server responds to a quick health-check.

    Uses the /api/tags endpoint which is always available in Ollama ≥ 0.1.
    Times out after _OLLAMA_HEALTH_TIMEOUT seconds to keep startup fast.
    """
    try:
        with urllib.request.urlopen(
            f"{_OLLAMA_BASE_URL}/api/tags",
            timeout=_OLLAMA_HEALTH_TIMEOUT,
        ):
            return True
    except (urllib.error.URLError, OSError, TimeoutError):
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

        if _ollama_is_running():
            return LLM(
                model=MODEL_GENERAL,
                temperature=temp,
                base_url=_OLLAMA_BASE_URL,
            )

        # Ollama is not available — attempt graceful fallback
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
