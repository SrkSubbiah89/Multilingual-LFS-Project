"""
backend/llm/llm_client.py

Shared LLM client factory for the LFS project.

Task types
----------
"general"  → GPT-4o-mini (OpenAI)
             Fast and cost-effective; used by conversational agents
             (ConversationManager, LanguageProcessor NER).

"critical" → Claude 3.5 Sonnet (Anthropic)
             High-accuracy; used for tasks where correctness is essential:
             employment-status classification, answer validation,
             code-switch disambiguation.

Usage
-----
from backend.llm.llm_client import get_llm, TaskType

llm = get_llm(TaskType.GENERAL)          # GPT-4o-mini, temp 0.3
llm = get_llm(TaskType.CRITICAL)         # Claude 3.5 Sonnet, temp 0.0
llm = get_llm("general", temperature=0)  # override temperature
"""

from __future__ import annotations

import os
from enum import Enum

from crewai import LLM
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Task type enum
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    """Semantic task category that determines which model is selected."""
    GENERAL  = "general"   # GPT-4o-mini  — conversational agents, NER
    CRITICAL = "critical"  # Claude 3.5 Sonnet — classification, validation


# ---------------------------------------------------------------------------
# Model identifiers (LiteLLM routing strings used by CrewAI)
# ---------------------------------------------------------------------------

MODEL_GENERAL  = "gpt-4o-mini"
MODEL_CRITICAL = "anthropic/claude-3-5-sonnet-20241022"

# Default temperatures per task type
_TEMP_GENERAL  = 0.3   # slight variation keeps conversation natural
_TEMP_CRITICAL = 0.0   # fully deterministic for classification / validation


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
        "general"  → GPT-4o-mini (OPENAI_API_KEY)
        "critical" → Claude 3.5 Sonnet (ANTHROPIC_API_KEY)
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
        If the required API key environment variable is not set.
    """
    try:
        task = TaskType(task_type)
    except ValueError:
        valid = [t.value for t in TaskType]
        raise ValueError(
            f"Unknown task_type {task_type!r}. Valid options: {valid}"
        )

    if task == TaskType.GENERAL:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file (see .env.example)."
            )
        return LLM(
            model=MODEL_GENERAL,
            temperature=temperature if temperature is not None else _TEMP_GENERAL,
            api_key=api_key,
        )

    # CRITICAL — Claude 3.5 Sonnet via Anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file (see .env.example)."
        )
    return LLM(
        model=MODEL_CRITICAL,
        temperature=temperature if temperature is not None else _TEMP_CRITICAL,
        api_key=api_key,
    )
