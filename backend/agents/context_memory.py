"""
backend/agents/context_memory.py

Redis-backed conversation context memory for the LFS survey.

Responsibilities
----------------
1. Storage / retrieval  — Persists complete session state (conversation
   history, FSM state, collected LFS fields) in Redis as a single JSON
   document per session.  TTL defaults to 24 hours.

2. Field tracking  — Exposes helpers to update collected fields and
   query which of the five required LFS fields are still missing, so the
   ConversationManager can avoid asking duplicate questions.

3. History management  — Provides an append-only log of every
   conversational turn (user + assistant) with timestamps and per-turn
   language tags.

4. Bilingual context summary  — A CrewAI agent (GPT-4o-mini) translates
   the structured session state into a compact, human-readable paragraph
   in both English and Arabic for consumption by downstream agents
   (ConversationManager, ValidationAgent, ISCOClassifier).

Redis key layout
----------------
One JSON document per session:
    lfs:session:{session_id}  →  SessionMemory (JSON)

All documents share a configurable TTL (default 24 h), refreshed on
every write.

Usage
-----
from backend.agents.context_memory import ContextMemory

cm = ContextMemory()

# Persist or update a session
cm.save_session(42, state="collecting_info", language="en",
                collected_fields={"employment_status": "employed"},
                history=[])

# Append a conversational turn
cm.append_turn(42, role="user", content="I work as a nurse.",
               detected_language="en")

# Check progress
print(cm.get_collected_fields(42))   # {"employment_status": "employed"}
print(cm.get_missing_fields(42))     # ["job_title", "industry", ...]

# Bilingual agent summary
summary = cm.get_summary(42, language="en")
print(summary.summary_en)
print(summary.summary_ar)
print(summary.missing_fields)
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

import redis as redis_lib
from crewai import Agent, Crew, Task
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from backend.llm import TaskType, get_llm

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KEY_PREFIX   = "lfs:session:"
_TTL_SECONDS  = 86_400      # 24 hours; override via ContextMemory(ttl=...)

# The five LFS fields the ConversationManager must collect.
_ALL_FIELDS: frozenset[str] = frozenset({
    "employment_status",
    "job_title",
    "industry",
    "hours_per_week",
    "employment_type",
})

# Bilingual display labels used in fallback summaries.
_FIELD_LABELS: dict[str, dict[str, str]] = {
    "employment_status": {"en": "Employment status",    "ar": "حالة التوظيف"},
    "job_title":         {"en": "Job title",            "ar": "المسمى الوظيفي"},
    "industry":          {"en": "Industry / sector",    "ar": "الصناعة / القطاع"},
    "hours_per_week":    {"en": "Hours per week",       "ar": "ساعات العمل الأسبوعية"},
    "employment_type":   {"en": "Employment type",      "ar": "نوع التوظيف"},
}

# State display labels for summaries.
_STATE_LABELS: dict[str, dict[str, str]] = {
    "greeting":       {"en": "greeting",           "ar": "الترحيب"},
    "collecting_info":{"en": "collecting data",    "ar": "جمع البيانات"},
    "clarifying":     {"en": "clarifying answers", "ar": "توضيح الإجابات"},
    "validating":     {"en": "validating answers", "ar": "التحقق من الإجابات"},
    "completing":     {"en": "completing survey",  "ar": "إتمام المسح"},
}


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class TurnRecord(BaseModel):
    """A single conversational turn stored in session memory."""

    role: str                                   # "user" | "assistant"
    content: str
    timestamp: str                              # ISO 8601 UTC
    detected_language: Optional[str] = None    # "en" | "ar" | "mixed" | None


class SessionMemory(BaseModel):
    """Complete session state persisted in Redis."""

    session_id: int
    language: str                               # "en" | "ar"
    state: str                                  # ConversationState value
    collected_fields: dict[str, str]            # field → raw value
    history: list[TurnRecord]
    turn_count: int                             # number of user turns
    created_at: str                             # ISO 8601 UTC
    last_updated: str                           # ISO 8601 UTC


class ContextSummary(BaseModel):
    """Bilingual session summary produced by ContextMemory.get_summary()."""

    session_id: int
    language: str                               # preferred language for the session
    state: str
    collected_fields: dict[str, str]
    missing_fields: list[str]                   # LFS fields not yet collected
    turn_count: int
    summary_en: str                             # human-readable summary in English
    summary_ar: str                             # human-readable summary in Arabic


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SUMMARY_INSTRUCTIONS = """\
You are a Labour Force Survey (LFS) session assistant. Your task is to produce \
a compact, machine-readable context summary of the current survey session so \
that other survey agents can quickly understand the respondent's progress.

Return ONLY a valid JSON object — no markdown fences, no extra text:
{
  "summary_en": "<two-sentence summary in English>",
  "summary_ar": "<two-sentence summary in Arabic>"
}

Guidelines for each summary:
  Sentence 1 — Describe what employment data has been collected so far.
               If nothing has been collected yet, write "No employment data \
collected yet."
  Sentence 2 — State which fields are still needed and the current FSM state.
               If all fields are collected, write "All required fields collected."
Arabic must be correct Modern Standard Arabic (فصحى).
"""


# ---------------------------------------------------------------------------
# ContextMemory
# ---------------------------------------------------------------------------

class ContextMemory:
    """
    Redis-backed session context memory with bilingual summary generation.

    Parameters
    ----------
    redis_url : str | None
        Full Redis URL (e.g. ``redis://localhost:6379``).
        Takes precedence over host/port when provided.
        Falls back to the ``REDIS_URL`` environment variable.
    host : str | None
        Redis hostname.  Falls back to ``REDIS_HOST`` env var → ``localhost``.
    port : int | None
        Redis port.  Falls back to ``REDIS_PORT`` env var → ``6379``.
    db : int
        Redis logical database index (default 0).
    ttl : int
        Time-to-live for each session key in seconds (default 86 400 = 24 h).
        Refreshed on every write.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: int = 0,
        ttl: int = _TTL_SECONDS,
    ) -> None:
        url = redis_url or os.getenv("REDIS_URL")
        if url:
            self._redis = redis_lib.Redis.from_url(url, decode_responses=True)
        else:
            _host = host or os.getenv("REDIS_HOST", "localhost")
            _port = int(port or os.getenv("REDIS_PORT", 6379))
            self._redis = redis_lib.Redis(
                host=_host, port=_port, db=db, decode_responses=True
            )
        self._ttl = ttl

        self._llm = get_llm(TaskType.GENERAL)   # GPT-4o-mini, temp 0.3
        self._agent = Agent(
            role="LFS Survey Context Summariser",
            goal=(
                "Produce concise, accurate bilingual summaries (English and "
                "Arabic) of a Labour Force Survey session's current state so "
                "that other survey agents can quickly understand respondent "
                "progress without reading the full conversation history."
            ),
            backstory=(
                "You are a multilingual survey coordinator at a national "
                "statistics office. You process hundreds of LFS sessions daily "
                "and excel at distilling session context into crisp, factual "
                "summaries — in both English and Arabic — that let your "
                "colleagues pick up any session instantly."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

    # ------------------------------------------------------------------
    # Core storage
    # ------------------------------------------------------------------

    def save_session(
        self,
        session_id: int,
        state: str,
        language: str,
        collected_fields: dict[str, str],
        history: list[dict | TurnRecord],
    ) -> SessionMemory:
        """
        Persist (or overwrite) the full session state in Redis.

        Parameters
        ----------
        session_id : int
        state : str
            ConversationState value string (e.g. ``"collecting_info"``).
        language : str
            Active survey language (``"en"`` or ``"ar"``).
        collected_fields : dict[str, str]
            Mapping of LFS field name → raw collected value.
        history : list[dict | TurnRecord]
            Sequence of past turns.  Plain dicts are coerced to TurnRecord.

        Returns
        -------
        SessionMemory
            The memory object that was stored.
        """
        existing = self.load_session(session_id)
        created_at = existing.created_at if existing else _now()

        turns = [
            r if isinstance(r, TurnRecord) else TurnRecord(**r)
            for r in history
        ]
        user_turns = sum(1 for t in turns if t.role == "user")

        mem = SessionMemory(
            session_id=session_id,
            language=language,
            state=state,
            collected_fields={k: str(v) for k, v in collected_fields.items()},
            history=turns,
            turn_count=user_turns,
            created_at=created_at,
            last_updated=_now(),
        )
        self._set(session_id, mem)
        return mem

    def load_session(self, session_id: int) -> Optional[SessionMemory]:
        """
        Retrieve session memory from Redis.

        Returns
        -------
        SessionMemory | None
            ``None`` when the key does not exist or has expired.
        """
        raw = self._redis.get(self._key(session_id))
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            return SessionMemory(**data)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def delete_session(self, session_id: int) -> bool:
        """
        Remove the session from Redis.

        Returns
        -------
        bool
            ``True`` if the key existed and was deleted; ``False`` otherwise.
        """
        return bool(self._redis.delete(self._key(session_id)))

    # ------------------------------------------------------------------
    # Incremental updates
    # ------------------------------------------------------------------

    def append_turn(
        self,
        session_id: int,
        role: str,
        content: str,
        detected_language: Optional[str] = None,
    ) -> TurnRecord:
        """
        Append one conversational turn to the session history.

        If no session exists yet, a minimal stub is created automatically.
        The TTL is refreshed on every call.

        Parameters
        ----------
        session_id : int
        role : str
            ``"user"`` or ``"assistant"``.
        content : str
            The message text.
        detected_language : str | None
            Language detected for this turn (``"en"``, ``"ar"``, ``"mixed"``).

        Returns
        -------
        TurnRecord
            The newly appended record.
        """
        mem = self._load_or_create(session_id)
        turn = TurnRecord(
            role=role,
            content=content,
            timestamp=_now(),
            detected_language=detected_language,
        )
        mem.history.append(turn)
        if role == "user":
            mem.turn_count += 1
        mem.last_updated = _now()
        self._set(session_id, mem)
        return turn

    def update_fields(
        self,
        session_id: int,
        new_fields: dict[str, str],
    ) -> dict[str, str]:
        """
        Merge *new_fields* into the session's collected_fields.

        Existing values are overwritten only when a new value is non-empty.
        Creates a stub session if none exists yet.

        Parameters
        ----------
        session_id : int
        new_fields : dict[str, str]
            Field-value pairs to merge.

        Returns
        -------
        dict[str, str]
            The complete collected_fields after the merge.
        """
        mem = self._load_or_create(session_id)
        for field, value in new_fields.items():
            if value:
                mem.collected_fields[field] = str(value)
        mem.last_updated = _now()
        self._set(session_id, mem)
        return dict(mem.collected_fields)

    # ------------------------------------------------------------------
    # Read-only queries
    # ------------------------------------------------------------------

    def get_collected_fields(self, session_id: int) -> dict[str, str]:
        """
        Return the fields collected so far for a session.

        Returns an empty dict when the session does not exist.
        """
        mem = self.load_session(session_id)
        return dict(mem.collected_fields) if mem else {}

    def get_missing_fields(self, session_id: int) -> list[str]:
        """
        Return the sorted list of required LFS fields not yet collected.

        Returns all five fields when the session does not exist.
        """
        collected = set(self.get_collected_fields(session_id).keys())
        return sorted(_ALL_FIELDS - collected)

    def get_history(
        self,
        session_id: int,
        last_n: Optional[int] = None,
    ) -> list[TurnRecord]:
        """
        Retrieve the conversation history for a session.

        Parameters
        ----------
        session_id : int
        last_n : int | None
            If provided, return only the most recent *last_n* turns.

        Returns
        -------
        list[TurnRecord]
            Empty when the session does not exist.
        """
        mem = self.load_session(session_id)
        if mem is None:
            return []
        history = mem.history
        if last_n is not None:
            history = history[-last_n:]
        return list(history)

    # ------------------------------------------------------------------
    # Bilingual summary (CrewAI agent)
    # ------------------------------------------------------------------

    def get_summary(
        self,
        session_id: int,
        language: str = "en",
    ) -> ContextSummary:
        """
        Generate a bilingual context summary for the given session.

        The summary is produced by a GPT-4o-mini CrewAI agent and is
        suitable for passing as context to other survey agents.

        If no session exists, or if the LLM call fails, a rule-based
        fallback summary is used instead.

        Parameters
        ----------
        session_id : int
        language : str
            Preferred language for the session (used in the summary preamble).

        Returns
        -------
        ContextSummary
        """
        mem    = self.load_session(session_id)
        if mem is None:
            mem = SessionMemory(
                session_id=session_id,
                language=language,
                state="greeting",
                collected_fields={},
                history=[],
                turn_count=0,
                created_at=_now(),
                last_updated=_now(),
            )

        missing = sorted(_ALL_FIELDS - set(mem.collected_fields.keys()))
        summary_en, summary_ar = self._generate_summary(mem, missing)

        return ContextSummary(
            session_id=session_id,
            language=mem.language,
            state=mem.state,
            collected_fields=dict(mem.collected_fields),
            missing_fields=missing,
            turn_count=mem.turn_count,
            summary_en=summary_en,
            summary_ar=summary_ar,
        )

    # ------------------------------------------------------------------
    # Summary generation (private)
    # ------------------------------------------------------------------

    def _generate_summary(
        self,
        mem: SessionMemory,
        missing: list[str],
    ) -> tuple[str, str]:
        """
        Ask GPT-4o-mini to summarise the session; fall back to rule-based text.
        """
        fields_block = (
            "\n".join(f"  {k}: {v}" for k, v in sorted(mem.collected_fields.items()))
            or "  (none)"
        )
        missing_block = (
            ", ".join(_FIELD_LABELS[f]["en"] for f in missing)
            or "none — all fields collected"
        )
        state_label = _STATE_LABELS.get(mem.state, {}).get("en", mem.state)

        task = Task(
            description=(
                f"{_SUMMARY_INSTRUCTIONS}\n\n"
                f"Session ID      : {mem.session_id}\n"
                f"Language        : {mem.language}\n"
                f"FSM state       : {state_label}\n"
                f"Turn count      : {mem.turn_count}\n"
                f"Collected fields:\n{fields_block}\n"
                f"Missing fields  : {missing_block}"
            ),
            expected_output=(
                'JSON: {"summary_en": "<string>", "summary_ar": "<string>"}'
            ),
            agent=self._agent,
        )

        crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
        raw  = str(crew.kickoff()).strip()
        return self._parse_summary_response(raw, mem, missing)

    def _parse_summary_response(
        self,
        raw: str,
        mem: SessionMemory,
        missing: list[str],
    ) -> tuple[str, str]:
        """
        Parse the LLM JSON and return (summary_en, summary_ar).

        Falls back to rule-based summaries on any parse failure.
        """
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()

        data: dict = {}
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        en = str(data.get("summary_en", "")).strip()
        ar = str(data.get("summary_ar", "")).strip()

        if not en or not ar:
            en, ar = self._build_fallback_summary(mem, missing)

        return en, ar

    # ------------------------------------------------------------------
    # Rule-based fallback summary
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fallback_summary(
        mem: SessionMemory,
        missing: list[str],
    ) -> tuple[str, str]:
        """
        Generate deterministic bilingual summaries without calling the LLM.

        Used when the LLM response is absent or unparseable.
        """
        state_en = _STATE_LABELS.get(mem.state, {}).get("en", mem.state)
        state_ar = _STATE_LABELS.get(mem.state, {}).get("ar", mem.state)

        if mem.collected_fields:
            parts_en = [
                f"{_FIELD_LABELS.get(k, {}).get('en', k)}: {v}"
                for k, v in sorted(mem.collected_fields.items())
            ]
            parts_ar = [
                f"{_FIELD_LABELS.get(k, {}).get('ar', k)}: {v}"
                for k, v in sorted(mem.collected_fields.items())
            ]
            s1_en = "Collected so far — " + "; ".join(parts_en) + "."
            s1_ar = "تم جمعه حتى الآن — " + "؛ ".join(parts_ar) + "."
        else:
            s1_en = "No employment data collected yet."
            s1_ar = "لم يتم جمع أي بيانات توظيف حتى الآن."

        if missing:
            miss_en = ", ".join(_FIELD_LABELS.get(f, {}).get("en", f) for f in missing)
            miss_ar = "، ".join(_FIELD_LABELS.get(f, {}).get("ar", f) for f in missing)
            s2_en = (
                f"Still needed: {miss_en}. "
                f"Current state: {state_en} ({mem.turn_count} turn(s))."
            )
            s2_ar = (
                f"لا يزال مطلوبًا: {miss_ar}. "
                f"الحالة الحالية: {state_ar} ({mem.turn_count} دورة)."
            )
        else:
            s2_en = (
                f"All required fields collected. "
                f"Current state: {state_en} ({mem.turn_count} turn(s))."
            )
            s2_ar = (
                f"تم جمع جميع الحقول المطلوبة. "
                f"الحالة الحالية: {state_ar} ({mem.turn_count} دورة)."
            )

        return f"{s1_en} {s2_en}", f"{s1_ar} {s2_ar}"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _key(self, session_id: int) -> str:
        """Return the Redis key for a session."""
        return f"{_KEY_PREFIX}{session_id}"

    def _set(self, session_id: int, mem: SessionMemory) -> None:
        """Serialise and write a SessionMemory to Redis, refreshing the TTL."""
        self._redis.set(
            self._key(session_id),
            mem.model_dump_json(),
            ex=self._ttl,
        )

    def _load_or_create(
        self,
        session_id: int,
        language: str = "en",
        state: str = "greeting",
    ) -> SessionMemory:
        """
        Load an existing session or create a minimal stub if absent.

        The stub is NOT persisted; the caller must call _set() after
        making changes.
        """
        mem = self.load_session(session_id)
        if mem is None:
            mem = SessionMemory(
                session_id=session_id,
                language=language,
                state=state,
                collected_fields={},
                history=[],
                turn_count=0,
                created_at=_now(),
                last_updated=_now(),
            )
        return mem


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[ContextMemory] = None


def get_context_memory() -> ContextMemory:
    """
    Return the module-level ContextMemory singleton.

    Creates the instance on the first call (connects to Redis, loads the
    LLM client).  Subsequent calls return the cached instance.
    """
    global _instance
    if _instance is None:
        _instance = ContextMemory()
    return _instance
