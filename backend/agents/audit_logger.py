"""
backend/agents/audit_logger.py

GDPR-compliant audit trail for the LFS survey system.

Responsibilities
----------------
1. Interaction logging  — Persists every notable survey event (session start,
   message sent, session completed, auth events, …) to the ``audit_logs``
   PostgreSQL table.

2. Data-access tracking — Records every read/write/delete/export of personal
   data in ``data_access_logs`` to satisfy GDPR Art. 30 Record-of-Processing
   obligations.

3. Agent-decision logging — Stores each AI-agent decision (ISCO classification,
   NER, validation, …) in ``agent_decision_logs`` for transparency and
   human-review readiness.

4. Audit-trail reports — A CrewAI agent (Claude 3.5 Sonnet) generates bilingual
   (English + Arabic) compliance summaries from aggregated activity statistics.

5. Data-retention enforcement — ``purge_expired()`` hard-deletes rows whose
   retention window has passed (24 h granularity; default 90 days).  A
   ``dry_run`` mode previews deletion counts without committing.

Table layout
------------
- ``audit_logs``         — one row per system event
- ``data_access_logs``   — one row per personal-data access, with per-row TTL
- ``agent_decision_logs``— one row per AI-agent decision

Usage
-----
from backend.agents.audit_logger import AuditLogger, EventType, AccessType

al = AuditLogger()

al.log_interaction(
    event_type=EventType.SESSION_STARTED,
    description="Survey session created.",
    session_id=42,
    user_id=7,
)

al.log_data_access(
    user_id=7,
    resource_type="survey_session",
    resource_id=42,
    access_type=AccessType.READ,
    purpose="Labour Force Survey data collection under GDPR Art. 6(1)(e).",
)

al.log_agent_decision(
    agent_name="ISCOClassifier",
    decision_type=AgentDecisionType.ISCO_CLASSIFICATION,
    input_summary="Job title: 'software engineer'",
    output_summary="ISCO-08 2512 (confidence 0.93)",
    confidence=0.93,
    session_id=42,
)

report = al.generate_report(session_id=42, language="en")
print(report.report_en)

result = al.purge_expired(dry_run=True)
print(result.deleted_audit_logs)   # preview count — nothing deleted yet
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Callable, Optional

from crewai import Agent, Crew, Task
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database.connection import SessionLocal
from backend.database.models import AgentDecisionLog, AuditLog, DataAccessLog
from backend.llm import TaskType, get_llm

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RETENTION_DAYS = 90


class EventType:
    """Predefined event-type tokens for ``log_interaction``."""

    SESSION_STARTED          = "session_started"
    SESSION_COMPLETED        = "session_completed"
    SESSION_DELETED          = "session_deleted"
    MESSAGE_SENT             = "message_sent"
    RESPONSE_SUBMITTED       = "response_submitted"
    AUTH_OTP_REQUESTED       = "auth_otp_requested"
    AUTH_OTP_VERIFIED        = "auth_otp_verified"
    AUTH_TOKEN_ISSUED        = "auth_token_issued"
    DATA_EXPORT_REQUESTED    = "data_export_requested"
    DATA_DELETION_REQUESTED  = "data_deletion_requested"
    PURGE_EXECUTED           = "purge_executed"


class AccessType:
    """GDPR-relevant data-access categories for ``log_data_access``."""

    READ      = "read"
    WRITE     = "write"
    DELETE    = "delete"
    EXPORT    = "export"
    ANONYMIZE = "anonymize"


class AgentDecisionType:
    """Decision-type tokens for ``log_agent_decision``."""

    ISCO_CLASSIFICATION = "isco_classification"
    LANGUAGE_DETECTION  = "language_detection"
    VALIDATION          = "validation"
    CONTEXT_SUMMARY     = "context_summary"
    RAG_RETRIEVAL       = "rag_retrieval"
    NER_EXTRACTION      = "ner_extraction"


# ---------------------------------------------------------------------------
# Output models (Pydantic)
# ---------------------------------------------------------------------------

class AuditEntry(BaseModel):
    """Serialisable view of one ``AuditLog`` row."""

    id:            int
    session_id:    Optional[int]  = None
    user_id:       Optional[int]  = None
    event_type:    str
    actor:         Optional[str]  = None
    description:   str
    ip_address:    Optional[str]  = None
    metadata_json: Optional[str]  = None
    timestamp:     str                      # ISO 8601 UTC


class DataAccessEntry(BaseModel):
    """Serialisable view of one ``DataAccessLog`` row."""

    id:            int
    user_id:       int
    accessor_id:   Optional[int]  = None
    resource_type: str
    resource_id:   Optional[int]  = None
    access_type:   str
    purpose:       str
    ip_address:    Optional[str]  = None
    timestamp:     str
    retained_until: str


class AgentDecisionEntry(BaseModel):
    """Serialisable view of one ``AgentDecisionLog`` row."""

    id:             int
    session_id:     Optional[int]   = None
    agent_name:     str
    decision_type:  str
    input_summary:  str
    output_summary: str
    confidence:     Optional[float] = None
    reasoning:      Optional[str]   = None
    duration_ms:    Optional[int]   = None
    timestamp:      str


class AuditReport(BaseModel):
    """Bilingual compliance summary produced by ``generate_report``."""

    period_start:          str
    period_end:            str
    session_id:            Optional[int]        = None
    user_id:               Optional[int]        = None
    total_audit_events:    int
    total_data_accesses:   int
    total_agent_decisions: int
    event_counts:          dict[str, int]       = Field(default_factory=dict)
    access_type_counts:    dict[str, int]       = Field(default_factory=dict)
    agent_counts:          dict[str, int]       = Field(default_factory=dict)
    report_en:             str
    report_ar:             str
    generated_at:          str


class PurgeResult(BaseModel):
    """Summary of a ``purge_expired`` run."""

    deleted_audit_logs:     int
    deleted_access_logs:    int
    deleted_decision_logs:  int
    dry_run:                bool
    retention_days:         int
    cutoff_timestamp:       str     # rows older than this were targeted
    purge_timestamp:        str


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_REPORT_INSTRUCTIONS = """\
You are a GDPR Data Protection Officer at a national statistics office. Your \
task is to produce a concise, factual compliance summary of LFS survey audit \
activity so that data-governance stakeholders can quickly assess system health.

Return ONLY a valid JSON object — no markdown fences, no extra text:
{
  "report_en": "<three to four sentence compliance report in English>",
  "report_ar": "<three to four sentence compliance report in Arabic>"
}

Each report must cover:
  1. Volume  — total survey interactions, data accesses, and agent decisions.
  2. Breakdown — most frequent event types and data-access patterns observed.
  3. Compliance — whether activity is consistent with stated GDPR Art. 6(1)(e) purposes.
  4. Recommendation — note any patterns that may warrant review; otherwise confirm clean status.

Arabic must be correct Modern Standard Arabic (فصحى).
Be factual. Do not invent data not present in the statistics provided.
"""


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class AuditLogger:
    """
    PostgreSQL-backed audit trail with bilingual GDPR compliance reports.

    Parameters
    ----------
    session_factory : callable | None
        Zero-argument callable that returns a SQLAlchemy ``Session``.
        Defaults to ``SessionLocal`` from ``backend.database.connection``.
        Inject a test factory to avoid touching PostgreSQL in unit tests.
    retention_days : int
        Default number of days to keep audit records (default 90).
        ``AuditLog`` and ``AgentDecisionLog`` rows are purged when
        ``timestamp < now - retention_days``.
        ``DataAccessLog`` rows are purged when ``retained_until < now``
        (their TTL is set at creation time to ``now + retention_days``).
    """

    def __init__(
        self,
        session_factory: Optional[Callable[[], Session]] = None,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
    ) -> None:
        self._sf             = session_factory or SessionLocal
        self._retention_days = retention_days
        self._llm            = get_llm(TaskType.CRITICAL)   # Claude 3.5 Sonnet

        self._agent = Agent(
            role="GDPR Audit Compliance Analyst",
            goal=(
                "Produce accurate, bilingual (English and Arabic) compliance "
                "reports from LFS survey audit statistics, enabling data-governance "
                "stakeholders to assess system health and regulatory adherence."
            ),
            backstory=(
                "You are a Data Protection Officer at a national statistics office. "
                "You review audit logs daily, assess GDPR compliance, and draft "
                "concise reports in both English and Arabic for senior management "
                "and regulatory bodies."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

    # ------------------------------------------------------------------
    # Public API — logging
    # ------------------------------------------------------------------

    def log_interaction(
        self,
        event_type: str,
        description: str,
        session_id: Optional[int]  = None,
        user_id:    Optional[int]  = None,
        actor:      Optional[str]  = None,
        ip_address: Optional[str]  = None,
        metadata:   Optional[dict] = None,
    ) -> AuditEntry:
        """
        Persist a survey interaction event to ``audit_logs``.

        Parameters
        ----------
        event_type : str
            Use an ``EventType`` constant or a custom string token.
        description : str
            Human-readable description of what happened.
        session_id : int | None
            Associated survey session, if any.
        user_id : int | None
            Acting user, if known.
        actor : str | None
            Actor label, e.g. ``"user"``, ``"agent:ISCOClassifier"``, ``"system"``.
        ip_address : str | None
            Client IP for access-control audit trails.
        metadata : dict | None
            Optional extra key-value data serialised as JSON.
        """
        now = datetime.utcnow()
        row = AuditLog(
            session_id=session_id,
            user_id=user_id,
            event_type=event_type,
            actor=actor,
            description=description,
            ip_address=ip_address,
            metadata_json=json.dumps(metadata) if metadata else None,
            timestamp=now,
        )
        self._write(row)
        return self._to_audit_entry(row)

    def log_data_access(
        self,
        user_id:       int,
        resource_type: str,
        access_type:   str,
        purpose:       str,
        resource_id:   Optional[int] = None,
        accessor_id:   Optional[int] = None,
        ip_address:    Optional[str] = None,
    ) -> DataAccessEntry:
        """
        Persist a personal-data access record to ``data_access_logs``.

        Parameters
        ----------
        user_id : int
            The data subject (whose data was accessed).
        resource_type : str
            Type of resource, e.g. ``"survey_session"``, ``"user_profile"``.
        access_type : str
            Use an ``AccessType`` constant.
        purpose : str
            GDPR lawful basis or declared purpose for the access.
        resource_id : int | None
            Primary key of the accessed record.
        accessor_id : int | None
            User who performed the access (if different from the data subject).
        ip_address : str | None
            Client IP.
        """
        now            = datetime.utcnow()
        retained_until = now + timedelta(days=self._retention_days)
        row = DataAccessLog(
            user_id=user_id,
            accessor_id=accessor_id,
            resource_type=resource_type,
            resource_id=resource_id,
            access_type=access_type,
            purpose=purpose,
            ip_address=ip_address,
            timestamp=now,
            retained_until=retained_until,
        )
        self._write(row)
        return self._to_access_entry(row)

    def log_agent_decision(
        self,
        agent_name:     str,
        decision_type:  str,
        input_summary:  str,
        output_summary: str,
        confidence:     Optional[float] = None,
        reasoning:      Optional[str]   = None,
        session_id:     Optional[int]   = None,
        duration_ms:    Optional[int]   = None,
    ) -> AgentDecisionEntry:
        """
        Persist an AI-agent decision to ``agent_decision_logs``.

        Parameters
        ----------
        agent_name : str
            Class or role name of the agent, e.g. ``"ISCOClassifier"``.
        decision_type : str
            Use an ``AgentDecisionType`` constant.
        input_summary : str
            Brief summary of the agent's input (avoid raw PII).
        output_summary : str
            Brief summary of the agent's output.
        confidence : float | None
            Decision confidence score [0, 1].
        reasoning : str | None
            Free-text explanation from the agent, if available.
        session_id : int | None
            Associated survey session.
        duration_ms : int | None
            Wall-clock execution time in milliseconds.
        """
        row = AgentDecisionLog(
            session_id=session_id,
            agent_name=agent_name,
            decision_type=decision_type,
            input_summary=input_summary,
            output_summary=output_summary,
            confidence=confidence,
            reasoning=reasoning,
            duration_ms=duration_ms,
            timestamp=datetime.utcnow(),
        )
        self._write(row)
        return self._to_decision_entry(row)

    # ------------------------------------------------------------------
    # Public API — reporting
    # ------------------------------------------------------------------

    def generate_report(
        self,
        session_id: Optional[int]      = None,
        user_id:    Optional[int]      = None,
        from_dt:    Optional[datetime] = None,
        to_dt:      Optional[datetime] = None,
        language:   str                = "en",
    ) -> AuditReport:
        """
        Generate a bilingual compliance summary for the specified period.

        Parameters
        ----------
        session_id : int | None
            Filter to events related to a single survey session.
        user_id : int | None
            Filter to events related to a single user.
        from_dt : datetime | None
            Period start (defaults to 30 days ago).
        to_dt : datetime | None
            Period end (defaults to now).
        language : str
            Preferred language hint passed to the report agent (``"en"``/``"ar"``).

        Returns
        -------
        AuditReport
            Aggregated statistics and bilingual prose summary.
        """
        if from_dt is None:
            from_dt = datetime.utcnow() - timedelta(days=30)
        if to_dt is None:
            to_dt = datetime.utcnow()

        # ── Collect statistics from DB ────────────────────────────────────
        stats = self._collect_stats(session_id, user_id, from_dt, to_dt)

        # ── Ask Claude for bilingual prose (outside the DB session) ───────
        report_en, report_ar = self._generate_report_text(stats)

        return AuditReport(
            period_start=from_dt.isoformat(),
            period_end=to_dt.isoformat(),
            session_id=session_id,
            user_id=user_id,
            total_audit_events=stats["total_audit_events"],
            total_data_accesses=stats["total_data_accesses"],
            total_agent_decisions=stats["total_agent_decisions"],
            event_counts=stats["event_counts"],
            access_type_counts=stats["access_type_counts"],
            agent_counts=stats["agent_counts"],
            report_en=report_en,
            report_ar=report_ar,
            generated_at=datetime.utcnow().isoformat(),
        )

    # ------------------------------------------------------------------
    # Public API — purge
    # ------------------------------------------------------------------

    def purge_expired(self, dry_run: bool = False) -> PurgeResult:
        """
        Delete audit records whose retention window has passed.

        Deletion criteria
        -----------------
        * ``AuditLog``        — ``timestamp < now - retention_days``
        * ``AgentDecisionLog``— ``timestamp < now - retention_days``
        * ``DataAccessLog``   — ``retained_until < now``

        Parameters
        ----------
        dry_run : bool
            When ``True``, count matching rows but do **not** delete them.

        Returns
        -------
        PurgeResult
            Counts of records deleted (or that would be deleted on dry_run).
        """
        now    = datetime.utcnow()
        cutoff = now - timedelta(days=self._retention_days)

        db = self._sf()
        try:
            q_audit    = db.query(AuditLog).filter(AuditLog.timestamp < cutoff)
            q_access   = db.query(DataAccessLog).filter(DataAccessLog.retained_until < now)
            q_decision = db.query(AgentDecisionLog).filter(AgentDecisionLog.timestamp < cutoff)

            n_audit    = q_audit.count()
            n_access   = q_access.count()
            n_decision = q_decision.count()

            if not dry_run:
                q_audit.delete(synchronize_session=False)
                q_access.delete(synchronize_session=False)
                q_decision.delete(synchronize_session=False)
                db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

        # Log the purge event (after the session is closed to avoid mixing)
        if not dry_run and (n_audit + n_access + n_decision) > 0:
            self.log_interaction(
                event_type=EventType.PURGE_EXECUTED,
                actor="system",
                description=(
                    f"Data retention purge: {n_audit} audit logs, "
                    f"{n_access} access logs, {n_decision} decision logs deleted."
                ),
                metadata={
                    "deleted_audit_logs":    n_audit,
                    "deleted_access_logs":   n_access,
                    "deleted_decision_logs": n_decision,
                    "retention_days":        self._retention_days,
                    "cutoff":                cutoff.isoformat(),
                },
            )

        return PurgeResult(
            deleted_audit_logs=n_audit,
            deleted_access_logs=n_access,
            deleted_decision_logs=n_decision,
            dry_run=dry_run,
            retention_days=self._retention_days,
            cutoff_timestamp=cutoff.isoformat(),
            purge_timestamp=now.isoformat(),
        )

    # ------------------------------------------------------------------
    # Public API — GDPR subject access
    # ------------------------------------------------------------------

    def get_user_data_accesses(self, user_id: int) -> list[DataAccessEntry]:
        """
        Return all ``DataAccessLog`` rows for a data subject (GDPR Art. 15).

        Results are ordered newest-first.
        """
        db = self._sf()
        try:
            rows = (
                db.query(DataAccessLog)
                .filter(DataAccessLog.user_id == user_id)
                .order_by(DataAccessLog.timestamp.desc())
                .all()
            )
            return [self._to_access_entry(r) for r in rows]
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Private — DB helpers
    # ------------------------------------------------------------------

    def _write(self, row) -> None:
        """Add a single ORM row, commit, refresh (to populate auto-generated id)."""
        db = self._sf()
        try:
            db.add(row)
            db.commit()
            db.refresh(row)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _collect_stats(
        self,
        session_id: Optional[int],
        user_id:    Optional[int],
        from_dt:    datetime,
        to_dt:      datetime,
    ) -> dict:
        """Query all three audit tables and return aggregated statistics."""
        db = self._sf()
        try:
            # AuditLog
            q_audit = db.query(AuditLog).filter(
                AuditLog.timestamp >= from_dt,
                AuditLog.timestamp <= to_dt,
            )
            if session_id is not None:
                q_audit = q_audit.filter(AuditLog.session_id == session_id)
            if user_id is not None:
                q_audit = q_audit.filter(AuditLog.user_id == user_id)
            audit_rows = q_audit.all()

            # DataAccessLog
            q_access = db.query(DataAccessLog).filter(
                DataAccessLog.timestamp >= from_dt,
                DataAccessLog.timestamp <= to_dt,
            )
            if user_id is not None:
                q_access = q_access.filter(DataAccessLog.user_id == user_id)
            access_rows = q_access.all()

            # AgentDecisionLog
            q_decision = db.query(AgentDecisionLog).filter(
                AgentDecisionLog.timestamp >= from_dt,
                AgentDecisionLog.timestamp <= to_dt,
            )
            if session_id is not None:
                q_decision = q_decision.filter(AgentDecisionLog.session_id == session_id)
            decision_rows = q_decision.all()

        finally:
            db.close()

        event_counts: dict[str, int] = {}
        for r in audit_rows:
            event_counts[r.event_type] = event_counts.get(r.event_type, 0) + 1

        access_type_counts: dict[str, int] = {}
        for r in access_rows:
            access_type_counts[r.access_type] = access_type_counts.get(r.access_type, 0) + 1

        agent_counts: dict[str, int] = {}
        for r in decision_rows:
            agent_counts[r.agent_name] = agent_counts.get(r.agent_name, 0) + 1

        return {
            "total_audit_events":    len(audit_rows),
            "total_data_accesses":   len(access_rows),
            "total_agent_decisions": len(decision_rows),
            "event_counts":          event_counts,
            "access_type_counts":    access_type_counts,
            "agent_counts":          agent_counts,
        }

    # ------------------------------------------------------------------
    # Private — report generation
    # ------------------------------------------------------------------

    def _generate_report_text(self, stats: dict) -> tuple[str, str]:
        """Ask Claude 3.5 Sonnet to summarise *stats*; fall back to rule-based text."""
        events_block = (
            "\n".join(f"  {k}: {v}" for k, v in sorted(stats["event_counts"].items()))
            or "  (none)"
        )
        access_block = (
            "\n".join(f"  {k}: {v}" for k, v in sorted(stats["access_type_counts"].items()))
            or "  (none)"
        )
        agents_block = (
            "\n".join(f"  {k}: {v}" for k, v in sorted(stats["agent_counts"].items()))
            or "  (none)"
        )

        task = Task(
            description=(
                f"{_REPORT_INSTRUCTIONS}\n\n"
                f"Total survey interactions  : {stats['total_audit_events']}\n"
                f"Total data-access events   : {stats['total_data_accesses']}\n"
                f"Total agent decisions       : {stats['total_agent_decisions']}\n"
                f"Event-type breakdown:\n{events_block}\n"
                f"Data-access breakdown:\n{access_block}\n"
                f"Agent-decision breakdown:\n{agents_block}"
            ),
            expected_output='JSON: {"report_en": "<string>", "report_ar": "<string>"}',
            agent=self._agent,
        )

        crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
        raw  = str(crew.kickoff()).strip()
        return self._parse_report_response(raw, stats)

    def _parse_report_response(self, raw: str, stats: dict) -> tuple[str, str]:
        """Parse the LLM JSON and return (report_en, report_ar)."""
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

        en = str(data.get("report_en", "")).strip()
        ar = str(data.get("report_ar", "")).strip()

        if not en or not ar:
            en, ar = self._build_fallback_report(stats)

        return en, ar

    @staticmethod
    def _build_fallback_report(stats: dict) -> tuple[str, str]:
        """Produce deterministic bilingual text without calling the LLM."""
        n_audit    = stats["total_audit_events"]
        n_access   = stats["total_data_accesses"]
        n_decision = stats["total_agent_decisions"]

        if n_audit == 0 and n_access == 0 and n_decision == 0:
            en = (
                "Audit period: No survey activity recorded. "
                "No data-access events or agent decisions were logged. "
                "System was idle or audit logging was not active during this period."
            )
            ar = (
                "فترة التدقيق: لم يُسجَّل أي نشاط لمسح القوى العاملة. "
                "لم تُسجَّل أي أحداث وصول إلى البيانات أو قرارات الوكلاء. "
                "كان النظام في وضع الخمول أو أن تسجيل التدقيق لم يكن نشطًا خلال هذه الفترة."
            )
        else:
            top_events = sorted(stats["event_counts"].items(), key=lambda x: -x[1])[:3]
            top_str_en = ", ".join(f"{k} ({v})" for k, v in top_events) or "none"

            en = (
                f"Audit summary: {n_audit} survey interaction(s), "
                f"{n_access} data-access event(s), and {n_decision} agent decision(s) recorded. "
                f"Most frequent event types: {top_str_en}. "
                "All logged activities appear consistent with declared LFS survey purposes "
                "under GDPR Article 6(1)(e)."
            )
            ar = (
                f"ملخص التدقيق: تم تسجيل {n_audit} تفاعل مسح، "
                f"و{n_access} حدث وصول إلى البيانات، و{n_decision} قرار وكيل. "
                "تبدو جميع الأنشطة المسجلة متوافقة مع أغراض مسح القوى العاملة المُعلنة "
                "بموجب المادة 6(1)(ه) من اللائحة العامة لحماية البيانات."
            )

        return en, ar

    # ------------------------------------------------------------------
    # Private — ORM-to-Pydantic converters
    # ------------------------------------------------------------------

    @staticmethod
    def _to_audit_entry(row: AuditLog) -> AuditEntry:
        return AuditEntry(
            id=row.id,
            session_id=row.session_id,
            user_id=row.user_id,
            event_type=row.event_type,
            actor=row.actor,
            description=row.description,
            ip_address=row.ip_address,
            metadata_json=row.metadata_json,
            timestamp=row.timestamp.isoformat(),
        )

    @staticmethod
    def _to_access_entry(row: DataAccessLog) -> DataAccessEntry:
        return DataAccessEntry(
            id=row.id,
            user_id=row.user_id,
            accessor_id=row.accessor_id,
            resource_type=row.resource_type,
            resource_id=row.resource_id,
            access_type=row.access_type,
            purpose=row.purpose,
            ip_address=row.ip_address,
            timestamp=row.timestamp.isoformat(),
            retained_until=row.retained_until.isoformat(),
        )

    @staticmethod
    def _to_decision_entry(row: AgentDecisionLog) -> AgentDecisionEntry:
        return AgentDecisionEntry(
            id=row.id,
            session_id=row.session_id,
            agent_name=row.agent_name,
            decision_type=row.decision_type,
            input_summary=row.input_summary,
            output_summary=row.output_summary,
            confidence=row.confidence,
            reasoning=row.reasoning,
            duration_ms=row.duration_ms,
            timestamp=row.timestamp.isoformat(),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """
    Return the module-level ``AuditLogger`` singleton.

    Creates the instance on the first call (connects to PostgreSQL, loads the
    LLM client).  Subsequent calls return the cached instance.
    """
    global _instance
    if _instance is None:
        _instance = AuditLogger()
    return _instance
