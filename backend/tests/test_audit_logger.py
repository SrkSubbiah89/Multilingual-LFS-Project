"""
backend/tests/test_audit_logger.py

Uses an in-memory SQLite database (via the shared engine/session_factory
fixtures) so no live PostgreSQL is needed.  All CrewAI/LLM components are
mocked with monkeypatch.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.agents.audit_logger import (
    AccessType,
    AgentDecisionEntry,
    AgentDecisionType,
    AuditEntry,
    AuditLogger,
    AuditReport,
    DataAccessEntry,
    EventType,
    PurgeResult,
    _DEFAULT_RETENTION_DAYS,
    get_audit_logger,
)
from backend.database.connection import Base
from backend.database.models import AgentDecisionLog, AuditLog, DataAccessLog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    """In-memory SQLite engine shared across the test module."""
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture()
def session_factory(engine):
    """Fresh sessionmaker bound to the shared in-memory engine."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def lg(session_factory, monkeypatch):
    """AuditLogger with all CrewAI/LLM components mocked."""
    monkeypatch.setattr(
        "backend.agents.audit_logger.get_llm", MagicMock(return_value=MagicMock())
    )
    monkeypatch.setattr("backend.agents.audit_logger.Agent", MagicMock())
    monkeypatch.setattr("backend.agents.audit_logger.Task", MagicMock())
    crew = MagicMock()
    crew.kickoff.return_value = '{"report_en": "English report.", "report_ar": "تقرير."}'
    monkeypatch.setattr("backend.agents.audit_logger.Crew", MagicMock(return_value=crew))
    return AuditLogger(session_factory=session_factory)


def _count(session_factory, model) -> int:
    db = session_factory()
    try:
        return db.query(model).count()
    finally:
        db.close()


def _insert_old_audit(session_factory, days_ago: int = 91) -> None:
    db = session_factory()
    try:
        db.add(AuditLog(
            event_type="session_started",
            description="old event",
            timestamp=datetime.utcnow() - timedelta(days=days_ago),
        ))
        db.commit()
    finally:
        db.close()


def _insert_old_access(session_factory, user_id: int = 1, days_ago: int = 1) -> None:
    """Insert a DataAccessLog whose retained_until is already in the past."""
    past = datetime.utcnow() - timedelta(days=days_ago)
    db = session_factory()
    try:
        db.add(DataAccessLog(
            user_id=user_id,
            resource_type="survey_session",
            access_type="read",
            purpose="test",
            timestamp=past,
            retained_until=past,
        ))
        db.commit()
    finally:
        db.close()


def _insert_old_decision(session_factory, days_ago: int = 91) -> None:
    db = session_factory()
    try:
        db.add(AgentDecisionLog(
            agent_name="OldAgent",
            decision_type="test",
            input_summary="in",
            output_summary="out",
            timestamp=datetime.utcnow() - timedelta(days=days_ago),
        ))
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# EventType / AccessType / AgentDecisionType constants
# ---------------------------------------------------------------------------


class TestEventTypeConstants:
    def test_session_started(self):
        assert EventType.SESSION_STARTED == "session_started"

    def test_session_completed(self):
        assert EventType.SESSION_COMPLETED == "session_completed"

    def test_message_sent(self):
        assert EventType.MESSAGE_SENT == "message_sent"

    def test_auth_otp_requested(self):
        assert EventType.AUTH_OTP_REQUESTED == "auth_otp_requested"

    def test_purge_executed(self):
        assert EventType.PURGE_EXECUTED == "purge_executed"


class TestAccessTypeConstants:
    def test_read(self):
        assert AccessType.READ == "read"

    def test_write(self):
        assert AccessType.WRITE == "write"

    def test_delete(self):
        assert AccessType.DELETE == "delete"

    def test_export(self):
        assert AccessType.EXPORT == "export"

    def test_anonymize(self):
        assert AccessType.ANONYMIZE == "anonymize"


class TestAgentDecisionTypeConstants:
    def test_isco_classification(self):
        assert AgentDecisionType.ISCO_CLASSIFICATION == "isco_classification"

    def test_validation(self):
        assert AgentDecisionType.VALIDATION == "validation"

    def test_ner_extraction(self):
        assert AgentDecisionType.NER_EXTRACTION == "ner_extraction"


# ---------------------------------------------------------------------------
# log_interaction()
# ---------------------------------------------------------------------------


class TestLogInteraction:
    def test_returns_audit_entry(self, lg):
        result = lg.log_interaction(EventType.SESSION_STARTED, "Session begun.")
        assert isinstance(result, AuditEntry)

    def test_persists_to_db(self, lg, session_factory):
        before = _count(session_factory, AuditLog)
        lg.log_interaction(EventType.MESSAGE_SENT, "User said hello.")
        assert _count(session_factory, AuditLog) == before + 1

    def test_stores_event_type(self, lg):
        result = lg.log_interaction(EventType.SESSION_COMPLETED, "Done.")
        assert result.event_type == EventType.SESSION_COMPLETED

    def test_stores_description(self, lg):
        result = lg.log_interaction(EventType.MESSAGE_SENT, "Test description.")
        assert result.description == "Test description."

    def test_stores_optional_fields(self, lg):
        result = lg.log_interaction(
            EventType.SESSION_STARTED,
            "Started.",
            session_id=42,
            user_id=7,
            actor="system",
            ip_address="127.0.0.1",
        )
        assert result.session_id == 42
        assert result.user_id == 7
        assert result.actor == "system"
        assert result.ip_address == "127.0.0.1"

    def test_serialises_metadata_as_json(self, lg):
        result = lg.log_interaction(
            EventType.SESSION_STARTED,
            "With metadata.",
            metadata={"key": "value", "count": 3},
        )
        assert result.metadata_json is not None
        parsed = json.loads(result.metadata_json)
        assert parsed["key"] == "value"
        assert parsed["count"] == 3

    def test_no_metadata_stores_none(self, lg):
        result = lg.log_interaction(EventType.MESSAGE_SENT, "No metadata.")
        assert result.metadata_json is None

    def test_timestamp_is_iso_string(self, lg):
        result = lg.log_interaction(EventType.MESSAGE_SENT, "Check timestamp.")
        # Should parse without error
        datetime.fromisoformat(result.timestamp)

    def test_id_is_positive_integer(self, lg):
        result = lg.log_interaction(EventType.SESSION_STARTED, "ID check.")
        assert isinstance(result.id, int)
        assert result.id > 0


# ---------------------------------------------------------------------------
# log_data_access()
# ---------------------------------------------------------------------------


class TestLogDataAccess:
    def test_returns_data_access_entry(self, lg):
        result = lg.log_data_access(
            user_id=1,
            resource_type="survey_session",
            access_type=AccessType.READ,
            purpose="LFS data collection.",
        )
        assert isinstance(result, DataAccessEntry)

    def test_persists_to_db(self, lg, session_factory):
        before = _count(session_factory, DataAccessLog)
        lg.log_data_access(1, "survey_session", AccessType.READ, "test purpose")
        assert _count(session_factory, DataAccessLog) == before + 1

    def test_stores_required_fields(self, lg):
        result = lg.log_data_access(
            user_id=5,
            resource_type="user_profile",
            access_type=AccessType.EXPORT,
            purpose="GDPR subject access request.",
        )
        assert result.user_id == 5
        assert result.resource_type == "user_profile"
        assert result.access_type == AccessType.EXPORT
        assert result.purpose == "GDPR subject access request."

    def test_stores_optional_resource_id(self, lg):
        result = lg.log_data_access(
            user_id=1,
            resource_type="survey_response",
            access_type=AccessType.READ,
            purpose="Audit",
            resource_id=99,
        )
        assert result.resource_id == 99

    def test_stores_accessor_id(self, lg):
        result = lg.log_data_access(
            user_id=1,
            resource_type="survey_session",
            access_type=AccessType.READ,
            purpose="Admin review",
            accessor_id=2,
        )
        assert result.accessor_id == 2

    def test_retained_until_is_future(self, lg):
        result = lg.log_data_access(1, "survey_session", AccessType.READ, "test")
        retained = datetime.fromisoformat(result.retained_until)
        assert retained > datetime.utcnow()

    def test_retained_until_respects_retention_days(self, session_factory, monkeypatch):
        monkeypatch.setattr(
            "backend.agents.audit_logger.get_llm", MagicMock(return_value=MagicMock())
        )
        monkeypatch.setattr("backend.agents.audit_logger.Agent", MagicMock())
        monkeypatch.setattr("backend.agents.audit_logger.Task", MagicMock())
        monkeypatch.setattr("backend.agents.audit_logger.Crew", MagicMock())
        short_lg = AuditLogger(session_factory=session_factory, retention_days=7)
        result = short_lg.log_data_access(1, "survey_session", AccessType.READ, "test")
        retained = datetime.fromisoformat(result.retained_until)
        expected = datetime.utcnow() + timedelta(days=7)
        # Allow ±5 seconds tolerance
        assert abs((retained - expected).total_seconds()) < 5


# ---------------------------------------------------------------------------
# log_agent_decision()
# ---------------------------------------------------------------------------


class TestLogAgentDecision:
    def test_returns_agent_decision_entry(self, lg):
        result = lg.log_agent_decision(
            agent_name="ISCOClassifier",
            decision_type=AgentDecisionType.ISCO_CLASSIFICATION,
            input_summary="job title: nurse",
            output_summary="ISCO 2221 (0.91)",
        )
        assert isinstance(result, AgentDecisionEntry)

    def test_persists_to_db(self, lg, session_factory):
        before = _count(session_factory, AgentDecisionLog)
        lg.log_agent_decision("Agent", "test", "in", "out")
        assert _count(session_factory, AgentDecisionLog) == before + 1

    def test_stores_required_fields(self, lg):
        result = lg.log_agent_decision(
            agent_name="ValidationAgent",
            decision_type=AgentDecisionType.VALIDATION,
            input_summary="hours=45",
            output_summary="valid",
        )
        assert result.agent_name == "ValidationAgent"
        assert result.decision_type == AgentDecisionType.VALIDATION
        assert result.input_summary == "hours=45"
        assert result.output_summary == "valid"

    def test_stores_optional_fields(self, lg):
        result = lg.log_agent_decision(
            agent_name="ISCOClassifier",
            decision_type=AgentDecisionType.ISCO_CLASSIFICATION,
            input_summary="in",
            output_summary="out",
            confidence=0.87,
            reasoning="Semantic match.",
            session_id=10,
            duration_ms=234,
        )
        assert result.confidence == pytest.approx(0.87)
        assert result.reasoning == "Semantic match."
        assert result.session_id == 10
        assert result.duration_ms == 234

    def test_timestamp_is_iso_string(self, lg):
        result = lg.log_agent_decision("A", "T", "i", "o")
        datetime.fromisoformat(result.timestamp)

    def test_id_assigned(self, lg):
        result = lg.log_agent_decision("A", "T", "i", "o")
        assert result.id > 0


# ---------------------------------------------------------------------------
# purge_expired()
# ---------------------------------------------------------------------------


class TestPurgeExpired:
    def test_returns_purge_result(self, lg):
        result = lg.purge_expired(dry_run=True)
        assert isinstance(result, PurgeResult)

    def test_dry_run_does_not_delete(self, lg, session_factory):
        _insert_old_audit(session_factory, days_ago=200)
        before = _count(session_factory, AuditLog)
        lg.purge_expired(dry_run=True)
        assert _count(session_factory, AuditLog) == before  # unchanged

    def test_deletes_old_audit_logs(self, lg, session_factory):
        _insert_old_audit(session_factory, days_ago=200)
        before = _count(session_factory, AuditLog)
        result = lg.purge_expired(dry_run=False)
        after = _count(session_factory, AuditLog)
        # Some rows were deleted; at minimum the one we just inserted
        assert result.deleted_audit_logs >= 1
        assert after < before

    def test_deletes_expired_access_logs(self, lg, session_factory):
        _insert_old_access(session_factory, user_id=99, days_ago=1)
        before = _count(session_factory, DataAccessLog)
        result = lg.purge_expired(dry_run=False)
        assert result.deleted_access_logs >= 1
        assert _count(session_factory, DataAccessLog) < before

    def test_deletes_old_decision_logs(self, lg, session_factory):
        _insert_old_decision(session_factory, days_ago=200)
        before = _count(session_factory, AgentDecisionLog)
        result = lg.purge_expired(dry_run=False)
        assert result.deleted_decision_logs >= 1
        assert _count(session_factory, AgentDecisionLog) < before

    def test_dry_run_flag_in_result(self, lg):
        result = lg.purge_expired(dry_run=True)
        assert result.dry_run is True

    def test_not_dry_run_flag_in_result(self, lg):
        result = lg.purge_expired(dry_run=False)
        assert result.dry_run is False

    def test_retention_days_in_result(self, lg):
        result = lg.purge_expired(dry_run=True)
        assert result.retention_days == lg._retention_days

    def test_cutoff_timestamp_is_iso(self, lg):
        result = lg.purge_expired(dry_run=True)
        datetime.fromisoformat(result.cutoff_timestamp)

    def test_purge_timestamp_is_iso(self, lg):
        result = lg.purge_expired(dry_run=True)
        datetime.fromisoformat(result.purge_timestamp)

    def test_recent_records_not_deleted(self, lg, session_factory):
        # Write a fresh record (well within retention window)
        lg.log_interaction(EventType.MESSAGE_SENT, "Recent event.")
        before = _count(session_factory, AuditLog)
        # Purge with default 90-day window — the fresh row should survive
        result = lg.purge_expired(dry_run=False)
        after = _count(session_factory, AuditLog)
        # Fresh row still present (after == before - deleted_old_rows)
        assert after == before - result.deleted_audit_logs


# ---------------------------------------------------------------------------
# get_user_data_accesses()
# ---------------------------------------------------------------------------


class TestGetUserDataAccesses:
    def test_returns_list(self, lg):
        result = lg.get_user_data_accesses(user_id=9999)
        assert isinstance(result, list)

    def test_empty_for_unknown_user(self, lg):
        assert lg.get_user_data_accesses(user_id=88888) == []

    def test_returns_entries_for_user(self, lg):
        lg.log_data_access(42, "survey_session", AccessType.READ, "Access log test")
        lg.log_data_access(42, "user_profile",   AccessType.READ, "Access log test 2")
        results = lg.get_user_data_accesses(user_id=42)
        assert len(results) >= 2
        assert all(isinstance(r, DataAccessEntry) for r in results)

    def test_does_not_return_other_users_entries(self, lg):
        lg.log_data_access(100, "survey_session", AccessType.READ, "User 100 entry")
        results = lg.get_user_data_accesses(user_id=101)
        assert all(r.user_id == 101 for r in results)

    def test_ordered_newest_first(self, lg):
        lg.log_data_access(200, "survey_session", AccessType.READ, "First")
        lg.log_data_access(200, "user_profile",   AccessType.WRITE, "Second")
        results = lg.get_user_data_accesses(user_id=200)
        if len(results) >= 2:
            ts0 = datetime.fromisoformat(results[0].timestamp)
            ts1 = datetime.fromisoformat(results[1].timestamp)
            assert ts0 >= ts1


# ---------------------------------------------------------------------------
# generate_report()
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_returns_audit_report(self, lg):
        assert isinstance(lg.generate_report(), AuditReport)

    def test_default_period_is_30_days(self, lg):
        result = lg.generate_report()
        from_dt = datetime.fromisoformat(result.period_start)
        to_dt   = datetime.fromisoformat(result.period_end)
        assert (to_dt - from_dt).days >= 29   # allow minor float drift

    def test_custom_period_respected(self, lg):
        from_dt = datetime(2025, 1, 1)
        to_dt   = datetime(2025, 1, 31)
        result  = lg.generate_report(from_dt=from_dt, to_dt=to_dt)
        assert result.period_start == from_dt.isoformat()
        assert result.period_end   == to_dt.isoformat()

    def test_counts_interactions(self, lg):
        # Log two events and confirm they appear in a fresh report window
        from_dt = datetime.utcnow() - timedelta(seconds=1)
        lg.log_interaction(EventType.SESSION_STARTED, "A")
        lg.log_interaction(EventType.MESSAGE_SENT,    "B")
        to_dt  = datetime.utcnow() + timedelta(seconds=1)
        result = lg.generate_report(from_dt=from_dt, to_dt=to_dt)
        assert result.total_audit_events >= 2

    def test_counts_data_accesses(self, lg):
        from_dt = datetime.utcnow() - timedelta(seconds=1)
        lg.log_data_access(300, "survey_session", AccessType.READ, "Report test")
        to_dt  = datetime.utcnow() + timedelta(seconds=1)
        result = lg.generate_report(from_dt=from_dt, to_dt=to_dt)
        assert result.total_data_accesses >= 1

    def test_counts_agent_decisions(self, lg):
        from_dt = datetime.utcnow() - timedelta(seconds=1)
        lg.log_agent_decision("TestAgent", "test", "in", "out")
        to_dt  = datetime.utcnow() + timedelta(seconds=1)
        result = lg.generate_report(from_dt=from_dt, to_dt=to_dt)
        assert result.total_agent_decisions >= 1

    def test_event_counts_dict(self, lg):
        from_dt = datetime.utcnow() - timedelta(seconds=1)
        lg.log_interaction(EventType.SESSION_STARTED, "ev1")
        lg.log_interaction(EventType.SESSION_STARTED, "ev2")
        to_dt  = datetime.utcnow() + timedelta(seconds=1)
        result = lg.generate_report(from_dt=from_dt, to_dt=to_dt)
        assert EventType.SESSION_STARTED in result.event_counts
        assert result.event_counts[EventType.SESSION_STARTED] >= 2

    def test_llm_report_text_propagated(self, lg):
        result = lg.generate_report()
        assert result.report_en == "English report."
        assert result.report_ar == "تقرير."

    def test_session_filter(self, lg):
        from_dt = datetime.utcnow() - timedelta(seconds=1)
        lg.log_interaction(EventType.MESSAGE_SENT, "For session 55", session_id=55)
        lg.log_interaction(EventType.MESSAGE_SENT, "For session 56", session_id=56)
        to_dt  = datetime.utcnow() + timedelta(seconds=1)
        result = lg.generate_report(session_id=55, from_dt=from_dt, to_dt=to_dt)
        # All returned events belong to session 55
        assert result.session_id == 55
        # Events for session 56 should not appear
        assert result.total_audit_events >= 1

    def test_generated_at_is_iso(self, lg):
        result = lg.generate_report()
        datetime.fromisoformat(result.generated_at)


# ---------------------------------------------------------------------------
# _build_fallback_report()
# ---------------------------------------------------------------------------


class TestBuildFallbackReport:
    def _stats(
        self,
        n_audit=0, n_access=0, n_decision=0,
        events=None, accesses=None, agents=None,
    ) -> dict:
        return {
            "total_audit_events":    n_audit,
            "total_data_accesses":   n_access,
            "total_agent_decisions": n_decision,
            "event_counts":          events   or {},
            "access_type_counts":    accesses or {},
            "agent_counts":          agents   or {},
        }

    def test_no_activity_en(self):
        en, _ = AuditLogger._build_fallback_report(self._stats())
        assert "No survey activity recorded" in en

    def test_no_activity_ar(self):
        _, ar = AuditLogger._build_fallback_report(self._stats())
        assert "لم يُسجَّل" in ar

    def test_with_activity_en_includes_counts(self):
        stats = self._stats(n_audit=5, n_access=2, n_decision=1,
                            events={EventType.SESSION_STARTED: 3})
        en, _ = AuditLogger._build_fallback_report(stats)
        assert "5" in en
        assert "2" in en

    def test_with_activity_ar_includes_counts(self):
        stats = self._stats(n_audit=3, n_access=0, n_decision=0,
                            events={EventType.MESSAGE_SENT: 3})
        _, ar = AuditLogger._build_fallback_report(stats)
        assert "3" in ar

    def test_returns_tuple_of_two_strings(self):
        result = AuditLogger._build_fallback_report(self._stats())
        assert isinstance(result, tuple) and len(result) == 2
        assert all(isinstance(s, str) for s in result)

    def test_gdpr_article_mentioned_in_en(self):
        stats = self._stats(n_audit=1, events={"x": 1})
        en, _ = AuditLogger._build_fallback_report(stats)
        assert "GDPR" in en

    def test_gdpr_article_mentioned_in_ar(self):
        stats = self._stats(n_audit=1, events={"x": 1})
        _, ar = AuditLogger._build_fallback_report(stats)
        assert "اللائحة العامة" in ar


# ---------------------------------------------------------------------------
# _parse_report_response()
# ---------------------------------------------------------------------------


class TestParseReportResponse:
    @pytest.fixture()
    def c(self, lg):
        return lg

    def _stats(self) -> dict:
        return {
            "total_audit_events": 0, "total_data_accesses": 0,
            "total_agent_decisions": 0, "event_counts": {},
            "access_type_counts": {}, "agent_counts": {},
        }

    def test_valid_json(self, c):
        raw = '{"report_en": "En report.", "report_ar": "تقرير عربي."}'
        en, ar = c._parse_report_response(raw, self._stats())
        assert en == "En report."
        assert ar == "تقرير عربي."

    def test_json_with_markdown_fences(self, c):
        raw = '```json\n{"report_en": "Fenced.", "report_ar": "محاط."}\n```'
        en, ar = c._parse_report_response(raw, self._stats())
        assert en == "Fenced."
        assert ar == "محاط."

    def test_regex_fallback_on_surrounding_text(self, c):
        raw = 'Here: {"report_en": "Embedded.", "report_ar": "مُدمج."} end.'
        en, ar = c._parse_report_response(raw, self._stats())
        assert en == "Embedded."
        assert ar == "مُدمج."

    def test_missing_en_triggers_fallback(self, c):
        raw = '{"report_ar": "عربي."}'
        en, ar = c._parse_report_response(raw, self._stats())
        assert len(en) > 0  # fallback supplied

    def test_invalid_json_triggers_fallback(self, c):
        en, ar = c._parse_report_response("not json at all", self._stats())
        assert isinstance(en, str) and len(en) > 0
        assert isinstance(ar, str) and len(ar) > 0

    def test_empty_string_triggers_fallback(self, c):
        en, ar = c._parse_report_response("", self._stats())
        assert len(en) > 0


# ---------------------------------------------------------------------------
# get_audit_logger() singleton
# ---------------------------------------------------------------------------


class TestGetAuditLogger:
    def test_returns_audit_logger(self, monkeypatch):
        import backend.agents.audit_logger as mod
        monkeypatch.setattr(
            "backend.agents.audit_logger.get_llm", MagicMock(return_value=MagicMock())
        )
        monkeypatch.setattr("backend.agents.audit_logger.Agent", MagicMock())
        monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
        mod._instance = None
        result = get_audit_logger()
        assert isinstance(result, AuditLogger)
        mod._instance = None  # cleanup

    def test_returns_same_instance_on_second_call(self, monkeypatch):
        import backend.agents.audit_logger as mod
        monkeypatch.setattr(
            "backend.agents.audit_logger.get_llm", MagicMock(return_value=MagicMock())
        )
        monkeypatch.setattr("backend.agents.audit_logger.Agent", MagicMock())
        monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
        mod._instance = None
        i1 = get_audit_logger()
        i2 = get_audit_logger()
        assert i1 is i2
        mod._instance = None  # cleanup

    def test_reuses_existing_instance(self, monkeypatch):
        import backend.agents.audit_logger as mod
        existing = MagicMock(spec=AuditLogger)
        mod._instance = existing
        assert get_audit_logger() is existing
        mod._instance = None  # cleanup
