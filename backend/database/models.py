from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime,
    ForeignKey, Float, Text
)
from sqlalchemy.orm import relationship
from backend.database.connection import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    phone = Column(String, unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    deleted_at = Column(DateTime, nullable=True, index=True)  # soft-delete: NULL = active

    otp_codes = relationship("OTPCode", back_populates="user")
    survey_sessions = relationship("SurveySession", back_populates="user")


class OTPCode(Base):
    __tablename__ = "otp_codes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    code = Column(String, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    is_used = Column(Boolean, default=False, nullable=False)

    user = relationship("User", back_populates="otp_codes")


class SurveySession(Base):
    __tablename__ = "survey_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String, default="in_progress", nullable=False)
    language = Column(String, nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    deleted_at = Column(DateTime, nullable=True, index=True)  # soft-delete: NULL = active

    user = relationship("User", back_populates="survey_sessions")
    responses = relationship("SurveyResponse", back_populates="session", cascade="all, delete-orphan")


class SurveyResponse(Base):
    """
    One row per (session, question) answer. Answers are versioned, not
    overwritten: a correction that lands after the original row already
    exists (e.g. job_title, which is written immediately at answer-time
    together with its ISCO classification, before VALIDATING can happen)
    is inserted as a NEW row with `supersedes_id` pointing at the row it
    replaces; the old row gets `deleted_at` set. This keeps the original
    answer, code, confidence, and the fact a correction occurred — material
    the Audit and HITL quality flows depend on — instead of destroying it.

    `deleted_at` here means "not the active revision," same soft-delete
    convention used elsewhere in this schema; it does not mean the row was
    a GDPR-style deletion. Superseded rows are retained indefinitely for
    audit purposes and never physically removed by a correction.

    All reads must filter `deleted_at IS NULL` to see only the active
    revision per (session_id, question_id).
    """
    __tablename__ = "survey_responses"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("survey_sessions.id"), nullable=False)
    question_id = Column(String, nullable=False)
    answer = Column(Text, nullable=False)
    isco_code = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    supersedes_id = Column(Integer, ForeignKey("survey_responses.id"), nullable=True, index=True)
    deleted_at = Column(DateTime, nullable=True, index=True)  # soft-delete: NULL = active revision

    session = relationship("SurveySession", back_populates="responses")


# ---------------------------------------------------------------------------
# Audit tables (GDPR compliance & audit trail)
# ---------------------------------------------------------------------------

class AuditLog(Base):
    """General survey interaction log — one row per notable system event."""

    __tablename__ = "audit_logs"

    id            = Column(Integer, primary_key=True, index=True)
    session_id    = Column(Integer, ForeignKey("survey_sessions.id"), nullable=True,  index=True)
    user_id       = Column(Integer, ForeignKey("users.id"),           nullable=True,  index=True)
    event_type    = Column(String,  nullable=False, index=True)
    actor         = Column(String,  nullable=True)          # "user" | "agent:X" | "system"
    description   = Column(Text,    nullable=False)
    ip_address    = Column(String,  nullable=True)
    metadata_json = Column(Text,    nullable=True)          # JSON string of extra data
    timestamp     = Column(DateTime, nullable=False, index=True)


class DataAccessLog(Base):
    """GDPR data-access record — tracks who read/wrote/exported personal data."""

    __tablename__ = "data_access_logs"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    accessor_id   = Column(Integer, ForeignKey("users.id"), nullable=True)
    resource_type = Column(String,  nullable=False)         # "survey_session" | "user_profile" | …
    resource_id   = Column(Integer, nullable=True)          # PK of the accessed record
    access_type   = Column(String,  nullable=False)         # "read" | "write" | "delete" | "export"
    purpose       = Column(Text,    nullable=False)          # GDPR lawful basis / declared purpose
    ip_address    = Column(String,  nullable=True)
    timestamp     = Column(DateTime, nullable=False, index=True)
    retained_until = Column(DateTime, nullable=False, index=True)  # auto-purge after this date


class AgentDecisionLog(Base):
    """Records every AI-agent decision for transparency and auditability."""

    __tablename__ = "agent_decision_logs"

    id             = Column(Integer, primary_key=True, index=True)
    session_id     = Column(Integer, ForeignKey("survey_sessions.id"), nullable=True, index=True)
    agent_name     = Column(String,  nullable=False, index=True)
    decision_type  = Column(String,  nullable=False)
    input_summary  = Column(Text,    nullable=False)
    output_summary = Column(Text,    nullable=False)
    confidence     = Column(Float,   nullable=True)
    reasoning      = Column(Text,    nullable=True)
    duration_ms    = Column(Integer, nullable=True)
    timestamp      = Column(DateTime, nullable=False, index=True)


# ---------------------------------------------------------------------------
# HITL quality-review table
# ---------------------------------------------------------------------------

class QualityReview(Base):
    """
    Automated quality assessment result for one survey session.

    Created by HITLQualityManager.review_session(); updated by
    HITLQualityManager.resolve_review() when a human supervisor acts.
    """

    __tablename__ = "quality_reviews"

    id                 = Column(Integer, primary_key=True, index=True)
    session_id         = Column(Integer, ForeignKey("survey_sessions.id"), nullable=False, index=True)
    quality_score      = Column(Float,   nullable=False)
    passed             = Column(Boolean, nullable=False)
    flagged_count      = Column(Integer, nullable=False, default=0)
    flagged_items_json = Column(Text,    nullable=True)         # JSON list of FlaggedItem dicts
    escalated          = Column(Boolean, default=False, nullable=False, index=True)
    escalation_reason  = Column(Text,    nullable=True)
    reviewer_notes     = Column(Text,    nullable=True)         # set by human supervisor
    reviewed_by        = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at         = Column(DateTime, nullable=False, index=True)
    reviewed_at        = Column(DateTime, nullable=True)


# ---------------------------------------------------------------------------
# HITL occupation-classification queue
# ---------------------------------------------------------------------------

class HITLQueue(Base):
    """
    Low-confidence ISCO-08 classifications flagged for human review.

    Created by ISCOClassifier when confidence < HITL_THRESHOLD (0.70).
    Reviewed via GET /hitl/queue + POST /hitl/review endpoints.
    Priority: HIGH when confidence < 0.50, MEDIUM when 0.50 <= confidence < 0.70.
    """

    __tablename__ = "hitl_queue"

    id               = Column(Integer, primary_key=True, index=True)
    session_id       = Column(Integer, ForeignKey("survey_sessions.id"), nullable=True,  index=True)
    response_id      = Column(Integer, ForeignKey("survey_responses.id"), nullable=True, index=True)
    respondent_email = Column(String,  nullable=True)
    raw_text         = Column(Text,    nullable=False)    # original job description text
    ai_code          = Column(String,  nullable=False)    # ISCO code suggested by AI
    ai_confidence    = Column(Float,   nullable=False)
    ai_reasoning     = Column(Text,    nullable=True)
    hierarchy_path   = Column(Text,    nullable=True)     # JSON list e.g. ["2","25","251","2512"]
    priority         = Column(String,  nullable=False, default="MEDIUM", index=True)  # HIGH | MEDIUM
    status           = Column(String,  nullable=False, default="pending", index=True)  # pending | reviewed | rejected
    reviewer_code    = Column(String,  nullable=True)     # final code after human review
    reviewer_notes   = Column(Text,    nullable=True)
    reviewed_by      = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at       = Column(DateTime, nullable=False, index=True)
    reviewed_at      = Column(DateTime, nullable=True)


# ---------------------------------------------------------------------------
# Survey report table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Person Register — pre-filled demographic records (gap: 40-50% question reduction)
# ---------------------------------------------------------------------------

class PersonRegister(Base):
    """
    Stores known respondent attributes from previous survey rounds or
    admin data imports.  When a new survey session starts for a known user,
    these fields are injected into the ConversationContext so the agent
    can skip (or confirm) already-known answers.

    A single user may have multiple records (one per reference period).
    The most recent record with is_active=True is used for pre-fill.
    """

    __tablename__ = "person_register"

    id                  = Column(Integer, primary_key=True, index=True)
    user_id             = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    reference_period    = Column(String,  nullable=False)          # e.g. "2025-Q1"
    employment_status   = Column(String,  nullable=True)           # employed | unemployed | not_in_labour_force
    job_title           = Column(String,  nullable=True)
    industry            = Column(String,  nullable=True)
    isco_code           = Column(String,  nullable=True)           # last known ISCO code
    isic_code           = Column(String,  nullable=True)           # last known ISIC division
    isced_level         = Column(Integer, nullable=True)           # ISCED 2011 level 0-8
    hours_per_week      = Column(Float,   nullable=True)
    employment_type     = Column(String,  nullable=True)           # full_time | part_time | self_employed | contractor
    nationality         = Column(String,  nullable=True)
    age_group           = Column(String,  nullable=True)           # e.g. "25-34"
    gender              = Column(String,  nullable=True)
    is_active           = Column(Boolean, default=True, nullable=False)
    source              = Column(String,  nullable=True)           # "admin_import" | "survey_round"
    created_at          = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at          = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User")


class SurveyReportRecord(Base):
    """
    Persisted output of ReportGenerator.generate().

    One row per generation run.  Multiple rows may exist for the same
    session if generate(regenerate=True) is called; callers should use
    the most recent row (ORDER BY generated_at DESC).
    """

    __tablename__ = "survey_report_records"

    id                 = Column(Integer, primary_key=True, index=True)
    session_id         = Column(Integer, ForeignKey("survey_sessions.id"), nullable=False, index=True)
    language           = Column(String,  nullable=False)
    profile_json       = Column(Text,    nullable=False)     # JSON of EmploymentProfile
    quality_score      = Column(Float,   nullable=True)
    quality_status     = Column(String,  nullable=True)
    flagged_count      = Column(Integer, nullable=False, default=0)
    report_en          = Column(Text,    nullable=False)
    report_ar          = Column(Text,    nullable=False)
    recommendations_en = Column(Text,    nullable=False)
    recommendations_ar = Column(Text,    nullable=False)
    generated_at       = Column(DateTime, nullable=False, index=True)
