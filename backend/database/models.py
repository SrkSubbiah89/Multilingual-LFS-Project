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

    user = relationship("User", back_populates="survey_sessions")
    responses = relationship("SurveyResponse", back_populates="session", cascade="all, delete-orphan")


class SurveyResponse(Base):
    __tablename__ = "survey_responses"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("survey_sessions.id"), nullable=False)
    question_id = Column(String, nullable=False)
    answer = Column(Text, nullable=False)
    isco_code = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)

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
