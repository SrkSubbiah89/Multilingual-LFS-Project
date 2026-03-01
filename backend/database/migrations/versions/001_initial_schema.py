"""Initial schema — all tables

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-03-02

Covers
------
  users                — respondent accounts
  otp_codes            — time-limited email OTP records
  survey_sessions      — one session per survey attempt
  survey_responses     — one row per collected field / ISCO result
  audit_logs           — general system event audit trail (GDPR)
  data_access_logs     — GDPR personal-data access log
  agent_decision_logs  — AI agent decision transparency log
  quality_reviews      — HITL automated quality assessment results
  survey_report_records — persisted bilingual LFS report output
"""

from alembic import op
import sqlalchemy as sa

# ── Revision identifiers ────────────────────────────────────────────────────

revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


# ── Upgrade ─────────────────────────────────────────────────────────────────

def upgrade() -> None:

    # ── users ────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id",         sa.Integer(),  nullable=False),
        sa.Column("email",      sa.String(),   nullable=False),
        sa.Column("phone",      sa.String(),   nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("is_active",  sa.Boolean(),  nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_id",    "users", ["id"],    unique=False)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # ── otp_codes ────────────────────────────────────────────────────────────
    op.create_table(
        "otp_codes",
        sa.Column("id",         sa.Integer(),  nullable=False),
        sa.Column("user_id",    sa.Integer(),  nullable=False),
        sa.Column("code",       sa.String(),   nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("is_used",    sa.Boolean(),  nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_otp_codes_id", "otp_codes", ["id"], unique=False)

    # ── survey_sessions ──────────────────────────────────────────────────────
    op.create_table(
        "survey_sessions",
        sa.Column("id",           sa.Integer(),  nullable=False),
        sa.Column("user_id",      sa.Integer(),  nullable=False),
        sa.Column("status",       sa.String(),   nullable=False),
        sa.Column("language",     sa.String(),   nullable=False),
        sa.Column("started_at",   sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_survey_sessions_id", "survey_sessions", ["id"], unique=False)

    # ── survey_responses ─────────────────────────────────────────────────────
    op.create_table(
        "survey_responses",
        sa.Column("id",               sa.Integer(), nullable=False),
        sa.Column("session_id",       sa.Integer(), nullable=False),
        sa.Column("question_id",      sa.String(),  nullable=False),
        sa.Column("answer",           sa.Text(),    nullable=False),
        sa.Column("isco_code",        sa.String(),  nullable=True),
        sa.Column("confidence_score", sa.Float(),   nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["survey_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_survey_responses_id", "survey_responses", ["id"], unique=False)

    # ── audit_logs ───────────────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("id",            sa.Integer(),  nullable=False),
        sa.Column("session_id",    sa.Integer(),  nullable=True),
        sa.Column("user_id",       sa.Integer(),  nullable=True),
        sa.Column("event_type",    sa.String(),   nullable=False),
        sa.Column("actor",         sa.String(),   nullable=True),
        sa.Column("description",   sa.Text(),     nullable=False),
        sa.Column("ip_address",    sa.String(),   nullable=True),
        sa.Column("metadata_json", sa.Text(),     nullable=True),
        sa.Column("timestamp",     sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["survey_sessions.id"]),
        sa.ForeignKeyConstraint(["user_id"],    ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_logs_id",         "audit_logs", ["id"],         unique=False)
    op.create_index("ix_audit_logs_session_id", "audit_logs", ["session_id"], unique=False)
    op.create_index("ix_audit_logs_user_id",    "audit_logs", ["user_id"],    unique=False)
    op.create_index("ix_audit_logs_event_type", "audit_logs", ["event_type"], unique=False)
    op.create_index("ix_audit_logs_timestamp",  "audit_logs", ["timestamp"],  unique=False)

    # ── data_access_logs ─────────────────────────────────────────────────────
    op.create_table(
        "data_access_logs",
        sa.Column("id",             sa.Integer(),  nullable=False),
        sa.Column("user_id",        sa.Integer(),  nullable=False),
        sa.Column("accessor_id",    sa.Integer(),  nullable=True),
        sa.Column("resource_type",  sa.String(),   nullable=False),
        sa.Column("resource_id",    sa.Integer(),  nullable=True),
        sa.Column("access_type",    sa.String(),   nullable=False),
        sa.Column("purpose",        sa.Text(),     nullable=False),
        sa.Column("ip_address",     sa.String(),   nullable=True),
        sa.Column("timestamp",      sa.DateTime(), nullable=False),
        sa.Column("retained_until", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"],     ["users.id"]),
        sa.ForeignKeyConstraint(["accessor_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_data_access_logs_id",             "data_access_logs", ["id"],             unique=False)
    op.create_index("ix_data_access_logs_user_id",        "data_access_logs", ["user_id"],        unique=False)
    op.create_index("ix_data_access_logs_timestamp",      "data_access_logs", ["timestamp"],      unique=False)
    op.create_index("ix_data_access_logs_retained_until", "data_access_logs", ["retained_until"], unique=False)

    # ── agent_decision_logs ──────────────────────────────────────────────────
    op.create_table(
        "agent_decision_logs",
        sa.Column("id",             sa.Integer(),  nullable=False),
        sa.Column("session_id",     sa.Integer(),  nullable=True),
        sa.Column("agent_name",     sa.String(),   nullable=False),
        sa.Column("decision_type",  sa.String(),   nullable=False),
        sa.Column("input_summary",  sa.Text(),     nullable=False),
        sa.Column("output_summary", sa.Text(),     nullable=False),
        sa.Column("confidence",     sa.Float(),    nullable=True),
        sa.Column("reasoning",      sa.Text(),     nullable=True),
        sa.Column("duration_ms",    sa.Integer(),  nullable=True),
        sa.Column("timestamp",      sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["survey_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_decision_logs_id",          "agent_decision_logs", ["id"],          unique=False)
    op.create_index("ix_agent_decision_logs_session_id",  "agent_decision_logs", ["session_id"],  unique=False)
    op.create_index("ix_agent_decision_logs_agent_name",  "agent_decision_logs", ["agent_name"],  unique=False)
    op.create_index("ix_agent_decision_logs_timestamp",   "agent_decision_logs", ["timestamp"],   unique=False)

    # ── quality_reviews ──────────────────────────────────────────────────────
    op.create_table(
        "quality_reviews",
        sa.Column("id",                 sa.Integer(), nullable=False),
        sa.Column("session_id",         sa.Integer(), nullable=False),
        sa.Column("quality_score",      sa.Float(),   nullable=False),
        sa.Column("passed",             sa.Boolean(), nullable=False),
        sa.Column("flagged_count",      sa.Integer(), nullable=False),
        sa.Column("flagged_items_json", sa.Text(),    nullable=True),
        sa.Column("escalated",          sa.Boolean(), nullable=False),
        sa.Column("escalation_reason",  sa.Text(),    nullable=True),
        sa.Column("reviewer_notes",     sa.Text(),    nullable=True),
        sa.Column("reviewed_by",        sa.Integer(), nullable=True),
        sa.Column("created_at",         sa.DateTime(), nullable=False),
        sa.Column("reviewed_at",        sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"],  ["survey_sessions.id"]),
        sa.ForeignKeyConstraint(["reviewed_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_quality_reviews_id",         "quality_reviews", ["id"],         unique=False)
    op.create_index("ix_quality_reviews_session_id", "quality_reviews", ["session_id"], unique=False)
    op.create_index("ix_quality_reviews_escalated",  "quality_reviews", ["escalated"],  unique=False)
    op.create_index("ix_quality_reviews_created_at", "quality_reviews", ["created_at"], unique=False)

    # ── survey_report_records ────────────────────────────────────────────────
    op.create_table(
        "survey_report_records",
        sa.Column("id",                 sa.Integer(), nullable=False),
        sa.Column("session_id",         sa.Integer(), nullable=False),
        sa.Column("language",           sa.String(),  nullable=False),
        sa.Column("profile_json",       sa.Text(),    nullable=False),
        sa.Column("quality_score",      sa.Float(),   nullable=True),
        sa.Column("quality_status",     sa.String(),  nullable=True),
        sa.Column("flagged_count",      sa.Integer(), nullable=False),
        sa.Column("report_en",          sa.Text(),    nullable=False),
        sa.Column("report_ar",          sa.Text(),    nullable=False),
        sa.Column("recommendations_en", sa.Text(),    nullable=False),
        sa.Column("recommendations_ar", sa.Text(),    nullable=False),
        sa.Column("generated_at",       sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["survey_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_survey_report_records_id",           "survey_report_records", ["id"],           unique=False)
    op.create_index("ix_survey_report_records_session_id",   "survey_report_records", ["session_id"],   unique=False)
    op.create_index("ix_survey_report_records_generated_at", "survey_report_records", ["generated_at"], unique=False)


# ── Downgrade ────────────────────────────────────────────────────────────────

def downgrade() -> None:
    op.drop_table("survey_report_records")
    op.drop_table("quality_reviews")
    op.drop_table("agent_decision_logs")
    op.drop_table("data_access_logs")
    op.drop_table("audit_logs")
    op.drop_table("survey_responses")
    op.drop_table("survey_sessions")
    op.drop_table("otp_codes")
    op.drop_table("users")
