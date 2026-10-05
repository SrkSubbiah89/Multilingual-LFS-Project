"""Preserve historical reports while invalidating obsolete cached reports."""
from alembic import op
import sqlalchemy as sa

revision = "d841b2079c65"
down_revision = "c7e2a48f9d31"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("survey_report_records", sa.Column("invalidated_at", sa.DateTime(), nullable=True))
    op.create_index("ix_survey_report_records_invalidated_at", "survey_report_records", ["invalidated_at"])


def downgrade():
    op.drop_index("ix_survey_report_records_invalidated_at", table_name="survey_report_records")
    op.drop_column("survey_report_records", "invalidated_at")
