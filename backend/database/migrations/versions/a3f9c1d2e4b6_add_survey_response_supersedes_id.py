"""add_survey_response_supersedes_id

Revision ID: a3f9c1d2e4b6
Revises: f514fcb81c72
Create Date: 2026-08-02 00:00:00.000000

Adds versioning to survey_responses: a correction to a field that already
has a persisted row (e.g. job_title, written at answer-time together with
its ISCO classification) is stored as a NEW row with supersedes_id pointing
at the row it replaces, rather than overwritten in place or silently
dropped. The prior row is marked inactive via the existing deleted_at
soft-delete column. See backend/database/models.py::SurveyResponse for the
full rationale.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3f9c1d2e4b6'
down_revision: Union[str, None] = 'f514fcb81c72'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'survey_responses',
        sa.Column('supersedes_id', sa.Integer(), nullable=True),
    )
    op.create_index(
        op.f('ix_survey_responses_supersedes_id'),
        'survey_responses', ['supersedes_id'], unique=False,
    )
    op.create_foreign_key(
        'fk_survey_responses_supersedes_id',
        'survey_responses', 'survey_responses',
        ['supersedes_id'], ['id'],
    )


def downgrade() -> None:
    op.drop_constraint('fk_survey_responses_supersedes_id', 'survey_responses', type_='foreignkey')
    op.drop_index(op.f('ix_survey_responses_supersedes_id'), table_name='survey_responses')
    op.drop_column('survey_responses', 'supersedes_id')
