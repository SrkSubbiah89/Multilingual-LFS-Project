"""add_report_isic_isced_coherence_columns

Revision ID: c7e2a48f9d31
Revises: a3f9c1d2e4b6
Create Date: 2026-08-21 00:00:00.000000

Fixes a real bug: ReportGenerator computed semantic_coherence,
isic_classification, and isced_classification correctly on first
generation but never persisted them to survey_report_records, so any
later cache-hit read of an already-generated report (the common case --
a respondent or reviewer reopening a report after the first view) came
back with those three fields silently null, even though the underlying
data was present and classifiable all along. See
Documentation/Phase_2/ for the incident writeup that found this via a
live account (session 457).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c7e2a48f9d31'
down_revision: Union[str, None] = 'a3f9c1d2e4b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'survey_report_records',
        sa.Column('semantic_coherence_json', sa.Text(), nullable=True),
    )
    op.add_column(
        'survey_report_records',
        sa.Column('isic_classification_json', sa.Text(), nullable=True),
    )
    op.add_column(
        'survey_report_records',
        sa.Column('isced_classification_json', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('survey_report_records', 'isced_classification_json')
    op.drop_column('survey_report_records', 'isic_classification_json')
    op.drop_column('survey_report_records', 'semantic_coherence_json')
