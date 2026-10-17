"""Enforce ORM-declared uniqueness for users.phone.

Revision ID: a6c8f42e7d93
Revises: d841b2079c65

Fixes missing unique constraint present in User.phone ORM declaration.
Separate additive migration preserves history of the original six revisions.
PostgreSQL unique constraints allow multiple NULL values, as expected for
an optional phone number. Apply only after a full Stage 4 test review.
"""

from alembic import op

revision = "a6c8f42e7d93"
down_revision = "d841b2079c65"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint("uq_users_phone", "users", ["phone"])


def downgrade() -> None:
    op.drop_constraint("uq_users_phone", "users", type_="unique")
