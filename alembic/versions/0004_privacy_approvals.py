"""Add minimal privacy approval audit records.

Revision ID: 0004_privacy_approvals
Revises: 0003_chunk_embeddings
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004_privacy_approvals"
down_revision: str | None = "0003_chunk_embeddings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "privacy_approvals",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("destination_digest", sa.String(length=64), nullable=False),
        sa.Column("payload_sha256", sa.String(length=64), nullable=True),
        sa.Column("patient_ref", sa.String(length=64), nullable=True),
        sa.Column("actor_ref", sa.String(length=64), nullable=True),
        sa.Column("question_type", sa.String(length=32), nullable=False),
        sa.Column("observation_count", sa.Integer(), nullable=False),
        sa.Column("reason_codes", sa.JSON(), nullable=False),
        sa.Column("review_flags", sa.JSON(), nullable=False),
        sa.Column("entity_counts", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_privacy_approvals_request_id", "privacy_approvals", ["request_id"]
    )
    op.create_index("ix_privacy_approvals_status", "privacy_approvals", ["status"])
    op.create_index(
        "ix_privacy_approvals_status_created",
        "privacy_approvals",
        ["status", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_privacy_approvals_status_created", table_name="privacy_approvals"
    )
    op.drop_index("ix_privacy_approvals_status", table_name="privacy_approvals")
    op.drop_index("ix_privacy_approvals_request_id", table_name="privacy_approvals")
    op.drop_table("privacy_approvals")
