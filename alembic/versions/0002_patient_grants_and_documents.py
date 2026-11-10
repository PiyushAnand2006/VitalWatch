"""Add patient grants and protected document metadata.

Revision ID: 0002_grants_and_documents
Revises: 0001_enable_pgvector
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002_grants_and_documents"
down_revision: str | None = "0001_enable_pgvector"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "patient_grants",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("patient_id", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "patient_id", name="uq_patient_grant"),
    )
    op.create_index("ix_patient_grants_user_id", "patient_grants", ["user_id"])
    op.create_index("ix_patient_grants_patient_id", "patient_grants", ["patient_id"])
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("patient_id", sa.String(length=128), nullable=False),
        sa.Column("uploader_id", sa.String(length=128), nullable=False),
        sa.Column("source_name", sa.String(length=255), nullable=False),
        sa.Column("storage_key", sa.String(length=128), nullable=False),
        sa.Column("media_type", sa.String(length=128), nullable=False),
        sa.Column("checksum", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("patient_id", "checksum", name="uq_document_patient_checksum"),
        sa.UniqueConstraint("storage_key"),
    )
    op.create_index("ix_documents_patient_id", "documents", ["patient_id"])
    op.create_index("ix_documents_patient_created", "documents", ["patient_id", "created_at"])


def downgrade() -> None:
    op.drop_table("documents")
    op.drop_table("patient_grants")
