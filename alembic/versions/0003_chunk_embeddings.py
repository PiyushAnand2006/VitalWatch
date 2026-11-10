"""Add patient-scoped chunk embeddings with pgvector and an HNSW index.

Revision ID: 0003_chunk_embeddings
Revises: 0002_grants_and_documents
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "0003_chunk_embeddings"
down_revision: str | None = "0002_grants_and_documents"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIMENSION = 384


def upgrade() -> None:
    op.create_table(
        "chunk_embeddings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("patient_id", sa.String(length=128), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(length=255), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("method", sa.String(length=32), nullable=False),
        sa.Column("quality", sa.String(length=16), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_revision", sa.String(length=128), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "chunk_id", name="uq_chunk_embedding_document_chunk"),
    )
    op.create_index("ix_chunk_embeddings_patient_id", "chunk_embeddings", ["patient_id"])
    op.create_index("ix_chunk_embeddings_document_id", "chunk_embeddings", ["document_id"])
    op.create_index(
        "ix_chunk_embeddings_patient_document",
        "chunk_embeddings",
        ["patient_id", "document_id"],
    )
    op.create_index(
        "ix_chunk_embeddings_model",
        "chunk_embeddings",
        ["model_name", "model_revision", "dimension"],
    )
    op.create_index(
        "ix_chunk_embeddings_embedding_hnsw",
        "chunk_embeddings",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_chunk_embeddings_embedding_hnsw", table_name="chunk_embeddings")
    op.drop_index("ix_chunk_embeddings_model", table_name="chunk_embeddings")
    op.drop_index("ix_chunk_embeddings_patient_document", table_name="chunk_embeddings")
    op.drop_index("ix_chunk_embeddings_document_id", table_name="chunk_embeddings")
    op.drop_index("ix_chunk_embeddings_patient_id", table_name="chunk_embeddings")
    op.drop_table("chunk_embeddings")
