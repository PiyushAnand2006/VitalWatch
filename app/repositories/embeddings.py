"""Patient-scoped pgvector persistence for document chunk embeddings."""

from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import AppError
from app.models import ChunkEmbeddingModel, DocumentModel
from app.schemas.embeddings import (
    EMBEDDING_DIMENSION,
    ChunkMatch,
    EmbeddedChunk,
    EmbeddingModelInfo,
)


def _require_index_model(model: EmbeddingModelInfo) -> None:
    """The index column is fixed at 384 dimensions; anything else is incompatible."""

    if model.dimension != EMBEDDING_DIMENSION:
        raise AppError(
            409,
            "embedding_model_incompatible",
            "The configured embedding model does not match the vector index.",
        )


class ChunkEmbeddingRepository:
    """All vector reads and writes are filtered by server-supplied patient scope."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def replace_document(
        self,
        *,
        patient_id: str,
        document_id: UUID,
        chunks: Sequence[EmbeddedChunk],
        model: EmbeddingModelInfo,
    ) -> int:
        """Atomically replace one document's vectors, never mixing model identities."""

        _require_index_model(model)
        for chunk in chunks:
            if len(chunk.embedding) != model.dimension:
                raise AppError(
                    400,
                    "embedding_dimension_mismatch",
                    "A chunk vector does not match the declared embedding dimension.",
                )
        async with self._sessions() as session:
            document_id_found = await session.scalar(
                select(DocumentModel.id).where(
                    DocumentModel.patient_id == patient_id,
                    DocumentModel.id == document_id,
                )
            )
            if document_id_found is None:
                raise AppError(404, "document_not_found", "The document was not found.")
            await self._reject_incompatible_index(session, model)
            await session.execute(
                delete(ChunkEmbeddingModel).where(
                    ChunkEmbeddingModel.patient_id == patient_id,
                    ChunkEmbeddingModel.document_id == document_id,
                )
            )
            session.add_all(
                [
                    ChunkEmbeddingModel(
                        patient_id=patient_id,
                        document_id=document_id,
                        chunk_id=chunk.chunk_id,
                        content_hash=chunk.content_hash,
                        page_number=chunk.page_number,
                        section=chunk.section,
                        text=chunk.text,
                        method=chunk.method,
                        quality=chunk.quality,
                        model_name=model.name,
                        model_revision=model.revision,
                        dimension=model.dimension,
                        embedding=chunk.embedding,
                    )
                    for chunk in chunks
                ]
            )
            await session.commit()
        return len(chunks)

    async def search(
        self,
        *,
        patient_id: str,
        query_embedding: Sequence[float],
        model: EmbeddingModelInfo,
        top_k: int = 5,
        document_ids: Sequence[UUID] | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> list[ChunkMatch]:
        """Rank only this patient's compatible vectors by cosine distance."""

        _require_index_model(model)
        if len(query_embedding) != EMBEDDING_DIMENSION:
            raise AppError(
                400, "invalid_query_vector", "The query vector has an invalid dimension."
            )
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if document_ids is not None and len(document_ids) == 0:
            return []
        distance = ChunkEmbeddingModel.embedding.cosine_distance(list(query_embedding))
        statement = (
            select(ChunkEmbeddingModel, DocumentModel.created_at, distance.label("distance"))
            .join(DocumentModel, DocumentModel.id == ChunkEmbeddingModel.document_id)
            .where(
                ChunkEmbeddingModel.patient_id == patient_id,
                DocumentModel.patient_id == patient_id,
                ChunkEmbeddingModel.model_name == model.name,
                ChunkEmbeddingModel.model_revision == model.revision,
                ChunkEmbeddingModel.dimension == model.dimension,
            )
            .order_by(distance)
            .limit(top_k)
        )
        if document_ids is not None:
            statement = statement.where(ChunkEmbeddingModel.document_id.in_(document_ids))
        if created_from is not None:
            statement = statement.where(DocumentModel.created_at >= created_from)
        if created_to is not None:
            statement = statement.where(DocumentModel.created_at <= created_to)
        async with self._sessions() as session:
            rows = (await session.execute(statement)).all()
        matches: list[ChunkMatch] = []
        for record, document_created_at, raw_distance in rows:
            if not isinstance(record, ChunkEmbeddingModel):
                continue
            resolved_distance = float(cast(float, raw_distance))
            matches.append(
                ChunkMatch(
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    patient_id=record.patient_id,
                    page_number=record.page_number,
                    section=record.section,
                    text=record.text,
                    content_hash=record.content_hash,
                    method=record.method,
                    quality=record.quality,
                    distance=resolved_distance,
                    score=1.0 - resolved_distance,
                    document_created_at=cast(datetime, document_created_at),
                )
            )
        return matches

    async def delete_index(self) -> int:
        """Clear every vector for a controlled rebuild; never exposed to request handlers."""

        async with self._sessions() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(delete(ChunkEmbeddingModel)),
            )
            await session.commit()
        return int(result.rowcount or 0)

    async def delete_document(self, patient_id: str, document_id: UUID) -> int:
        """Remove every vector belonging to one authorized patient document."""

        async with self._sessions() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    delete(ChunkEmbeddingModel).where(
                        ChunkEmbeddingModel.patient_id == patient_id,
                        ChunkEmbeddingModel.document_id == document_id,
                    )
                ),
            )
            await session.commit()
        return int(result.rowcount or 0)

    async def count(self, patient_id: str, document_id: UUID | None = None) -> int:
        """Count indexed vectors for diagnostics and tests, scoped by patient."""

        where = [ChunkEmbeddingModel.patient_id == patient_id]
        if document_id is not None:
            where.append(ChunkEmbeddingModel.document_id == document_id)
        async with self._sessions() as session:
            total = await session.scalar(
                select(func.count()).select_from(ChunkEmbeddingModel).where(*where)
            )
        return int(total or 0)

    @staticmethod
    async def _reject_incompatible_index(
        session: AsyncSession, model: EmbeddingModelInfo
    ) -> None:
        """Refuse to mix a second model identity into the same vector index."""

        identities = (
            await session.execute(
                select(
                    ChunkEmbeddingModel.model_name,
                    ChunkEmbeddingModel.model_revision,
                    ChunkEmbeddingModel.dimension,
                ).distinct()
            )
        ).all()
        expected = (model.name, model.revision, model.dimension)
        if any(tuple(identity) != expected for identity in identities):
            raise AppError(
                409,
                "embedding_model_mismatch",
                "This index already contains vectors from a different embedding model.",
            )
