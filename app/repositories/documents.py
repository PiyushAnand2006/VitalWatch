"""PostgreSQL repositories for grants and document metadata."""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import ChunkEmbeddingModel, DocumentModel, PatientGrantModel
from app.schemas.documents import DocumentStatus, DocumentSummary


def document_summary(document: DocumentModel) -> DocumentSummary:
    return DocumentSummary(
        id=document.id,
        patient_id=document.patient_id,
        source_name=document.source_name,
        media_type=document.media_type,
        size_bytes=document.size_bytes,
        checksum=document.checksum,
        status=DocumentStatus(document.status),
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


class PatientGrantRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def allows(self, user_id: str, patient_id: str) -> bool:
        async with self._sessions() as session:
            statement = select(PatientGrantModel.id).where(
                PatientGrantModel.user_id == user_id, PatientGrantModel.patient_id == patient_id
            )
            return (await session.scalar(statement)) is not None


class DocumentRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(self, document: DocumentModel) -> DocumentSummary:
        async with self._sessions() as session:
            session.add(document)
            await session.commit()
            await session.refresh(document)
            return document_summary(document)

    async def find_duplicate(self, patient_id: str, checksum: str) -> DocumentSummary | None:
        async with self._sessions() as session:
            result = await session.scalar(
                select(DocumentModel).where(
                    DocumentModel.patient_id == patient_id, DocumentModel.checksum == checksum
                )
            )
            return document_summary(result) if result is not None else None

    async def get(self, patient_id: str, document_id: UUID) -> tuple[DocumentSummary, str] | None:
        async with self._sessions() as session:
            result = await session.scalar(
                select(DocumentModel).where(
                    DocumentModel.patient_id == patient_id, DocumentModel.id == document_id
                )
            )
            return (document_summary(result), result.storage_key) if result is not None else None

    async def list(
        self, patient_id: str, page: int, page_size: int
    ) -> tuple[Sequence[DocumentSummary], int]:
        async with self._sessions() as session:
            where = DocumentModel.patient_id == patient_id
            total = int(
                await session.scalar(select(func.count()).select_from(DocumentModel).where(where))
                or 0
            )
            records = (
                await session.scalars(
                    select(DocumentModel)
                    .where(where)
                    .order_by(DocumentModel.created_at.desc())
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                )
            ).all()
            return [document_summary(record) for record in records], total

    async def update_status(
        self, patient_id: str, document_id: UUID, status: DocumentStatus
    ) -> DocumentSummary | None:
        async with self._sessions() as session:
            record = await session.scalar(
                select(DocumentModel).where(
                    DocumentModel.patient_id == patient_id, DocumentModel.id == document_id
                )
            )
            if record is None:
                return None
            record.status = status.value
            await session.commit()
            await session.refresh(record)
            return document_summary(record)

    async def list_processed(
        self, limit: int = 100, after: UUID | None = None
    ) -> Sequence[tuple[str, UUID]]:
        """Keyset-paginate processed documents for a controlled full index rebuild."""

        statement = (
            select(DocumentModel.patient_id, DocumentModel.id)
            .where(DocumentModel.status == DocumentStatus.PROCESSED.value)
            .order_by(DocumentModel.id)
            .limit(limit)
        )
        if after is not None:
            statement = statement.where(DocumentModel.id > after)
        async with self._sessions() as session:
            rows = (await session.execute(statement)).all()
        return [(str(patient_id), document_id) for patient_id, document_id in rows]

    async def delete(self, patient_id: str, document_id: UUID) -> str | None:
        async with self._sessions() as session:
            record = await session.scalar(
                select(DocumentModel).where(
                    DocumentModel.patient_id == patient_id, DocumentModel.id == document_id
                )
            )
            if record is None:
                return None
            storage_key = record.storage_key
            # Vectors are removed in the same transaction so no orphaned index rows remain.
            await session.execute(
                delete(ChunkEmbeddingModel).where(
                    ChunkEmbeddingModel.patient_id == patient_id,
                    ChunkEmbeddingModel.document_id == document_id,
                )
            )
            await session.delete(record)
            await session.commit()
            return storage_key
