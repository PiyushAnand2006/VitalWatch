"""Lifecycle tests for the extraction pipeline using only synthetic data."""

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.schemas.documents import DocumentStatus, DocumentSummary
from app.schemas.extraction import ExtractedDocument, ExtractedPage, ExtractionMethod
from app.services.document_processing import DocumentProcessor


class FakeRepository:
    def __init__(self, document: DocumentSummary) -> None:
        self.document = document
        self.statuses: list[DocumentStatus] = []

    async def get(self, patient_id: str, document_id: object) -> tuple[DocumentSummary, str]:
        return self.document, "original.txt"

    async def update_status(
        self, patient_id: str, document_id: object, status: DocumentStatus
    ) -> DocumentSummary:
        self.statuses.append(status)
        return self.document


class FakeStorage:
    def __init__(self, original: bytes) -> None:
        self.original = original
        self.written: dict[str, bytes] = {}

    def retrieve(self, key: str) -> bytes:
        return self.original

    def store(self, key: str, data: bytes) -> str:
        self.written[key] = data
        return key


class FakeExtractor:
    async def extract_async(self, filename: str, data: bytes) -> ExtractedDocument:
        if filename.endswith(".pdf"):
            raise ValueError("synthetic extraction failure")
        return ExtractedDocument(
            pages=[
                ExtractedPage(
                    page_number=1,
                    text="Laboratory Results:\nSodium 140 mmol/L",
                    method=ExtractionMethod.TEXT,
                    quality="high",
                )
            ]
        )


def _document() -> DocumentSummary:
    now = datetime.now(UTC)
    return DocumentSummary(
        id=uuid4(),
        patient_id="synthetic-patient",
        source_name="synthetic.txt",
        media_type="text/plain",
        size_bytes=20,
        checksum="a" * 64,
        status=DocumentStatus.UPLOADED,
        created_at=now,
        updated_at=now,
    )


def _run_immediate(coroutine: object) -> None:
    """Run a coroutine whose test doubles have no suspension point.

    The local Windows Python event-loop policy cannot create an asyncio loop,
    so lifecycle tests keep I/O fake and execute the coroutine directly.
    """

    with pytest.raises(StopIteration):
        coroutine.send(None)  # type: ignore[union-attr]


def test_processor_marks_document_processed_and_stores_private_result() -> None:
    document = _document()
    repository = FakeRepository(document)
    storage = FakeStorage(b"Laboratory Results:\nSodium 140 mmol/L")

    _run_immediate(
        DocumentProcessor(repository, storage, FakeExtractor()).process(  # type: ignore[arg-type]
            document.patient_id, document.id
        )
    )

    assert repository.statuses == [DocumentStatus.PROCESSING, DocumentStatus.PROCESSED]
    saved = json.loads(storage.written[f"{document.id}.extraction.json"])
    assert saved["pages"][0]["method"] == "text"
    assert saved["chunks"][0]["page_number"] == 1
    assert len(saved["chunks"][0]["content_hash"]) == 64


def test_processor_marks_document_failed_when_extraction_fails() -> None:
    document = _document().model_copy(update={"source_name": "synthetic.pdf"})
    repository = FakeRepository(document)
    storage = FakeStorage(b"not a PDF")

    _run_immediate(
        DocumentProcessor(repository, storage, FakeExtractor()).process(  # type: ignore[arg-type]
            document.patient_id, document.id
        )
    )

    assert repository.statuses == [DocumentStatus.PROCESSING, DocumentStatus.FAILED]
