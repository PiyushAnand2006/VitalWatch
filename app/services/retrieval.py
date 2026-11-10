"""Patient-scoped retrieval over the local pgvector index, with typed no-evidence results."""

from collections.abc import Sequence
from datetime import datetime
from difflib import SequenceMatcher
from typing import Protocol
from uuid import UUID

from pydantic import ValidationError

from app.core.exceptions import AppError
from app.schemas.embeddings import ChunkMatch, EmbeddingModelInfo
from app.schemas.retrieval import (
    NoEvidenceReason,
    RetrievalFilter,
    RetrievalResult,
    RetrievalStatus,
    RetrievedChunk,
)
from app.services.embeddings import EmbeddingService

_CANDIDATE_MULTIPLIER = 4
_MAX_CANDIDATES = 200
_MIN_OVERLAP_CHARS = 80


class RetrievalRepository(Protocol):
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
    ) -> list[ChunkMatch]: ...


class RetrievalService:
    """Embed the query locally, retrieve authorized chunks, and apply deterministic policy."""

    def __init__(
        self,
        repository: RetrievalRepository,
        embeddings: EmbeddingService,
        *,
        top_k: int = 5,
        min_score: float = 0.25,
        max_query_chars: int = 2000,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        if max_query_chars < 1:
            raise ValueError("max_query_chars must be positive")
        self._repository = repository
        self._embeddings = embeddings
        self._top_k = top_k
        self._min_score = min_score
        self._max_query_chars = max_query_chars

    async def retrieve(
        self,
        *,
        patient_id: str,
        query: str,
        as_of: datetime | None = None,
        document_ids: Sequence[UUID] | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> RetrievalResult:
        """Retrieve only this patient's chunks; text remains untrusted evidence."""

        cleaned = query.strip()
        if not cleaned:
            raise AppError(400, "invalid_query", "A non-empty query is required.")
        if len(cleaned) > self._max_query_chars:
            raise AppError(400, "query_too_long", "The query is too long.")
        try:
            filters = RetrievalFilter(
                as_of=as_of,
                document_ids=tuple(document_ids) if document_ids is not None else None,
                created_from=created_from,
                created_to=created_to,
            )
        except ValidationError:
            raise AppError(
                400, "invalid_retrieval_filter", "The retrieval filters are invalid."
            ) from None
        availability_boundary = filters.created_to
        if filters.as_of is not None:
            availability_boundary = (
                filters.as_of
                if availability_boundary is None
                else min(availability_boundary, filters.as_of)
            )
        model = await self._embeddings.ensure_loaded()
        if filters.document_ids is not None and len(filters.document_ids) == 0:
            matches: list[ChunkMatch] = []
        else:
            query_embedding = await self._embeddings.embed_query(cleaned)
            matches = await self._repository.search(
                patient_id=patient_id,
                query_embedding=query_embedding,
                model=model,
                top_k=min(self._top_k * _CANDIDATE_MULTIPLIER, _MAX_CANDIDATES),
                document_ids=filters.document_ids,
                created_from=filters.created_from,
                created_to=availability_boundary,
            )
        unique, duplicates_removed = _deduplicate(matches)
        above_threshold = [match for match in unique if match.score >= self._min_score]
        if not above_threshold:
            return RetrievalResult(
                query=cleaned,
                patient_id=patient_id,
                status=RetrievalStatus.NO_EVIDENCE,
                reason=(
                    NoEvidenceReason.BELOW_RELEVANCE_THRESHOLD
                    if unique
                    else NoEvidenceReason.NO_MATCHING_DOCUMENTS
                ),
                items=[],
                candidates_considered=len(matches),
                duplicates_removed=duplicates_removed,
                below_threshold=len(unique),
                top_k=self._top_k,
                min_score=self._min_score,
                model=model,
                filters=filters,
            )
        items = [
            RetrievedChunk(
                chunk_id=match.chunk_id,
                document_id=match.document_id,
                patient_id=match.patient_id,
                page_number=match.page_number,
                section=match.section,
                text=match.text,
                content_hash=match.content_hash,
                method=match.method,
                quality=match.quality,
                score=match.score,
                distance=match.distance,
                document_created_at=match.document_created_at,
            )
            for match in above_threshold[: self._top_k]
        ]
        return RetrievalResult(
            query=cleaned,
            patient_id=patient_id,
            status=RetrievalStatus.RETRIEVED,
            items=items,
            candidates_considered=len(matches),
            duplicates_removed=duplicates_removed,
            below_threshold=len(unique) - len(above_threshold),
            top_k=self._top_k,
            min_score=self._min_score,
            model=model,
            filters=filters,
        )



def _deduplicate(matches: Sequence[ChunkMatch]) -> tuple[list[ChunkMatch], int]:
    """Drop lower-ranked chunks whose content overlaps a better-ranked sibling."""

    kept: list[ChunkMatch] = []
    for candidate in matches:
        if any(_overlaps(candidate, existing) for existing in kept):
            continue
        kept.append(candidate)
    return kept, len(matches) - len(kept)


def _overlaps(left: ChunkMatch, right: ChunkMatch) -> bool:
    if left.document_id != right.document_id or left.page_number != right.page_number:
        return False
    if left.content_hash == right.content_hash:
        return True
    left_text, right_text = left.text.strip(), right.text.strip()
    if not left_text or not right_text:
        return left_text == right_text
    if left_text in right_text or right_text in left_text:
        return True
    matcher = SequenceMatcher(None, left_text, right_text, autojunk=False)
    match = matcher.find_longest_match(0, len(left_text), 0, len(right_text))
    return match.size >= _MIN_OVERLAP_CHARS
