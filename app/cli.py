"""Local operator commands; nothing here is exposed through HTTP."""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from app.core.config import get_settings
from app.core.database import get_database_manager
from app.core.event_loop import ensure_windows_selector_event_loop
from app.repositories.documents import DocumentRepository
from app.repositories.embeddings import ChunkEmbeddingRepository
from app.services.document_storage import LocalDocumentStorage
from app.services.embeddings import LocalEmbeddingService
from app.services.index_maintenance import IndexRebuildService
from app.services.indexing import DocumentIndexer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vitalwatch", description="VitalWatch local operations")
    commands = parser.add_subparsers(dest="command", required=True)
    rebuild = commands.add_parser(
        "rebuild-index",
        help="Clear the vector index and re-embed every processed document",
    )
    rebuild.add_argument(
        "--yes",
        action="store_true",
        help="Required confirmation; the rebuild deletes every existing chunk vector first",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "rebuild-index":
        if not args.yes:
            print(
                "Refusing to clear the vector index without --yes.",
                file=sys.stderr,
            )
            return 2
        ensure_windows_selector_event_loop()
        return asyncio.run(_rebuild_index())
    return 2


async def _rebuild_index() -> int:
    settings = get_settings()
    database = get_database_manager(settings)
    try:
        storage = LocalDocumentStorage(settings.document_storage_directory)
        embeddings = LocalEmbeddingService(
            model_name=settings.embedding_model_name,
            revision=settings.embedding_model_revision,
            cache_directory=settings.embedding_model_directory,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
            local_files_only=settings.embedding_local_files_only,
        )
        vectors = ChunkEmbeddingRepository(database.sessionmaker)
        service = IndexRebuildService(
            DocumentRepository(database.sessionmaker),
            vectors,
            DocumentIndexer(vectors, storage, embeddings),
            embeddings,
        )
        report = await service.rebuild()
    finally:
        await database.close()
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 1 if report.failed_document_ids else 0


if __name__ == "__main__":
    raise SystemExit(main())
