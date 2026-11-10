"""Document storage abstraction supporting local filesystem and future S3."""

from abc import ABC, abstractmethod
from pathlib import Path


class DocumentStorage(ABC):
    """Abstract base class for document storage backends."""

    @abstractmethod
    def store(self, key: str, data: bytes) -> str:
        """Store document bytes under a given key and return the storage identifier."""
        pass

    @abstractmethod
    def retrieve(self, key: str) -> bytes:
        """Retrieve stored document bytes."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete stored document by key."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a document exists by key."""
        pass


class LocalDocumentStorage(DocumentStorage):
    """Local filesystem document storage with directory confinement."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, key: str) -> Path:
        # Prevent path traversal
        clean_key = Path(key).name
        return self.root_dir / clean_key

    def store(self, key: str, data: bytes) -> str:
        path = self._resolve_path(key)
        path.write_bytes(data)
        return str(path)

    def retrieve(self, key: str) -> bytes:
        path = self._resolve_path(key)
        if not path.is_file():
            raise FileNotFoundError(f"Document {key} not found")
        return path.read_bytes()

    def delete(self, key: str) -> bool:
        path = self._resolve_path(key)
        if path.is_file():
            path.unlink()
            return True
        return False

    def exists(self, key: str) -> bool:
        return self._resolve_path(key).is_file()
