"""Validation and private persistence for untrusted uploaded originals."""

import hashlib
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.core.exceptions import AppError

_TYPES: dict[str, tuple[str, tuple[str, ...]]] = {
    ".pdf": ("application/pdf", ("application/pdf",)),
    ".txt": ("text/plain", ("text/plain",)),
    ".png": ("image/png", ("image/png",)),
    ".jpg": ("image/jpeg", ("image/jpeg",)),
    ".jpeg": ("image/jpeg", ("image/jpeg",)),
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",),
    ),
}


@dataclass(frozen=True)
class ValidatedUpload:
    source_name: str
    suffix: str
    media_type: str
    checksum: str
    data: bytes


class UploadValidator:
    def __init__(self, max_bytes: int, max_pages: int) -> None:
        self._max_bytes = max_bytes
        self._max_pages = max_pages

    def validate(
        self, source_name: str | None, media_type: str | None, data: bytes
    ) -> ValidatedUpload:
        safe_name = Path(source_name or "").name
        if safe_name != source_name or not safe_name or len(safe_name) > 255:
            raise AppError(400, "invalid_filename", "The uploaded filename is invalid.")
        suffix = Path(safe_name).suffix.lower()
        if suffix not in _TYPES:
            raise AppError(415, "unsupported_file_type", "This document type is not supported.")
        expected_type, acceptable_types = _TYPES[suffix]
        if media_type and media_type.lower().split(";", 1)[0] not in acceptable_types:
            raise AppError(
                415, "media_type_mismatch", "The declared document type does not match the file."
            )
        if not data or len(data) > self._max_bytes:
            raise AppError(
                413, "upload_too_large", "The uploaded document exceeds the allowed size."
            )
        self._validate_signature(suffix, data)
        return ValidatedUpload(
            safe_name, suffix, expected_type, hashlib.sha256(data).hexdigest(), data
        )

    def _validate_signature(self, suffix: str, data: bytes) -> None:
        valid = (
            (suffix == ".pdf" and data.startswith(b"%PDF-"))
            or (suffix == ".png" and data.startswith(b"\x89PNG\r\n\x1a\n"))
            or (suffix in {".jpg", ".jpeg"} and data.startswith(b"\xff\xd8\xff"))
            or (suffix == ".txt" and self._is_text(data))
            or (suffix == ".docx" and self._is_safe_docx(data))
        )
        if not valid:
            raise AppError(
                400, "malformed_document", "The uploaded document is malformed or unsafe."
            )
        if suffix == ".pdf" and self._pdf_page_count(data) > self._max_pages:
            raise AppError(
                413, "document_page_limit", "The uploaded document exceeds the page limit."
            )

    @staticmethod
    def _is_text(data: bytes) -> bool:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return b"\x00" not in data

    @staticmethod
    def _is_safe_docx(data: bytes) -> bool:
        try:
            with zipfile.ZipFile(__import__("io").BytesIO(data)) as archive:
                entries = archive.infolist()
                if len(entries) > 2_000 or "word/document.xml" not in archive.namelist():
                    return False
                if sum(item.file_size for item in entries) > 50 * 1024 * 1024:
                    return False
                return all(
                    not item.filename.lower().endswith((".exe", ".dll", ".js", ".vbs"))
                    for item in entries
                )
        except zipfile.BadZipFile:
            return False

    @staticmethod
    def _pdf_page_count(data: bytes) -> int:
        return len(re.findall(rb"/Type\s*/Page\b", data))


def generated_storage_key(suffix: str) -> str:
    return f"{uuid4().hex}{suffix}"
