"""Local, readable extraction helpers. They never interpret clinical values."""

import io
import re
from hashlib import sha256
from pathlib import Path

import fitz
import pytesseract
from anyio import to_thread
from docx import Document
from PIL import Image

from app.core.exceptions import AppError
from app.schemas.extraction import DocumentChunk, ExtractedDocument, ExtractedPage, ExtractionMethod


class DocumentExtractor:
    """Extract text locally; scanned pages use OCR and are always marked uncertain."""

    def __init__(self, language: str = "eng", tesseract_command: Path | None = None) -> None:
        self._language = language
        if tesseract_command is not None:
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_command)

    async def extract_async(self, filename: str, data: bytes) -> ExtractedDocument:
        """Run CPU-bound parsing outside FastAPI's async event loop."""

        return await to_thread.run_sync(self.extract, filename, data)

    def extract(self, filename: str, data: bytes) -> ExtractedDocument:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            return self._pdf(data)
        if suffix in {".png", ".jpg", ".jpeg"}:
            return self._image(data)
        if suffix == ".txt":
            return ExtractedDocument(
                pages=[
                    ExtractedPage(
                        page_number=1,
                        text=data.decode("utf-8"),
                        method=ExtractionMethod.TEXT,
                        quality="high",
                    )
                ]
            )
        if suffix == ".docx":
            return self._docx(data)
        raise AppError(415, "unsupported_file_type", "This document type is not supported.")

    def _pdf(self, data: bytes) -> ExtractedDocument:
        try:
            pdf = fitz.open(stream=data, filetype="pdf")
        except Exception as error:
            raise AppError(
                400, "malformed_document", "The uploaded document is malformed."
            ) from error
        pages: list[ExtractedPage] = []
        try:
            for number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append(
                        ExtractedPage(
                            page_number=number,
                            text=text,
                            method=ExtractionMethod.PDF_TEXT,
                            quality="high",
                        )
                    )
                else:
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                    pages.append(self._ocr_page(number, image))
        finally:
            pdf.close()
        return ExtractedDocument(pages=pages)

    def _image(self, data: bytes) -> ExtractedDocument:
        try:
            image = Image.open(io.BytesIO(data))
            return ExtractedDocument(pages=[self._ocr_page(1, image)])
        except OSError as error:
            raise AppError(400, "malformed_document", "The uploaded image is malformed.") from error

    def _ocr_page(self, number: int, image: Image.Image) -> ExtractedPage:
        try:
            text = pytesseract.image_to_string(image, lang=self._language).strip()
        except pytesseract.TesseractNotFoundError as error:
            raise AppError(503, "ocr_unavailable", "Local OCR is not available.") from error
        return ExtractedPage(
            page_number=number, text=text, method=ExtractionMethod.OCR, quality="uncertain"
        )

    @staticmethod
    def _docx(data: bytes) -> ExtractedDocument:
        try:
            document = Document(io.BytesIO(data))
        except Exception as error:
            raise AppError(
                400, "malformed_document", "The uploaded document is malformed."
            ) from error
        text = "\n".join(
            paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
        )
        return ExtractedDocument(
            pages=[
                ExtractedPage(
                    page_number=1, text=text, method=ExtractionMethod.DOCX, quality="high"
                )
            ]
        )


def chunk_document(
    document: ExtractedDocument, chunk_size: int = 800, overlap: int = 100
) -> list[DocumentChunk]:
    """Split each page at line boundaries while retaining its source provenance."""

    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")
    chunks: list[DocumentChunk] = []
    for page in document.pages:
        text = page.text.strip()
        start = 0
        index = 1
        while start < len(text):
            end = min(len(text), start + chunk_size)
            if end < len(text):
                line_break = text.rfind("\n", start, end)
                if line_break > start:
                    end = line_break
            part = text[start:end].strip()
            if part:
                first_line = part.splitlines()[0].strip()
                section = (
                    first_line.rstrip(":")
                    if re.fullmatch(r"[A-Za-z][A-Za-z ]{1,60}:?", first_line)
                    else None
                )
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"p{page.page_number}-c{index}",
                        content_hash=sha256(part.encode("utf-8")).hexdigest(),
                        page_number=page.page_number,
                        section=section,
                        text=part,
                        method=page.method,
                        quality=page.quality,
                    )
                )
                index += 1
            start = end if end == len(text) else end - overlap
    return chunks
