"""Synthetic-only tests for the local Stage 3 extractor."""

from io import BytesIO
from pathlib import Path

import fitz
import pytest
from docx import Document
from PIL import Image, ImageDraw, ImageFont

from app.core.exceptions import AppError
from app.schemas.extraction import ExtractionMethod
from app.services.extraction import DocumentExtractor, chunk_document

TESSERACT = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")


def _pdf_with_text(text: str) -> bytes:
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    data = pdf.tobytes()
    pdf.close()
    return data


def _image_with_text(text: str) -> bytes:
    image = Image.new("RGB", (900, 180), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 48)
    draw.text((20, 50), text, fill="black", font=font)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_extracts_native_pdf_text_with_high_quality() -> None:
    result = DocumentExtractor().extract("synthetic.pdf", _pdf_with_text("Sodium 140 mmol/L"))

    assert result.pages[0].method is ExtractionMethod.PDF_TEXT
    assert result.pages[0].quality == "high"
    assert "140 mmol/L" in result.pages[0].text


def test_ocrs_synthetic_image_with_uncertain_quality() -> None:
    extractor = DocumentExtractor(tesseract_command=TESSERACT)
    result = extractor.extract("synthetic.png", _image_with_text("SODIUM 140 mmol/L"))

    assert result.pages[0].method is ExtractionMethod.OCR
    assert result.pages[0].quality == "uncertain"
    assert "SODIUM 140" in result.pages[0].text


def test_extracts_docx_and_txt() -> None:
    docx = Document()
    docx.add_paragraph("Report heading")
    output = BytesIO()
    docx.save(output)
    extractor = DocumentExtractor()

    assert (
        extractor.extract("synthetic.txt", b"Potassium 4.0 mmol/L").pages[0].text.endswith("mmol/L")
    )
    assert extractor.extract("synthetic.docx", output.getvalue()).pages[0].text == "Report heading"


def test_rejects_malformed_pdf() -> None:
    with pytest.raises(AppError) as raised:
        DocumentExtractor().extract("synthetic.pdf", b"not a PDF")
    assert raised.value.code == "malformed_document"


def test_chunks_keep_page_method_and_content_hash() -> None:
    result = DocumentExtractor().extract(
        "synthetic.pdf",
        _pdf_with_text("Laboratory Results:\nSodium 140 mmol/L\nPotassium 4.0 mmol/L"),
    )
    chunks = chunk_document(result, chunk_size=32, overlap=8)

    assert len(chunks) >= 2
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert all(chunk.method is ExtractionMethod.PDF_TEXT for chunk in chunks)
    assert all(len(chunk.content_hash) == 64 for chunk in chunks)


def test_invalid_chunk_settings_are_rejected() -> None:
    result = DocumentExtractor().extract("synthetic.txt", b"Safe synthetic text")

    with pytest.raises(ValueError):
        chunk_document(result, chunk_size=20, overlap=20)
