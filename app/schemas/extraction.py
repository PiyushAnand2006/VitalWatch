"""Small, explicit contracts for local document extraction and chunks."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ExtractionMethod(StrEnum):
    PDF_TEXT = "pdf_text"
    OCR = "ocr"
    TEXT = "text"
    DOCX = "docx"


class ExtractedPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    page_number: int = Field(ge=1)
    text: str
    method: ExtractionMethod
    quality: str = Field(pattern=r"^(high|uncertain)$")


class ExtractedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pages: list[ExtractedPage]


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    page_number: int = Field(ge=1)
    section: str | None = None
    text: str
    method: ExtractionMethod
    quality: str = Field(pattern=r"^(high|uncertain)$")
