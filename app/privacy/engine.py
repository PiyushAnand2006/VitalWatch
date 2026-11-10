"""Fail-closed Microsoft Presidio analyzer adapter.

Presidio and its spaCy model are loaded lazily so the application can start without
them. Any initialization or analysis failure is surfaced as a typed error that the
privacy gate converts into a rejection with zero outbound calls.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from app.privacy.recognizers import build_custom_recognizers

logger = logging.getLogger(__name__)

LANGUAGE = "en"

# These are the only entities the privacy pipeline will ever detect and tokenize.
ALLOWED_ENTITIES: frozenset[str] = frozenset(
    {
        "ADDRESS",
        "AGE",
        "CREDIT_CARD",
        "DATE_OF_BIRTH",
        "DATE_TIME",
        "EMAIL",
        "EMAIL_ADDRESS",
        "FACILITY",
        "GOVERNMENT_ID",
        "IBAN_CODE",
        "ID",
        "INTERNAL_ID",
        "IP_ADDRESS",
        "LOCATION",
        "MEDICAL_ID",
        "MEDICAL_LICENSE",
        "MEDICAL_RECORD_NUMBER",
        "NRP",
        "ORGANIZATION",
        "PERSON",
        "PHONE_NUMBER",
        "UK_NHS",
        "URL",
        "US_BANK_NUMBER",
        "US_DRIVER_LICENSE",
        "US_ITIN",
        "US_PASSPORT",
        "US_SSN",
    }
)


class PiiAnalyzerUnavailable(RuntimeError):
    """Presidio or its local NLP model could not be initialized."""


class PiiAnalysisError(RuntimeError):
    """Presidio failed while analyzing text."""


@dataclass(frozen=True)
class PiiEntity:
    """One detected identifier span in the source text."""

    entity_type: str
    start: int
    end: int
    score: float


class PiiAnalyzer(Protocol):
    """Minimal synchronous analyzer contract used by the privacy gate."""

    def analyze(self, text: str) -> list[PiiEntity]:
        """Return identifier spans for the given untrusted text."""

        ...


class PresidioPiiAnalyzer:
    """Thread-safe, lazily initialized Presidio analyzer."""

    def __init__(
        self,
        *,
        model_name: str = "en_core_web_sm",
        score_threshold: float = 0.5,
    ) -> None:
        self._model_name = model_name
        self._score_threshold = score_threshold
        self._engine: Any | None = None
        self._supported_entities: frozenset[str] = frozenset()
        self._lock = threading.Lock()

    def analyze(self, text: str) -> list[PiiEntity]:
        """Analyze text, raising typed errors instead of leaking exception details."""

        if not text.strip():
            return []
        try:
            engine, supported = self._get_engine()
        except PiiAnalyzerUnavailable:
            raise
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise PiiAnalyzerUnavailable("Presidio initialization failed") from exc
        try:
            results = engine.analyze(
                text=text,
                language=LANGUAGE,
                entities=sorted(supported),
                score_threshold=self._score_threshold,
            )
        except Exception as exc:
            raise PiiAnalysisError("Presidio analysis failed") from exc
        return [
            PiiEntity(
                entity_type=str(result.entity_type),
                start=int(result.start),
                end=int(result.end),
                score=float(result.score),
            )
            for result in results
        ]

    def _get_engine(self) -> tuple[Any, frozenset[str]]:
        if self._engine is not None:
            return self._engine, self._supported_entities
        with self._lock:
            if self._engine is not None:
                return self._engine, self._supported_entities
            engine, supported = self._build_engine()
            self._engine = engine
            self._supported_entities = supported
            logger.info("presidio analyzer loaded model=%s", self._model_name)
            return engine, supported

    def _build_engine(self) -> tuple[Any, frozenset[str]]:
        try:
            from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
            from presidio_analyzer.nlp_engine import NlpEngineProvider
        except Exception as exc:
            raise PiiAnalyzerUnavailable("Presidio is not installed") from exc

        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": LANGUAGE, "model_name": self._model_name}],
        }
        try:
            nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
            registry = RecognizerRegistry(supported_languages=[LANGUAGE])
            registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=[LANGUAGE])
        except Exception as exc:
            raise PiiAnalyzerUnavailable(
                "Presidio NLP model is unavailable; install the configured spaCy model"
            ) from exc

        for recognizer in build_custom_recognizers():
            registry.add_recognizer(recognizer)

        supported = frozenset(
            entity
            for recognizer in registry.recognizers
            for entity in recognizer.supported_entities
            if entity in ALLOWED_ENTITIES
        )
        if not supported:
            raise PiiAnalyzerUnavailable("Presidio has no usable recognizers")
        return (
            AnalyzerEngine(
                nlp_engine=nlp_engine,
                registry=registry,
                supported_languages=[LANGUAGE],
            ),
            supported,
        )
