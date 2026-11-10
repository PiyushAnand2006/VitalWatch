"""Microsoft Presidio recognizers for identifiers common in clinical documents."""

from typing import Any


def build_custom_recognizers() -> list[Any]:
    """Build deterministic recognizers that supplement Presidio's built-ins.

    Patterns intentionally over-detect rather than under-detect: a false positive is
    redacted, while a missed identifier would leave PHI in an outbound payload.
    """

    from presidio_analyzer import Pattern, PatternRecognizer

    return [
        PatternRecognizer(
            supported_entity="MEDICAL_RECORD_NUMBER",
            name="medical_record_number",
            supported_language="en",
            context=["mrn", "medical record", "hospital number", "record number"],
            patterns=[
                Pattern(
                    name="mrn_labeled",
                    regex=(
                        r"(?i)\b(?:MRN|medical\s+record(?:\s+(?:number|no\.?|#))?"
                        r"|hospital\s+(?:number|no\.?)|record\s+(?:number|no\.?))"
                        r"\s*[:#=-]?\s*[A-Z0-9][A-Z0-9-]{3,19}\b"
                    ),
                    score=0.85,
                ),
                Pattern(
                    name="mrn_bare_number",
                    regex=r"(?i)\bMRN\s*[:#=-]\s*\d{5,12}\b",
                    score=0.85,
                ),
            ],
        ),
        PatternRecognizer(
            supported_entity="GOVERNMENT_ID",
            name="government_identifier",
            supported_language="en",
            context=[
                "ssn",
                "social security",
                "passport",
                "aadhaar",
                "national id",
                "driver license",
                "driving licence",
                "tax id",
            ],
            patterns=[
                Pattern(name="ssn", regex=r"\b\d{3}-\d{2}-\d{4}\b", score=0.85),
                Pattern(
                    name="aadhaar",
                    regex=r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b",
                    score=0.45,
                ),
                Pattern(
                    name="passport",
                    regex=r"\b[A-Z]{1,2}\d{6,9}\b",
                    score=0.3,
                ),
                Pattern(
                    name="labeled_government_id",
                    regex=(
                        r"(?i)\b(?:government|national|tax|identity)\s+(?:id|number|no\.?)"
                        r"\s*[:#=-]?\s*[A-Z0-9][A-Z0-9-]{3,24}\b"
                    ),
                    score=0.75,
                ),
            ],
        ),
        PatternRecognizer(
            supported_entity="MEDICAL_ID",
            name="medical_identifier",
            supported_language="en",
            context=[
                "insurance",
                "policy",
                "member",
                "subscriber",
                "claim",
                "health plan",
                "medicare",
                "medicaid",
            ],
            patterns=[
                Pattern(
                    name="insurance_policy_identifier",
                    regex=(
                        r"(?i)\b(?:insurance\s+policy|policy|member|subscriber|claim"
                        r"|health\s+plan|medicare|medicaid)\s*(?:id|no\.?|number|#)?"
                        r"\s*[:#=-]\s*[A-Z0-9][A-Z0-9-]{4,24}\b"
                    ),
                    score=0.75,
                ),
                Pattern(
                    name="insurance_identifier",
                    regex=(
                        r"(?i)\b(?:insurance|medicare|medicaid)\s+(?:id|no\.?|number|#)"
                        r"\s*[:#=-]?\s*[A-Z0-9][A-Z0-9-]{4,24}\b"
                    ),
                    score=0.75,
                ),
            ],
        ),
        PatternRecognizer(
            supported_entity="INTERNAL_ID",
            name="internal_identifier",
            supported_language="en",
            context=[
                "patient id",
                "document id",
                "encounter",
                "accession",
                "order id",
                "report id",
            ],
            patterns=[
                Pattern(
                    name="labeled_internal_id",
                    regex=(
                        r"(?i)\b(?:patient|document|doc|chunk|encounter|visit|accession"
                        r"|order|report|user|record)\s*(?:id|no\.?|number|#)"
                        r"\s*[:#=-]?\s*[A-Za-z0-9][A-Za-z0-9_.:-]{2,63}\b"
                    ),
                    score=0.8,
                ),
                Pattern(
                    name="uuid",
                    regex=(
                        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}"
                        r"-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
                    ),
                    score=0.8,
                ),
                Pattern(
                    name="evidence_id",
                    regex=r"\bev_[a-f0-9]{24}\b",
                    score=0.85,
                ),
            ],
        ),
        PatternRecognizer(
            supported_entity="FACILITY",
            name="identifying_facility",
            supported_language="en",
            context=["hospital", "clinic", "medical center", "health center", "laboratory"],
            patterns=[
                Pattern(
                    name="facility_name",
                    regex=(
                        r"(?i)\b(?:[A-Za-z][A-Za-z&.'-]*\s+){0,4}"
                        r"(?:hospital|clinic|medical\s+center|health\s+center|healthcare"
                        r"|nursing\s+home|laboratory|lab|pharmacy|institute"
                        r"|medical\s+college)\b"
                    ),
                    score=0.6,
                )
            ],
        ),
        PatternRecognizer(
            supported_entity="ADDRESS",
            name="street_address",
            supported_language="en",
            context=["street", "avenue", "road", "lane", "drive", "boulevard", "apartment"],
            patterns=[
                Pattern(
                    name="street_address",
                    regex=(
                        r"(?i)\b\d{1,6}\s+(?:[A-Za-z0-9][A-Za-z0-9.'-]*\s+){0,5}"
                        r"(?:street|st\.?|avenue|ave\.?|road|rd\.?|lane|ln\.?|drive|dr\."
                        r"|boulevard|blvd\.?|court|ct\.?|place|pl\.?|way)\b"
                    ),
                    score=0.75,
                ),
                Pattern(
                    name="po_box",
                    regex=r"(?i)\bP\.?\s*O\.?\s*Box\s+\d{1,8}\b",
                    score=0.8,
                ),
            ],
        ),
        PatternRecognizer(
            supported_entity="DATE_OF_BIRTH",
            name="date_of_birth",
            supported_language="en",
            context=["dob", "date of birth", "born"],
            patterns=[
                Pattern(
                    name="labeled_dob",
                    regex=(
                        r"(?i)\b(?:DOB|D\.O\.B\.?|date\s+of\s+birth|born(?:\s+on)?)"
                        r"\s*[:#=-]?\s*(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}"
                        r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
                        r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
                        r"[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
                        r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
                        r"[a-z]*\.?,?\s+\d{4})\b"
                    ),
                    score=0.9,
                )
            ],
        ),
        PatternRecognizer(
            supported_entity="AGE",
            name="explicit_age",
            supported_language="en",
            context=["age", "years old", "year old", "y/o"],
            patterns=[
                Pattern(
                    name="age_with_unit",
                    regex=(
                        r"(?i)\b(?:age[d]?\s*[:=]?\s*)?\d{1,3}"
                        r"(?:\s*[- ]?(?:year[- ]old|years?|yrs?|y/o))\b"
                    ),
                    score=0.75,
                )
            ],
        ),
        PatternRecognizer(
            supported_entity="PERSON",
            name="honorific_name",
            supported_language="en",
            context=["mr", "mrs", "ms", "miss", "dr", "professor"],
            patterns=[
                Pattern(
                    name="honorific_name",
                    regex=(
                        r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+"
                        r"(?:\s+[A-Z][a-z]+){0,2}\b"
                    ),
                    score=0.65,
                )
            ],
        ),
    ]
