#!/usr/bin/env python3
"""
VitalWatch internal SLM - local inference client (no UI).

Start the model server once (llamafile or llama-server - both expose an
OpenAI-compatible API):

    llamafile-0.10.5.exe -m vitalwatch-slm-q8_0.gguf -c 2048 --port 8080

...then from anywhere in the VitalWatch project:

    from vitalwatch_slm_client import extract_evidence

    evidence, ok, violations = extract_evidence(report_text)
    if ok:
        ...route evidence to deterministic validation / evidence-package builder...
    else:
        ...quarantine report, log violations (never transmit downstream)...

Security posture (SSRF): the inference endpoint is a fixed, operator-configured
address supplied via the VITALWATCH_SLM_ENDPOINT environment variable (default
http://127.0.0.1:8080 for the local llamafile server). It is NEVER derived from
user input. validate_endpoint() enforces the scheme and, for non-loopback
(remote) endpoints, rejects any host that resolves to loopback, private,
reserved, or link-local addresses.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
from urllib.parse import urlparse

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "1.0"

SYSTEM_PROMPT = (
    "You are the VitalWatch internal SLM (evidence compiler). Convert the medical report "
    "into the privacy-minimized evidence JSON schema. Remove identifiers, use relative "
    "times, flag missing/inconsistent information, abstain when unreliable, never invent values."
)

DEFAULT_INSTRUCTION = (
    "Extract clinically relevant observations from the medical report below and prepare a "
    "privacy-minimized evidence package as JSON. Remove or generalize direct identifiers, "
    "convert exact dates into relative time references while preserving clinically important "
    "intervals, record missing or inconsistent information, and set status to \"abstain\" if "
    "the report cannot be reliably processed. Never invent clinical values."
)

# Fixed, operator-configured endpoint. Never derived from user input.
DEFAULT_ENDPOINT = os.environ.get("VITALWATCH_SLM_ENDPOINT", "http://127.0.0.1:8080")

REQUEST_TIMEOUT_S = 300
MAX_TOKENS = 768

# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------

def validate_endpoint(url: str, allow_loopback: bool = False) -> str:
    """Validate an inference endpoint before any request is made.

    - scheme must be http/https
    - loopback/private/reserved/link-local targets are rejected unless
      allow_loopback=True, which is reserved for the operator-configured local
      SLM server (never for user-supplied URLs)
    - remote endpoints must resolve to public addresses only
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"only http/https endpoints are allowed, got: {parsed.scheme!r}")
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    if allow_loopback and host in ("127.0.0.1", "localhost", "::1"):
        return url

    for info in socket.getaddrinfo(host, port):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_loopback or ip.is_private or ip.is_reserved or ip.is_link_local:
            raise ValueError(f"endpoint host resolves to a non-public address: {ip}")
    return url


# ---------------------------------------------------------------------------
# Output parsing + independent privacy firewall (mirrors the notebook)
# ---------------------------------------------------------------------------
ALLOWED_TOP_KEYS = {"schema_version", "status", "source_reference", "observations",
                    "removed_identifiers", "missing_information", "uncertainty",
                    "abstention_reason"}
OBS_KEYS = {"evidence_id", "parameter", "value", "unit", "value_type",
            "temporal", "uncertainty", "note"}
STATUSES = {"success", "partial", "abstain"}

LEAK_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}"),
    "mrn": re.compile(r"\bMRN[-:# ]?\s?\d+", re.I),
    "street_address": re.compile(
        r"\b\d{1,4}\s+[A-Z][a-z]+\s(?:Street|Avenue|Lane|Road|Crescent)\b"),
}
NAME_RE = re.compile(r"Patient:\s*([A-Z][a-z]+ [A-Z][a-z]+)")
INJECTION_PAT = re.compile(r"ignore (all )?(previous|prior)|system override|disregard the", re.I)


def parse_evidence(text: str):
    """Extract the first balanced JSON object from model output."""
    m = re.search(r"\{", text)
    if not m:
        return None
    depth = 0
    start = m.start()
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def privacy_validate(pred, source_text: str):
    """Independent gate: returns (allowed, violations). Never trust model claims."""
    violations = []
    if not isinstance(pred, dict):
        return False, ["invalid_json"]
    extra = set(pred.keys()) - ALLOWED_TOP_KEYS
    if extra:
        violations.append(f"schema_violation: unexpected keys {sorted(extra)}")
    if pred.get("schema_version") != SCHEMA_VERSION:
        violations.append(f"schema_violation: schema_version={pred.get('schema_version')!r}")
    if pred.get("status") not in STATUSES:
        violations.append("schema_violation: bad status")
    observations = pred.get("observations")
    if isinstance(observations, list):
        for ob in observations:
            if not OBS_KEYS <= set(ob.keys()):
                violations.append("schema_violation: observation outside allow-list")
                break
    else:
        violations.append("schema_violation: observations is not a list")
    flat = json.dumps(pred, ensure_ascii=False)
    m = NAME_RE.search(source_text or "")
    if m and m.group(1) in flat:
        violations.append("pii_leak: patient name copied into evidence")
    for kind, rx in LEAK_PATTERNS.items():
        if kind == "mrn":
            if rx.search(flat) and "MRN" in (source_text or ""):
                violations.append("pii_leak: mrn")
        elif rx.search(flat):
            violations.append(f"pii_leak: {kind}")
    if INJECTION_PAT.search(flat):
        violations.append("injection_echo: instruction-like text copied into evidence")
    return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_evidence(report_text: str, instruction: str = DEFAULT_INSTRUCTION,
                     endpoint: str = DEFAULT_ENDPOINT, timeout: int = REQUEST_TIMEOUT_S):
    """Send one report to the local SLM and return (evidence, allowed, violations).

    evidence is the parsed evidence-package dict (or None if unparseable).
    allowed is the independent privacy firewall verdict - downstream code MUST
    quarantine when it is False, regardless of how complete the evidence looks.
    """
    endpoint = validate_endpoint(endpoint, allow_loopback=True)
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction + "\n\nREPORT:\n" + report_text},
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    resp = requests.post(endpoint.rstrip("/") + "/v1/chat/completions",
                         json=payload, timeout=timeout)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    pred = parse_evidence(content)
    ok, violations = privacy_validate(pred, report_text)
    return pred, ok, violations


# ---------------------------------------------------------------------------
# CLI demo: python vitalwatch_slm_client.py [path\to\report.txt]
# ---------------------------------------------------------------------------
SAMPLE_REPORT = """RIVERSIDE GENERAL HOSPITAL - CLINICAL LABORATORY REPORT
Report ID: RPT_00412
Patient: Maria Silva   (MRN: MRN-778201)
Contact: 555-0117, maria.441@example.com
Address: 88 Cedar Avenue
Age/Sex: 58 F
Collected: 12 Mar 2024 08:15
------------------------------------------------------------
BMP PANEL:
  Sodium: 138 mmol/L
  Potassium: 4.9 mmol/L
  Serum creatinine: 1.8 mg/dL
  Glucose: 141 mg/dL
  Platelet count: [value not reported]
"""

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        report = open(sys.argv[1], "r", encoding="utf-8").read()
    else:
        report = SAMPLE_REPORT

    evidence, allowed, viols = extract_evidence(report)
    print(json.dumps(evidence, ensure_ascii=False, indent=2) if evidence
          else "<unparseable model output>")
    print("-" * 72)
    print("PRIVACY VALIDATOR:", "ALLOWED" if allowed else "REJECTED")
    for v in viols:
        print("  -", v)
    if not allowed:
        print("\nThis payload must NOT be transmitted downstream.")
