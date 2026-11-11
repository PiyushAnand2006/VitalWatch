#!/usr/bin/env python3
"""
VitalWatch internal SLM - in-process GGUF predictor (llama-cpp-python, no server, no UI).

One-time setup:
    pip install llama-cpp-python
    (on Windows this may build from source; see the install notes at the bottom)

Get a GGUF: Kaggle Cell B converts the merged model - for this predictor the
smoothest variant is Q8_0 (no llama-quantize step needed):
    convert_hf_to_gguf.py /kaggle/working/vitalwatch-slm-merged \
        --outfile /kaggle/working/vitalwatch-slm-q8_0.gguf --outtype q8_0
(Q4_K_M made via llama-quantize works identically - just point MODEL_PATH at it.)

Usage - exactly the ML-model pattern, load once then predict per report:

    from vitalwatch_predictor import VitalWatchSLM

    slm = VitalWatchSLM(model_path="vitalwatch-slm-q8_0.gguf")   # loads once
    result = slm.predict(report_text)
    # result = {"evidence": {...} | None, "raw": ..., "allowed": bool, "violations": [...]}

Privacy note: every prediction carries the independent privacy-firewall verdict.
Downstream code MUST quarantine when result["allowed"] is False - fine-tuning
is not a privacy guarantee.
"""

from __future__ import annotations

MODEL_PATH_DEFAULT = "vitalwatch-slm-q8_0.gguf"
N_CTX = 2048
MAX_TOKENS = 768

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

# Shared output parser + privacy firewall (same rules as the notebook eval).
from vitalwatch_slm_client import parse_evidence, privacy_validate  # noqa: E402


class VitalWatchSLM:
    """GGUF evidence compiler loaded in-process (sklearn-style usage)."""

    def __init__(self, model_path: str = MODEL_PATH_DEFAULT, n_ctx: int = N_CTX,
                 max_tokens: int = MAX_TOKENS, verbose: bool = False):
        from llama_cpp import Llama

        self.max_tokens = max_tokens
        # n_threads defaults to all physical cores; n_gpu_layers=0 keeps it CPU.
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=verbose)

    def predict(self, report_text: str, instruction: str = DEFAULT_INSTRUCTION) -> dict:
        """One report -> {"evidence", "raw", "allowed", "violations"}."""
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction + "\n\nREPORT:\n" + report_text},
            ],
            temperature=0,
            max_tokens=self.max_tokens,
        )
        raw = out["choices"][0]["message"]["content"] or ""
        evidence = parse_evidence(raw)
        allowed, violations = privacy_validate(evidence, report_text)
        return {"evidence": evidence, "raw": raw, "allowed": allowed, "violations": violations}


# ---------------------------------------------------------------------------
# Module-level singleton: load once per process, reuse everywhere.
# ---------------------------------------------------------------------------
_slm_instance = None


def get_slm(model_path: str = MODEL_PATH_DEFAULT) -> VitalWatchSLM:
    global _slm_instance
    if _slm_instance is None:
        _slm_instance = VitalWatchSLM(model_path=model_path)
    return _slm_instance


def predict(report_text: str, model_path: str = MODEL_PATH_DEFAULT) -> dict:
    """One-liner usage: predict(report_text) - loads the model on first call."""
    return get_slm(model_path).predict(report_text)


# ---------------------------------------------------------------------------
# CLI demo: python vitalwatch_predictor.py [path\to\report.txt]
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
    import json
    import sys

    if len(sys.argv) > 1:
        report = open(sys.argv[1], "r", encoding="utf-8").read()
    else:
        report = SAMPLE_REPORT

    result = predict(report)
    print(json.dumps(result["evidence"], ensure_ascii=False, indent=2)
          if result["evidence"] else "<unparseable model output>")
    print("-" * 72)
    print("PRIVACY VALIDATOR:", "ALLOWED" if result["allowed"] else "REJECTED")
    for v in result["violations"]:
        print("  -", v)
