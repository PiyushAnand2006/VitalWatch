#!/usr/bin/env python3
"""
VitalWatch internal SLM - inference example.

Shows how the fine-tuned evidence compiler is used at runtime: medical report text in,
structured privacy-minimized evidence JSON out, followed by the independent privacy
validator verdict (which must pass before ANY downstream transmission).

Usage:
    python inference_example.py                          # bundled demo report
    python inference_example.py --adapter /path/to/adapter
    python inference_example.py --report my_report.txt   # any report text file
    python inference_example.py --base-model Qwen/Qwen2.5-1.5B-Instruct --adapter <hf-repo-or-path>

Requires: transformers, peft, torch (see requirements.txt). Runs on GPU (4-bit) or CPU (slow).
The model is decision-support ONLY: outputs must still pass the deterministic validator and
privacy firewall in the VitalWatch pipeline; this script is not a medical device.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SCHEMA_VERSION = "1.0"
ADAPTER_DEFAULT = "vitalwatch-slm/adapter"

SAMPLE_REPORT = """RIVERSIDE GENERAL HOSPITAL - CLINICAL LABORATORY REPORT
Report ID: RPT_00412
Patient: Maria Silva   (MRN: MRN-778201)
Contact: 555-0117, maria.441@example.com
Address: 88 Cedar Avenue
Age/Sex: 58 F
Collected: 12 Mar 2024 08:15    Reported: 12 Mar 2024 10:40
Ordering physician: Dr. M. Chen
------------------------------------------------------------
BMP PANEL:
  Sodium: 138 mmol/L
  Potassium: 4.9 mmol/L
  Serum creatinine: 1.8 mg/dL
  Glucose: 141 mg/dL
  Platelet count: [value not reported]
Note: sample mildly hemolyzed.
Next of kin: T. Silva (son), reachable at 555-0188.
"""

SYSTEM_PROMPT = (
    "You are the VitalWatch internal SLM (evidence compiler). Convert the medical report "
    "into the privacy-minimized evidence JSON schema. Remove identifiers, use relative "
    "times, flag missing/inconsistent information, abstain when unreliable, never invent values."
)

ALLOWED_TOP_KEYS = {"schema_version", "status", "source_reference", "observations",
                    "removed_identifiers", "missing_information", "uncertainty",
                    "abstention_reason"}
OBS_KEYS = {"evidence_id", "parameter", "value", "unit", "value_type",
            "temporal", "uncertainty", "note"}
LEAK_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}"),
    "mrn": re.compile(r"\bMRN[-:# ]?\s?\d+", re.I),
    "street_address": re.compile(r"\b\d{1,4}\s+[A-Z][a-z]+\s(?:Street|Avenue|Lane|Road|Crescent)\b"),
}
NAME_RE = re.compile(r"Patient:\s*([A-Z][a-z]+ [A-Z][a-z]+)")


def privacy_validate(pred, source_text):
    """Independent privacy gate: returns (allowed, violations). Never trust model claims."""
    violations = []
    if not isinstance(pred, dict):
        return False, ["invalid_json"]
    extra = set(pred.keys()) - ALLOWED_TOP_KEYS
    if extra:
        violations.append(f"schema_violation: unexpected keys {sorted(extra)}")
    if pred.get("schema_version") != SCHEMA_VERSION:
        violations.append(f"schema_violation: schema_version={pred.get('schema_version')!r}")
    for ob in pred.get("observations", []) if isinstance(pred.get("observations"), list) else []:
        if not {"evidence_id", "parameter", "value", "unit", "value_type",
                "temporal", "uncertainty", "note"} <= set(ob.keys()):
            violations.append("schema_violation: observation outside allow-list")
            break
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
    if re.search(r"ignore (all )?(previous|prior)|system override|disregard the", flat, re.I):
        violations.append("injection_echo: instruction-like text copied into evidence")
    return len(violations) == 0, violations


def parse_evidence(text):
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


def main():
    ap = argparse.ArgumentParser(description="VitalWatch internal SLM inference example")
    ap.add_argument("--adapter", type=str, default=ADAPTER_DEFAULT,
                    help="path to LoRA adapter directory or HF repo id")
    ap.add_argument("--base-model", type=str, default=BASE_MODEL)
    ap.add_argument("--report", type=str, default=None, help="optional path to a report text file")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_is_local = bool(args.adapter) and os.path.isdir(args.adapter)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter if adapter_is_local else args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    if args.report:
        report_text = Path(args.report).read_text(encoding="utf-8")
    else:
        report_text = SAMPLE_REPORT

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Extract clinically relevant observations and prepare a privacy-minimized "
            "evidence package as JSON.\n\nREPORT:\n" + report_text)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    pred = parse_evidence(raw)
    print("=" * 72)
    print("MODEL OUTPUT (raw, first 600 chars):")
    print(raw[:600])
    print("=" * 72)
    if pred is None:
        print("PARSED EVIDENCE: <unparseable output>")
    else:
        print("PARSED EVIDENCE:")
        print(json.dumps(pred, ensure_ascii=False, indent=2))
    ok, violations = privacy_validate(pred, report_text)
    print("=" * 72)
    print("PRIVACY VALIDATOR:", "ALLOWED" if ok else "REJECTED")
    for v in violations:
        print("  -", v)
    if not ok:
        print("\nThis payload must NOT be transmitted downstream.")


if __name__ == "__main__":
    main()
