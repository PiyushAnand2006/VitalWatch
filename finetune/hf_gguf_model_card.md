---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- vitalwatch
- clinical-evidence
- gguf
- llama-cpp-python
- decision-support
language:
- en
---

# VitalWatch Internal SLM — GGUF deployment builds

Quantized GGUF builds of the VitalWatch internal SLM ("evidence compiler"): converts retrieved
medical report text into a **privacy-minimized evidence JSON package** (schema v1.0) for
downstream deterministic validation. Decision-support only — not a diagnostic device, not the
temporal prediction model, and not a privacy guarantee.

## Lineage (provenance)

| Artifact | Source |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0) |
| Fine-tuning | QLoRA r=16, 1,087 examples (ELMTEX CC-BY-4.0 derived + labeled synthetic), 204 steps / 3 epochs, final eval loss 0.440 |
| Adapter | `TheBlindCoder/vitalwatch-slm-qwen-1p5b-lora` @ revision `72577e8` (source of truth) |
| Merge | adapter merged into fp16 base via `merge_and_unload()` |
| Conversion | llama.cpp `convert_hf_to_gguf.py` |

## Files

| File | Size | Use |
|---|---|---|
| `vitalwatch-slm-f16.gguf` | ~3.1 GB | full-precision container; quantization source |
| `vitalwatch-slm-q8_0.gguf` | ~1.6 GB | recommended CPU deployment (near-lossless) |

## Evaluation (95 held-out examples, 768-token generation, T4)

Fine-tuned model via the HF load path (see adapter repo for the full table):

- JSON validity **0.979** · firewall-clean schema compliance **0.968**
- PII leakage **0.000** · abstention accuracy **1.000**
- field F1 **0.561** (precision 0.664 / recall 0.524)

## Usage (Python, in-process)

```python
# pip install llama-cpp-python
from llama_cpp import Llama

llm = Llama(model_path="vitalwatch-slm-q8_0.gguf", n_ctx=2048)
out = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are the VitalWatch internal SLM (evidence compiler). Convert the medical report into the privacy-minimized evidence JSON schema. Remove identifiers, use relative times, flag missing/inconsistent information, abstain when unreliable, never invent values."},
        {"role": "user", "content": "Extract clinically relevant observations and prepare a privacy-minimized evidence package as JSON.\n\nREPORT:\n" + report_text},
    ],
    temperature=0, max_tokens=768)
print(out["choices"][0]["message"]["content"])
```

Validate every output against schema v1.0 and the privacy firewall before any downstream use —
reference implementation: `vitalwatch_slm_client.py` / `vitalwatch_predictor.py` in the
VitalWatch repository.

## License & data

Apache-2.0 (inherited from the base model). Trained on ELMTEX (Zenodo 14793810, CC-BY-4.0)
derived extraction examples plus clearly-labeled synthetic edge cases. **No real patient data
(PHI) was used** for training or evaluation. Keep this repository private unless the license and
privacy review are repeated for public distribution.
