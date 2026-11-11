# VitalWatch SLM — Local GGUF Integration Guide

How to run the fine-tuned VitalWatch internal SLM **in-process from project code** (no UI, no
Jupyter, no server) via `llama-cpp-python`. Companion files:

- `vitalwatch_predictor.py` — the predictor (load once, `.predict()` per report)
- `vitalwatch_slm_client.py` — shared JSON parser + privacy firewall (imported by the predictor)
- `vitalwatch-slm-q8_0.gguf` — the model weights (downloaded from the Kaggle run output)

---

## 1. Install the engine (one-time)

```powershell
pip install llama-cpp-python
```

⚠️ On Windows this may build from source — if it errors, install **CMake** + Visual Studio
**Build Tools ("Desktop development with C++")** and retry, or grab a prebuilt Windows wheel
from the [llama-cpp-python GitHub releases](https://github.com/abetlen/llama-cpp-python/releases)
matching your Python version (3.12 here).

## 2. Get the GGUF

The smoothest variant is **Q8_0** — no `llama-quantize` step at all. On Kaggle, the conversion
cell becomes:

```python
!python /kaggle/working/llama.cpp/convert_hf_to_gguf.py /kaggle/working/vitalwatch-slm-merged --outfile /kaggle/working/vitalwatch-slm-q8_0.gguf --outtype q8_0
```

→ **Save Version → Quick Save** → download `vitalwatch-slm-q8_0.gguf` (1.6 GB) into this
`finetune/` folder.

If you already made a `Q4_K_M` file with `llama-quantize`, that works identically — just point
`model_path` at it. Provenance of either file: base `Qwen/Qwen2.5-1.5B-Instruct` merged with
adapter revision `72577e8`.

Reliable download for multi-GB files (browser button can fail): Kaggle API —

```powershell
pip install kaggle                      # + kaggle.json from kaggle.com → Settings → API
kaggle kernels output piyushtheblindcoder/model-fine-tunning -p .
```

## 3. Predict from anywhere in the project code

```python
from vitalwatch_predictor import predict

result = predict(report_text)          # first call loads the model (~10 s), then fast
print(result["evidence"])              # the evidence-package dict (or None)
print(result["allowed"])               # privacy firewall verdict
print(result["violations"])            # quarantine reasons if not allowed
```

Or as a long-lived object (load once at app startup):

```python
from vitalwatch_predictor import VitalWatchSLM

slm = VitalWatchSLM(model_path="vitalwatch-slm-q8_0.gguf")
result = slm.predict(report_text)
```

CLI smoke test: `python vitalwatch_predictor.py` (uses the bundled sample report).

---

## What to expect

- **Speed:** ~15–25 tokens/s on an Iris Xe CPU → a full evidence JSON in ~30–60 s
  (5–10× faster than the fp32 transformers route)
- **RAM:** ~2 GB while loaded — much friendlier than fp32's 6.2 GB
- **Deterministic:** temperature 0 → same report, same JSON
- **Privacy firewall verdict travels with every prediction** — downstream code MUST quarantine
  whenever `allowed` is `False`, exactly per the VitalWatch architecture
  (`prd (1).md`: no raw reports or unvalidated output leave the controlled environment)

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install llama-cpp-python` build errors | Install CMake + VS Build Tools (C++ workload), or use a prebuilt wheel from the GitHub releases page |
| Output JSON malformed / model rambles | Confirm temperature 0 and the system prompt (hardcoded in `vitalwatch_predictor.py`); if a Q4 file misbehaves, switch to Q8_0 |
| "unparseable model output" | Output hit the token cap mid-JSON — rerun with `VitalWatchSLM(model_path=..., max_tokens=1024)` |
| Model file not found | Point `model_path` at the GGUF's actual location, or move the file next to this script |
| Slow first call | Normal — the model loads into RAM once per process; reuse the singleton `predict()` instead of re-creating `VitalWatchSLM` |

## Alternative: HTTP service mode

For a production VitalWatch deployment, `vitalwatch_slm_client.py` talks to the same model
served by `llamafile`/`llama-server` on `http://127.0.0.1:8080` (OpenAI-compatible API) —
load once, serve many requests, no model reload per process. The in-process predictor above
is the simpler path for scripts and demos.
