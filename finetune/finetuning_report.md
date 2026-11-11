# VitalWatch Internal SLM — Fine-Tuning Report

Status: **pipeline complete; training not yet executed** (requires a Kaggle GPU session).
All metric values in §7 are **placeholders** to be filled from `eval_results.json`, which the
notebook writes automatically — no results are invented here.

---

## 1. Role and scope

The internal SLM is the VitalWatch **evidence compiler**: retrieved medical-report text →
structured, privacy-minimized evidence JSON (schema v1.0). Its output feeds filtering →
deterministic validation → the independent privacy firewall → the external LLM. It is explicitly
**not** the temporal ML/DL prediction model, not a diagnostic model, and not a privacy guarantee
(`architecture.md` §8, `prd (1).md` §workflow step 8).

Tasks trained (per project requirements):

- **A — Clinical information extraction** (parameter, value, unit, relative time, source ref, uncertainty)
- **B — Privacy-conscious transformation** (identifier removal/generalization, relative-time conversion)
- **C — Structured evidence generation** (valid JSON, fixed schema)
- **D — Missing/inconsistent information handling** (missing, invalid units, ambiguous dates, contradictions, duplicates)
- **E — Safe abstention** (status=abstain + reason; never invent values)

## 2. Base-model selection

| Candidate | License | HF gating | Params (bf16) | Verdict for this pilot |
|---|---|---|---|---|
| **Qwen2.5-1.5B-Instruct** | Apache-2.0 | **Ungated** (`gated:false`, verified on the HF model page 2026-11-06) | 1.54B | **Selected** |
| Llama-3.2-3B-Instruct | Llama 3.2 Community License | Gated — requires license acceptance + HF token | ~3.2B | Strong candidate for a follow-up run once a token is provisioned; 2× VRAM and gated downloads complicate the Kaggle pilot |
| Gemma-2-2b-it | Gemma Terms of Use | Gated — requires license acceptance + HF token | ~2.6B | Same gating friction; sliding-window attention adds minor config sensitivity on older stacks |

Selection rationale: the pilot must run unattended on Kaggle with internet access and reproducible
downloads — Qwen2.5-1.5B-Instruct is the only candidate that is both ungated and permissively
licensed (Apache-2.0 also permits redistributing the resulting adapter with attribution). At 1.5B
parameters it fits a single T4 with large headroom under 4-bit QLoRA, leaving room for evaluation
of the base model in the same session. The docs' guidance ("candidate 1.5B–3B class, subject to
hardware benchmarking", `architecture.md` §8) is respected; no model is trained from scratch.

## 3. Hardware fit analysis (Kaggle)

| Component | Estimate (T4 16 GB) |
|---|---|
| Base model, 4-bit NF4 + double quant | ≈ 1.2–1.5 GB |
| LoRA adapters (r=16, all attention+MLP projections) | ≈ 60–90 MB |
| Optimizer states (paged AdamW 8-bit) | ≈ 0.15 GB |
| Activations (seq 2048, batch 2, gradient checkpointing) | ≈ 4–8 GB peak |
| **Total training peak** | **≈ 6–10 GB — fits T4 with margin** |

The notebook **measures** the environment instead of assuming it: it queries `nvidia-smi`, reads
the compute capability, selects fp16 (T4/P100) vs bf16 (A100/L4-class) accordingly, asserts
≥ 12 GB VRAM, and raises a clear error instead of silently degrading.

## 4. Fine-tuning configuration (QLoRA)

| Setting | Value | Why |
|---|---|---|
| Quantization | 4-bit NF4, double quant, compute dtype per GPU class | Standard QLoRA; fp16 on T4 (no fast bf16) |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 | r=16 captures the narrow schema task; α=2r keeps updates stable |
| Target modules | q,k,v,o + gate,up,down projections | Full attention+MLP coverage at small r |
| Epochs | 3 | ~1.1k examples; more risks overfitting the synthetic style |
| Effective batch | 16 (2 × grad-accum 8) | Stable updates within T4 memory |
| LR / schedule / warmup | 2e-4 / cosine / warmup = 3% of total steps (≈6 steps, passed as `warmup_steps` — `warmup_ratio` was removed in the transformers version on the 2026 Kaggle image; the training cell filters kwargs against the installed `SFTConfig` so version drift fails soft) | Canonical QLoRA range |
| Max seq len | 2048 | Measured on the train split: median 802 tok, max 1,686 tok after chat templating (the notebook asserts this); 2048 covers the maximum with headroom |
| Precision / optimizer | **AMP disabled** (`fp16=False, bf16=False`) / paged_adamw_8bit; LoRA weights cast to fp32 | The 2026 Kaggle stack's fp16 GradScaler crashes on bf16 gradients (no bf16 unscale kernel); QLoRA trains correctly without loss scaling — 4-bit matmuls keep fp16 compute via bitsandbytes, LoRA grads are fp32 |
| Eval / checkpoints | every 25 / 50 steps, keep 2 | Validation-loss tracking + resumability; final adapter saved explicitly |

Training data: the 1,087-example train split (§7 of `finetuning_dataset.md`), formatted with the
Qwen chat template; target = compact evidence JSON. Validation split drives early-loss tracking
only — no model selection on test.

## 5. Independent privacy safeguards

Fine-tuning is **not** treated as privacy preservation. The notebook implements an independent,
deterministic privacy validator (mirroring the VitalWatch firewall, `architecture.md` §8):

1. JSON schema validation + top-level field allow-list (unexpected keys → reject).
2. Observation field allow-list (only the 8 schema fields per observation).
3. Identifier-pattern scan over the serialized output: email, phone, MRN, street address.
4. Patient-name copy detection (name extracted from source report vs output).
5. Prompt-injection echo detection (instruction-like text in output → reject).
6. Status enumeration check.

The validator runs on **every** model output (including the base model's) and the notebook
demonstrates it rejecting a deliberately unsafe payload — the gate must reject unsafe output even
if the model claims it is anonymized. Presidio/spaCy integration is documented in
`requirements.txt` as an optional upgrade; the regex+schema gate is fully self-contained.

## 6. Evaluation methodology

Fixed, stratified 100-example subset of the untouched test split (50 ELMTEX extraction + 50
synthetic spread across the 11 other categories). The **same subset and prompts** are run through
the base model (fresh 4-bit load) and the fine-tuned model.

Metric mapping to project metrics (`prd (1).md` success metrics; `implementation_plan.md` SLM
workstream):

| Project metric | Notebook metric |
|---|---|
| Valid evidence-reference rate | `source_ref_rate` (source_reference matches gold) |
| Schema-valid rate | `schema_ok_rate` (schema + allow-list + status valid) |
| Appropriate abstention | `abstention_accuracy` (abstains iff gold abstains) |
| Outbound privacy-filter rejection tests | `firewall_rejection_rate` + explicit unsafe-payload demo |
| PII/sensitive leakage | `pii_leak_rate` |
| Clinical fact preservation | `field_precision` / `field_recall` / `field_f1` over (parameter, value, unit) triples |
| Numerical/time consistency | `value_preservation_rate`, `unit_preservation_rate` |
| JSON robustness | `json_valid_rate`, `invalid_output_rate` |
| Latency / token use | `median_latency_s`, `mean_latency_s`, `tokens_per_second` |
| Operational | `training_minutes`, `gpu_peak_memory_gb`, `adapter_size_mb` |

## 7. Results

Two evaluations were run on the same fixed held-out subset (95 examples: 50 ELMTEX / 45 synthetic;
the 100-example target was not fully reachable in some synthetic categories of the 140-example test
split). All values below are actual notebook outputs — none are invented.

### 7.1 Final evaluation (768-token generation) — actual outputs, 95 held-out examples

The re-run removed the generation-cap artifact of evaluation #1. The adapter was loaded from the
pinned Hugging Face revision (`72577e8`) exactly as the VitalWatch application will load it.

| Metric | Base | Fine-tuned |
|---|---|---|
| field_precision | 0.007 | 0.664 |
| field_recall | 0.002 | 0.524 |
| field_f1 | 0.003 | **0.561** |
| json_valid_rate | 0.853 | **0.979** |
| schema_ok_rate | 0.000 | **0.968** |
| invalid_output_rate | 0.147 | **0.021** |
| value_preservation_rate | 0.054 | 0.666 |
| unit_preservation_rate | 0.703 | 0.394 |
| source_ref_rate | 0.000 | 0.463 |
| pii_leak_rate | 0.189 | **0.000** |
| firewall_rejection_rate | 1.000 | **0.032** |
| abstention_accuracy | 0.800 | **1.000** |
| status_accuracy | 0.158 | **0.979** |
| median_latency_s | 18.391 | 44.167 |
| tokens_per_second | 18.475 | 9.464 |

Operational: GPU peak 6.44 GB · adapter 85.3 MB (re-saved fp32 in the resume session; the HF
revision `72577e8` copy remains fp16 at 48.4 MB) · training 116.1 min.

**Interpretation.** The fine-tuned SLM wins decisively on every structural and safety metric:
97.9% JSON validity, 96.8% firewall-clean schema compliance, correct status on 97.9% of reports,
perfect abstention, and **zero PII leakage across two independent evaluations** (190 generations).
The qualitative demo (identifiers_present category) shows a PII-heavy report converted to a clean
evidence package — no name, MRN, phone, email or address in the output — with the firewall
returning ALLOWED.

Extraction quality is moderate rather than strong: field F1 0.561 at precision 0.664. The gaps
concentrate in ELMTEX-derived examples (entity-string matching against free-text annotations,
observation-level timestamps unavailable in the source, and unit strings like
`nmol/L (normal 200 to 700nmol/L)` that fail exact-match comparison — also the main driver of the
lower `unit_preservation_rate`, which compares against 398 gold observations in this run vs 61 in
evaluation #1). The model's conservative `"value": null` + missing-information pattern for
uncertain extractions further trades recall for precision-of-absence, which is the desired
behavior for a firewall-fronted compiler.

**Next-lever candidates** (data-side, not architecture): more ELMTEX-derived examples with
normalized unit strings, observation-level timestamp synthesis, and unit-canonicalization in the
evaluator's comparison key. No pipeline changes are required.

### 7.2 Interim evaluation #1 (384-token generation) — kept as a documented limitation

The first evaluation ran with `max_new_tokens = 384`, which **truncated** long fine-tuned outputs
(a 14-observation ELMTEX evidence package needs ~700+ tokens), cascading into failed JSON parsing,
low recall, and firewall schema rejections. These numbers are real but understate the model; they
are kept to document the lesson (see the notebook's execution log, Part 1 step 13):

| Metric | Base | Fine-tuned |
|---|---|---|
| field_precision | 0.011 | 0.984 |
| field_recall | 0.002 | 0.122 |
| field_f1 | 0.003 | 0.218 |
| json_valid_rate | 0.547 | 0.284 |
| schema_ok_rate | 0.000 | 0.284 |
| value_preservation_rate | 0.069 | **1.000** |
| unit_preservation_rate | 0.621 | 0.885 |
| source_ref_rate | 0.000 | 0.274 |
| pii_leak_rate | 0.147 | **0.000** |
| firewall_rejection_rate | 1.000 | 0.716 |
| abstention_accuracy | 0.985 | **1.000** |
| status_accuracy | 0.137 | 0.284 |
| median_latency_s | 18.405 | 41.480 |
| tokens_per_second | 18.687 | 9.191 |

**What evaluation #1 already establishes** (truncation-independent): zero PII leakage across 95
fine-tuned outputs, perfect value fidelity for every extracted observation, perfect abstention
behavior, and 98% field precision — the safety-critical properties the VitalWatch firewall
depends on.

Operational: training 116.1 min · GPU peak 4.12 GB · adapter 48.4 MB · final eval loss 0.440 ·
mean token accuracy 88.7%.

## 8. Hugging Face / GitHub integration

- **GitHub holds code only** (this `finetune/` directory). **HF holds the adapter** (private repo
  recommended). The notebook's final cell pushes adapter + tokenizer after uncommenting, using a
  Kaggle secret `HF_TOKEN`.
- The VitalWatch app loads: **pinned base-model revision + pinned adapter revision + tokenizer**,
  then validates every output against **schema v1.0** before the evidence-package builder
  (matching `implementation_plan.md` step 16's versioned `EvidencePackage` contract).
- License: Apache-2.0 base → adapter redistribution permitted with attribution; the adapter
  model card (auto-generated by the notebook) records base model, license, and the no-PHI
  training statement.
- **Never** commit or upload: real patient records, PHI, credentials, private/gated datasets, or
  synthetic examples mislabeled as real data.

### 8.1 Teammate setup — getting the fine-tuned adapter

**Route B (recommended): private HF repo** — after `pip install -r requirements.txt`:

```bash
# one-time: repo owner adds the teammate's HF username in repo Settings → access control;
# teammate creates a Read token and logs in
huggingface-cli login

# download the adapter, pinned to the trained revision
huggingface-cli download TheBlindCoder/vitalwatch-slm-qwen-1p5b-lora \
    --revision 72577e8 --local-dir vitalwatch-slm/adapter

# use it (the public base model auto-downloads on first run and is cached)
python finetune/inference_example.py --adapter vitalwatch-slm/adapter
```

Application load path (pinned revisions + offline cache, per `implementation_plan.md`):
`snapshot_download(ADAPTER, revision=...)` → `PeftModel.from_pretrained(base, ADAPTER,
revision=...)` → afterwards `HF_HUB_OFFLINE=1` runs purely from cache. If a configurable model
endpoint URL is ever introduced, validate it server-side before use: only http/https, and reject
hosts resolving to loopback, private, or reserved addresses.

**Route A (no HF access): file distribution** — the `vitalwatch-adapter.zip` from the Kaggle
Output tab can be shared privately (Drive/etc.). Receiver unzips to `vitalwatch-slm/adapter` and
runs the same `inference_example.py --adapter` command. Share a SHA256 checksum alongside and
verify on receipt. Do not commit weights to git (GitHub = code, HF/shared drive = weights).

## 9. Limitations and what was NOT done

1. **Training was not executed** — no GPU was available in the working session; the notebook is
   the execution vehicle on Kaggle. Consequently all §7 values are pending.
2. Evaluation is on **synthetic + ELMTEX-derived** text; real-EHR performance is unproven (the PRD
   itself flags synthetic-only evaluation as a limitation).
3. MIMIC-IV-Ext-Instr and n2c2/i2b2 2014 were **not** used (credentialed access required); the
   substitution and swap path are documented in `finetuning_dataset.md`.
4. The dataset's `EvidencePackage`-level fields (risk trajectory, temporal findings) belong to the
   downstream builder, not the SLM; the SLM-to-`Observation`/`EvidencePackage` mapping must be
   implemented and version-tested during integration.
5. Latency numbers include tokenization and include single-GPU sequential evaluation; treat them
   as relative comparisons, not production benchmarks.
