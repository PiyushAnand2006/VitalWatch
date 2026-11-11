# Detailed Implementation Plan

## Step-by-step delivery plan

**Version:** 1.1 | **Date:** 6 November 2026\
**Planning assumption:** 12-week university semester, small team,
student-laptop-scale compute. Adjust week numbers to the actual
calendar.

## 0. Strategy and milestones

Build and test the system in layers. The MVP includes synthetic data generation, validated ingestion, persistent storage plus Redis context caching, temporal prediction, patient-scoped RAG, internal SLM privacy preparation, outbound privacy validation, constrained external LLM explanation and Agent Orchestrator verification. Deliver a working vertical slice early; advanced models and broad dataset integrations remain staged.

  -----------------------------------------------------------------------
  Gate                                Exit condition
  ----------------------------------- -----------------------------------
  G1 Scope/data                       Research question, dataset, target,
                                      access, and split documented

  G2 Data pipeline                    Reproducible normalized dataset and
                                      audit

  G3 Baseline model                   Leakage checks pass; metrics
                                      reproducible

  G4 Replay MVP                       Timeline and risk trajectory work

  G5 Evidence engine                  Time-indexed findings trace to
                                      observations

  G6 Context and cache                Redis cache-aside, patient isolation,
                                      freshness and invalidation tested

  G7 Safe RAG/privacy hand-off        RAG provenance, SLM schema, privacy
                                      validator and blocked unsafe payload

  G8 Orchestrated explanation        LLM output verified; bounded revision
                                      and abstention tested

  G9 Final                            Comparative results, demo,
                                      limitations, reproducibility
  -----------------------------------------------------------------------

If a gate fails, fix it before adding complexity.

# Phase 1 --- Scope and repository

## Step 1. Freeze the research question

Working question:

> Does constraining an LLM with temporal physiological evidence and
> selectively retrieved patient context produce more reliable,
> explainable, and privacy-conscious clinical reasoning than asking an
> LLM to reason directly from patient data?

Sub-questions: - Does temporal modelling provide useful warning
performance on the selected dataset? - Can the evidence engine
accurately describe temporal patterns? - Does retrieval add relevant
context when compatible notes are available? - Does evidence-constrained
prompting reduce unsupported claims? - Can the system abstain when
evidence is insufficient or unsafe?

Do not claim clinical benefit.

## Step 2. Freeze MVP and extensions

**MVP:** one approved time-series dataset; synthetic patient/report generator; validated ingestion; patient-level split; causal features; threshold/logistic-regression and stronger baseline comparison; PostgreSQL persistence; Redis patient-context cache; replay dashboard; temporal evidence; patient-scoped RAG (using compatible approved notes or labelled synthetic reports); internal SLM context preparation; privacy validator; constrained external LLM; Agent Orchestrator verification and evaluation.

**Extension:** additional fine-tuning experiments, additional SLM comparisons, advanced sequence models, second-dataset validation and expanded robustness/subgroup analyses.

**Future development only:** smartphone sensing, wearables and external-device integrations; exclude from MVP implementation and acceptance criteria.

Deliverable: agreed scope in `docs/scope.md`, including the three gaps (context; hallucinations; accuracy/privacy) and their measurable evaluation criteria.

## Step 3. Create repository and environment

Create the repository using this layout:

``` text
project/
├── app/{api,core,ingestion,preprocessing,modelling,evidence,retrieval,privacy,llm,services}/
├── dashboard/
├── data/{raw,interim,processed}/
├── artifacts/
├── configs/
├── notebooks/
├── scripts/
├── tests/
├── docs/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Tasks: 1. Initialize Git and default branch. 2. Add setup/run
instructions and research-only safety notice. 3. Ignore raw data,
restricted processed data, `.env`, credentials, caches, and large
generated artifacts. 4. Create Python virtual environment. 5. Pin core
dependencies. 6. Add `pytest` and a repeatable test command. 7. Add
`.env.example` with variable names only.

Exit: a teammate can clone and set up the repository without needing
secrets committed.

# Phase 2 --- Dataset and task definition

## Step 4. Acquire and document the temporal dataset

Recommended start: PhysioNet Computing in Cardiology Challenge 2019\
https://physionet.org/content/challenge-2019/1.0.0/

Tasks: 1. Read official description, access/licence terms, and data
dictionary. 2. Download permitted files and record version/date. 3.
Inspect patient counts, sequence lengths, time coverage, and label
prevalence. 4. List time-dependent variables and units. 5. Inspect
missingness and irregular sampling. 6. Record the meaning and
limitations of the label. 7. Create `docs/data_sources.md` and
`docs/data_dictionary.md`.

Do not assume every column is clinically meaningful or consistently
observed.

## Step 5. Run a data-quality audit

For each variable calculate missing percentage, min/max/mean/std,
invalid values, patient coverage, and measurement frequency. Inspect
duplicate timestamps, sequence lengths, time gaps, class balance, and
label timing.

Create `scripts/audit_dataset.py`; save tables as CSV/JSON and plots as
image files. Manually inspect several patient timelines. Document
exclusions and transformations.

Exit: audit script runs reproducibly and every exclusion has a reason.

## Step 6. Define the prediction task

Write down: - target label/event; - prediction timestamp; - available
historical observation window; - prediction horizon; - event/label
window handling; - eligible rows and exclusion criteria.

Treat the dataset label as the dataset target, not a universal
deterioration definition. If deriving an early-warning target, specify
the exact rule.

Deliverable: `docs/task_definition.md`.

## Step 7. Create patient-level splits

1.  Split by patient, not by hourly row.
2.  Keep a held-out test partition.
3.  Fit preprocessing on training data only.
4.  Use validation data for model and threshold selection.
5.  Do not inspect test results during tuning.
6.  Save split manifest under appropriate access controls.
7.  Add automated overlap tests.

``` python
assert train_patient_ids.isdisjoint(valid_patient_ids)
assert train_patient_ids.isdisjoint(test_patient_ids)
assert valid_patient_ids.isdisjoint(test_patient_ids)
```

Exit: overlap tests pass.

# Phase 3 --- Preprocessing and features

## Step 8. Normalize data

1.  Convert records into a consistent patient-time-variable
    representation.
2.  Sort by patient and timestamp.
3.  Standardize column names while retaining source mapping.
4.  Preserve units and source metadata.
5.  Detect and document duplicate handling.
6.  Represent missingness explicitly.
7.  Never use future values to fill earlier timestamps.
8.  Separate raw, interim, and processed data.

Deliverable: deterministic preprocessing functions and tests.

## Step 9. Build causal features

Start with interpretable features: - current value; - past 3-hour and
6-hour rolling mean (adjust to sampling/task); - past-window
min/max/std; - slope/rate of change; - change from previous valid
observation; - deviation from a documented baseline, if available; -
time since last valid measurement; - missingness indicator and
past-window missingness fraction.

Implementation: 1. Use backward-looking windows ending at or before time
`t`. 2. Group every rolling operation by patient. 3. Ensure windows
never cross patient boundaries. 4. Fit imputers/scalers on training data
only. 5. Retain missingness indicators. 6. Unit-test against a small
hand-built timeline. 7. Test that appending future rows does not change
features at earlier time `t`.

Deliverables: `app/preprocessing/features.py`, tests, and
`docs/feature_definitions.md`.

## Step 10. Train simple baselines

Implement in order: 1. Prevalence/no-skill baseline. 2. Simple
documented threshold baseline. 3. Logistic regression on causal
features.

Use a fixed seed, a single training entry point, versioned
configuration, and saved validation predictions. Choose thresholds using
a documented validation objective, never the test set.

Deliverable: baseline model artifact and training log.

## Step 11. Try one stronger model

Only after the baselines work, try gradient boosting (e.g.,
LightGBM/XGBoost if installable). Use identical splits and features.
Tune on training/validation only. Compare performance and latency. Keep
the simpler model if gains are small or unstable.

# Phase 4 --- Evaluation and API

## Step 12. Evaluate the model

Calculate AUROC, AUPRC, sensitivity, specificity, precision, F1,
confusion matrix, Brier score/calibration plot, false alerts per
patient-day, and warning lead time where the target definition permits.

Report class balance, threshold-selection method, and confidence
intervals where feasible. Do not use accuracy as the main metric for
imbalanced data. Do not claim clinical validation.

Deliverables: `reports/model_evaluation.md`, metric tables, plots, and
held-out predictions.

## Step 13. Robustness checks

At minimum inspect performance by missingness level and sequence length;
repeated alert burden; false positives/negatives; patient leakage; and
temporal leakage. Subgroup analysis only where permitted fields and
sample sizes support it.

Deliverable: `reports/robustness.md`.

## Step 14. Build FastAPI replay endpoints

Implement: - `GET /health` - `GET /patients` -
`GET /patients/{id}/timeline` - `GET /patients/{id}/risk`

Load the versioned model on startup; validate IDs/time ranges; return
chronological observations, as-of time, and model version; handle
missing artifacts and unknown IDs; add API tests.

# Phase 5 --- Replay dashboard

## Step 15. Build the main screen

Suggested sections: 1. Patient selector and research disclaimer. 2.
Current risk/status, model version, and as-of time. 3. Time-series
charts for available variables. 4. Risk trajectory. 5. Replay controls:
play, pause, previous, next, slider. 6. Temporal evidence panel.

Tasks: - Choose a small set of representative demo patients. - Build
selector and charts with units. - Add replay controls. - Keep chart,
risk, and selected timestamp synchronized. - Show missing values
explicitly. - Add loading/error states. - Make the dashboard work
without LLM or RAG.

Exit: reviewer can replay a timeline and verify that displayed
risk/evidence corresponds to selected time.

# Phase 5A --- Persistence, synthetic data and Redis context cache

## Step 15A. Implement the primary patient database

Use PostgreSQL for persistent patient profiles, validated observations, report/document metadata, prediction runs, evidence references and audit metadata. Define patient-scoped foreign keys and source provenance. Use SQLite only for a deliberately limited local prototype if PostgreSQL setup is a blocker; the architecture must still keep the persistent store authoritative.

Exit: restart the application and confirm accepted observations remain available without Redis.

## Step 15B. Build the synthetic patient/report generator

Generate configurable synthetic profiles, chronological vitals/lab observations, report-like PDFs/text, missing values, duplicates, unit/date errors and stable, improving, deteriorating, fluctuating and sudden-abnormal scenarios. Add an explicit `data_origin=synthetic` marker to every record/document and display it in the dashboard. Send generated data through the same ingestion and validation path as uploaded data.

Exit: automated tests confirm synthetic labels survive ingestion, storage, cache, RAG and dashboard display.

## Step 15C. Add Redis patient-context cache

Implement cache-aside access: authorize patient → read Redis → on miss read PostgreSQL → populate Redis. Cache only frequently used, compact context (recent structured observations, trend summaries, recent prediction metadata and context version); store raw reports and durable observations in the primary database. Use pseudonymous patient-scoped key namespaces, TTLs for derived summaries, freshness timestamps and model/data version metadata. On a committed record update, update or invalidate affected keys. Fall back safely to PostgreSQL on Redis failure.

Tests: cache hit/miss, cold-start fallback, update invalidation, TTL expiry, stale prediction rejection, concurrent update behavior and cross-patient isolation. Do not allow a client-provided patient ID to bypass authorization.

Exit: Redis improves repeat-read latency without changing correctness or persistence semantics.

# Phase 6 --- Temporal evidence engine

## Step 16. Define the evidence schema

Create Pydantic/JSON Schema models for event ID, pseudonymous patient
key, as-of timestamp, risk/model metadata, observations, findings,
missingness, and provenance.

Deliverable: versioned `EvidencePackage` schema.

## Step 17. Implement deterministic extraction

For selected patient and as-of time: 1. Fetch only observations at or
before `t`. 2. Select documented historical windows. 3. Calculate
current values and trends. 4. Detect persistent/abrupt changes only
using documented rules. 5. Calculate time since last measurement. 6.
Summarize missingness. 7. Link every finding to source observation IDs.
8. Sort findings by time and relevance. 9. Emit JSON.

Avoid vague claims such as "vitals worsening" without a trend and time
window.

## Step 18. Test evidence quality

Create test timelines for stable values, gradual increase, abrupt
change, persistent abnormality, intermittent missingness, long gaps,
contradictory trends, and no recent measurements.

Exit: all findings have source IDs; no future data is used; missingness
and empty evidence are explicit.

## Step 19. Add evidence panel

Show as-of time, current values, trends/windows, persistent or abrupt
changes, missing observations, and source observation details. Keep this
deterministic panel available even when LLM is disabled.

# Phase 7 --- RAG and notes

## Step 20. Confirm notes access and compatibility

Candidate: MIMIC-IV and MIMIC-IV-Note: -
https://physionet.org/content/mimiciv/ -
https://physionet.org/content/mimic-iv-note/

Check credentialing/training, exact release, documented linkage keys,
note types, permissions, and whether records can be linked to the
temporal cohort.

**Hard rule:** PhysioNet Challenge 2019 IDs must not be joined to MIMIC
IDs. If no compatible notes exist, use synthetic notes clearly labelled
as synthetic or run a separate RAG demonstration. Do not imply
patient-specific linkage.

Deliverable: documented notes-data decision.

## Step 21. Parse and chunk notes

If approved: 1. Load permitted note types only. 2. Preserve document ID,
patient key, timestamp, and section. 3. Apply required de-identification
safeguards. 4. Chunk along section/semantic boundaries. 5. Store chunk
metadata and provenance. 6. Create a small retrieval test set.

For synthetic notes, label every record and keep synthetic/real sources
distinct.

## Step 22. Build local retrieval

1.  Generate embeddings locally.
2.  Build local FAISS or Chroma index.
3.  Filter by patient before ranking.
4.  Apply as-of-time filters where required.
5.  Retrieve small top-k.
6.  Preserve chunk/document IDs, timestamp, section, and score.
7.  Return no context when relevance is too low.
8.  Test cross-patient isolation and future-note exclusion.

Exit: retrieval is provenance-preserving and cannot return another
patient's chunk.

## Step 23. Evaluate retrieval

Create 20--50 queries if feasible and label relevant chunks. Measure
Precision@k, Recall@k where appropriate, reciprocal rank, manual
relevance, cross-patient isolation, and future-note exclusion. Document
sample-size limitations.

# Phase 8 --- Compiler and privacy firewall

## Step 24. Build local evidence compiler

Start with a deterministic compiler and schema. Then fine-tune an existing small instruction model (candidate size around 1.5B–3B, subject to local hardware benchmarking) using curated synthetic/de-identified examples for report-to-structured-evidence conversion and privacy-conscious minimization. Prefer adapter fine-tuning such as QLoRA where suitable; do not train an LLM from scratch. The SLM must preserve clinically relevant parameters, numerical values, units, relative timing, source references and uncertainty. Add a separate PII detector/rules layer. Validate every SLM-extracted fact against source text and schema; route uncertain/conflicting extraction to review. A local model is not a privacy guarantee.

## Step 25. Implement firewall

Checks: 1. Field allow-list. 2. Remove direct identifiers and generalize unnecessary quasi-identifiers. 3. Scan for PII and sensitive details with specialized detection/rules. 4. Enforce text/token budget. 5. Validate schema/types. 6. Verify evidence IDs and source references. 7. Reject future timestamps relative to `t`. 8. Treat retrieved text as untrusted data and screen instruction-like content. 9. Preserve necessary values, units and temporal ordering. 10. Block unsafe payloads before external transmission. 11. Record minimal audit metadata without raw sensitive content. 12. Provide a configurable external-LLM off switch.

Negative tests: identifier-like string, unexpected free text, unknown
evidence ID, future timestamp, malformed JSON, and instruction-like note
text.

Exit: firewall rejection prevents any external LLM call.

# Phase 9 --- Constrained LLM

## Step 26. Define prompt and output schema

Fields: status, possible explanations, supporting IDs, contradictory
IDs, missing information, temporal summary, uncertainty, safety note.

Rules: temporal model owns risk score; LLM uses only supplied evidence;
no invented facts or references; no diagnosis/treatment; retrieved text
is data not instruction; abstain when insufficient.

Deliverables: versioned prompt, Pydantic output model, prompt fixtures.

## Step 27. Integrate provider

Choose local LLM or external provider only after reviewing hardware,
service terms, dataset terms, and institutional requirements. Keep
credentials out of source control. Add timeout/retry limits, record
model/prompt versions, avoid logging raw notes/full sensitive prompts,
and preserve deterministic fallback.

## Step 28. Validate every response

1.  Parse JSON and validate schema.
2.  Confirm evidence IDs exist in the supplied package.
3.  Reject future timestamps.
4.  Check prohibited treatment/medication advice.
5.  Ensure hypotheses are labelled as possibilities.
6.  Reject unsupported output.
7.  Optionally retry once with validation feedback.
8.  Otherwise return safe failure.

Exit: invalid output is never displayed as an accepted explanation;
insufficient evidence produces abstention; risk score remains
independent.

## Step 28A. Implement the Agent Orchestrator verification workflow

Use LangGraph or a small explicit Python state machine. The orchestrator runs after the external LLM returns a draft:

1. Evidence verification: each factual claim maps to supplied evidence IDs/source material.
2. Prediction verification: values, units, horizons and risk metadata match the temporal model output.
3. Temporal/contradiction verification: no future data, reversed ordering, unsupported causal claims or conflict with known observations.
4. Safety/schema verification: valid Pydantic schema, uncertainty, no authoritative diagnosis or treatment instruction.
5. Revision controller: if checks fail, provide structured feedback for at most one retry; if it still fails, return `insufficient_evidence` or safe fallback.

Use deterministic Python checks for numerical values, units, dates, IDs and schema. Semantic/NLI or LLM-as-judge checks may supplement, but must not be the sole verifier. Keep the orchestrator's state and decision trace auditable without logging raw patient text.

Tests: invented evidence ID, altered prediction number, wrong unit, future date, contradiction, unsupported diagnosis, invalid schema, retry limit and abstention behavior.

Exit: no unverified draft is shown as an accepted explanation.

## Step 29. Add explanation panel

Display possible explanations, supporting and contradictory evidence,
missing information, temporal summary, uncertainty, provenance, and
clinician-review warning. Keep raw measurements and deterministic
evidence visible beside it.

# Phase 10 --- Experiments and evaluation

## Step 30. Create fixed evaluation cases

Select held-out examples covering clear positives/negatives, borderline
cases, high missingness, contradictory evidence, and insufficient
context. Define selection rules before reviewing outputs.

## Step 31. Compare the following configurations

| Configuration | Input | Purpose |
|---|---|---|
| A: Snapshot baseline | Current measurements → LLM | Static baseline; use synthetic/de-identified cases only |
| B: History-aware | Current measurements + selected history → LLM | Measure value of historical context |
| C: Evidence-constrained | Temporal model + structured temporal evidence + approved RAG context → LLM | Measure constrained reasoning |
| D: Orchestrated | Configuration C + Agent Orchestrator verification/revision/abstention | Measure reduction in unsupported output |

Also compare cache disabled vs Redis enabled for response latency and context freshness. Compare privacy strategies (rule-based minimization versus SLM + privacy validator) for PII leakage, clinical fact preservation, numerical/temporal consistency and latency. Never send real identifiable patient data to an external baseline. Use identical held-out cases and output schemas. If notes are not validly linked to the temporal cohort, do not present them as patient-specific context; evaluate RAG separately or use clearly labelled synthetic notes.

## Step 32. Score explanations

Manually assess factual consistency, temporal ordering,
evidence-reference correctness, unsupported claims, missing-information
recognition, uncertainty, abstention, prohibited claims, and
readability. Have two reviewers score a subset if possible; report
disagreements and sample limitations.

## Step 33. Run privacy/robustness tests

Test identifier blocking, no raw-note passthrough, cross-patient
isolation, future-note exclusion, prompt-injection resistance, invalid
evidence rejection, LLM outage behavior, and missing-note behavior.

Deliverable: `reports/llm_and_privacy_evaluation.md`.

# Phase 11 --- Integration and submission

## Step 34. End-to-end test

Run: start API → select patient → replay time → risk prediction →
evidence extraction → optional retrieval → compiler → firewall → LLM →
output validation → dashboard.

Also test no notes, RAG unavailable, firewall rejection, LLM timeout,
invalid output, and missing model artifact.

## Step 35. Prepare 5--7 minute demo

1.  Explain problem and research question.
2.  Replay patient timeline.
3.  Show changing risk trajectory.
4.  Open temporal evidence and trace it to observations.
5.  Show retrieved context/provenance if enabled.
6.  Show evidence package and firewall result.
7.  Show possible explanations, evidence for/against, and uncertainty.
8.  Demonstrate insufficient evidence or service failure.
9.  End with limitations and future work.

Use synthetic data for public screenshots unless the real-data terms
explicitly permit display.

## Step 36. Complete documentation

Required: PRD, architecture, implementation plan, dataset/data
dictionary, target definition, feature definitions, model evaluation,
RAG evaluation if implemented, LLM/privacy evaluation if implemented,
limitations/safety statement, and setup/reproduction guide.

## Step 37. Final report and presentation

Include problem, research gap supported by reviewed papers,
requirements, architecture, dataset/access constraints, methods, leakage
prevention, model and alert metrics, evidence engine, RAG/LLM,
comparative experiments, privacy/safety, limitations, and future work.
Do not claim novelty merely because components are combined.

# Suggested 12-week schedule

    Week Work                                            Deliverable
  ------ ----------------------------------------------- ---------------------
       1 Scope, research question, repo, access review   G1
       2 Dataset audit and target definition             Data inventory
       3 Patient split, preprocessing, leakage tests     G2
       4 Causal features and baselines                   Feature tests/model
       5 Evaluation, threshold, robustness               G3
       6 API, replay dashboard and persistent DB          G4
       7 Synthetic generator and Redis cache              G5/G6
       8 Temporal evidence + patient-scoped RAG           Evidence/RAG demo
       9 Internal SLM compiler + privacy validator        G7
      10 External LLM + Agent Orchestrator                 G8
      11 Comparative experiments and integration          Evaluation report
      12 Demo, report, presentation, reproducibility      G9

### If only 6--8 weeks are available

Prioritize one end-to-end vertical slice: scope/data, synthetic cases, validated ingestion, patient split, causal features, baseline, persistent storage, basic Redis cache, replay dashboard and temporal evidence. Implement a minimal patient-scoped RAG → SLM/rules → privacy gate → LLM → orchestrator path only if core evaluation is stable; otherwise clearly report the deferred integration as incomplete rather than claiming it is implemented.

# Team ownership suggestion

  Role         Responsibility
  ------------ ------------------------------------------------------
  Data/ML      Dataset, preprocessing, features, model, evaluation
  Backend      API, schemas, replay service, orchestration
  Frontend     Timeline, risk plots, evidence/explanation panels
  RAG/LLM/QA   Retrieval, compiler, firewall, LLM validation, tests

Combine roles for smaller teams. Everyone reviews leakage, privacy, and
safety.

# Definition-of-done checklist

## Data/model

-   [ ] Dataset terms and target definition documented.
-   [ ] Patient-level split and overlap tests pass.
-   [ ] Causal feature tests pass.
-   [ ] Preprocessing fit on training data only.
-   [ ] Model and held-out evaluation reproducible.
-   [ ] Imbalance-aware metrics and alert burden reported.

## Product

-   [ ] Patient timeline replays.
-   [ ] Risk trajectory synchronizes with selected time.
-   [ ] Missing observations visible.
-   [ ] Dashboard works without RAG/LLM.
-   [ ] Risk and LLM explanation are separate.

## Context/cache

-   [ ] PostgreSQL (or chosen persistent store) remains authoritative.
-   [ ] Redis cache hit/miss, invalidation, TTL and fallback tests pass.
-   [ ] Patient/tenant isolation is enforced and tested.
-   [ ] Synthetic data is labelled end-to-end and uses the same ingestion path.

## Evidence/RAG

-   [ ] Findings cite source observations.
-   [ ] No future observations enter evidence at time t.
-   [ ] Patient linkage documented and valid.
-   [ ] Retrieval preserves provenance and patient filters.
-   [ ] Synthetic notes clearly labelled.

## Privacy/LLM

-   [ ] Compiler schema-valid.
-   [ ] Firewall rejects disallowed fields and invalid references.
-   [ ] Raw notes are not sent externally.
-   [ ] LLM output schema/evidence references validated.
-   [ ] Insufficient evidence produces abstention.
-   [ ] Internal SLM output is source-validated; privacy validator blocks unsafe outbound payloads.
-   [ ] Agent Orchestrator checks evidence IDs, numerical claims, temporal consistency and contradictions.
-   [ ] Revision is bounded and failure results in abstention/safe fallback.
-   [ ] No diagnosis/treatment presented as authoritative.

## Submission

-   [ ] Clean setup instructions work.
-   [ ] Demo script works.
-   [ ] Reports and diagrams included.
-   [ ] Limitations explicit.
-   [ ] No credentials or restricted raw data committed.


# Version 1.1 implementation alignment notes

The three core gaps are explicit workstreams: **context** (PostgreSQL + Redis cache + RAG), **hallucination reduction** (Agent Orchestrator), and **accuracy/privacy** (temporal model experiments plus fine-tuned internal SLM and privacy validator). Accuracy and privacy must be evaluated independently. Redis is not an accuracy model; it supplies fresh context efficiently. The orchestrator reduces unsupported output but cannot guarantee zero hallucinations. The SLM does not guarantee anonymity.

**Required data-flow rule:** report chunks → SLM extraction/minimization → filtering → deterministic validation → privacy gate. Separately, validated structured observations → temporal feature pipeline → ML/DL models. Only the validated prediction/evidence package and approved report context converge at the external LLM. Never feed raw RAG chunks or unvalidated SLM output directly into the temporal models.

**Deployment/distribution:** GitHub stores source code, configs without secrets, tests and training/inference scripts. Model weights/adapters may be hosted separately (e.g. private Hugging Face repository) subject to license and data restrictions. The application should pin base-model and adapter revisions, download/cache on first run where authorized, and support offline cache. Benchmark the actual student hardware before selecting a model size; do not promise fixed RAM/VRAM requirements. Never publish patient records, PHI, credentials or sensitive training examples.

**Out of MVP:** smartphone camera pulse, mobility/pose, voice/respiratory sensing, accelerometer/gyroscope, wearables and external sensors. Keep these only in a future-development pitch section, not in MVP tasks or acceptance tests.


# Version 1.1 implementation alignment

## Workstream A — Context and Redis

1. Implement PostgreSQL persistence for patient profiles, validated observations, report metadata, prediction runs and evidence references.
2. Add Redis cache-aside reads: authorize patient → Redis → database fallback → cache population.
3. Cache compact recent context, trend summaries and prediction metadata only; do not make Redis the durable record store.
4. Add pseudonymous patient-scoped keys, TTLs, freshness/data/model versions and explicit invalidation after committed updates.
5. Test cache hit/miss, cold start, fallback, expiry, stale prediction rejection, concurrent updates and cross-patient isolation.

## Workstream B — Synthetic data in MVP

Generate configurable synthetic patient profiles, chronological observations and report-like documents. Include stable, improving, deteriorating, fluctuating, sudden abnormal, missing and inconsistent cases. Label every generated artifact `synthetic` end-to-end and pass it through the same ingestion/validation path as uploaded data. Synthetic-only results are not evidence of real-world clinical performance.

## Workstream C — Privacy-preserving SLM

1. Establish deterministic extraction/schema baseline.
2. Fine-tune an existing small instruction model for report-to-structured-evidence transformation; benchmark model size against available hardware before choosing.
3. Prefer LoRA/QLoRA adapters where appropriate; do not train an LLM from scratch.
4. Add PII detection and rules as an independent control.
5. Preserve clinical values, units, relative timing, evidence IDs and provenance.
6. Validate each extracted fact against source text and schema; quarantine uncertain/conflicting extraction.
7. Evaluate PII/sensitive leakage, clinical fact preservation, numerical/time consistency, latency and token use.

## Workstream D — Agent Orchestrator

After external LLM generation, verify factual claims against evidence IDs, numerical claims against model outputs, temporal ordering, contradictions, schema and safety rules. Use deterministic checks for numbers, units, dates, identifiers and schema. Semantic/NLI or LLM-as-judge checks may supplement, not replace, deterministic checks. Permit at most one controlled revision; otherwise abstain or return a safe fallback. Test invented IDs, wrong values/units, future dates, unsupported diagnoses, contradictions, malformed outputs and retry limits.

## Workstream E — Integrated evaluation

Compare snapshot LLM, history-aware LLM, temporal-evidence + approved RAG + constrained LLM, and the same pipeline with Agent Orchestrator verification. Separately compare cache disabled/enabled and rule-based privacy filtering versus SLM + privacy validator. Use fixed cases and the same splits. Never send real identifiable records to an external baseline. If report notes are not validly linked to the temporal cohort, evaluate RAG separately or use labelled synthetic reports.

## Deployment and repository rules

GitHub contains application source, tests, configs without secrets and training/inference scripts. Model weights/adapters may be hosted separately in an access-controlled model registry subject to license and data restrictions. Pin base-model and adapter revisions, cache authorized downloads locally and support offline use where feasible. Benchmark actual hardware; do not promise fixed memory requirements. Never publish patient records, PHI, credentials or sensitive training examples.

Smartphone sensing, wearables and external sensors are future development only and must not appear as MVP implementation tasks or acceptance tests.
