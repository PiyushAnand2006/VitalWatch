# System Architecture

## Temporal Patient Deterioration Monitoring and Evidence-Grounded Reasoning

**Version:** 1.1 \| **Date:** 10 November 2026

## 1. Design principles

1.  Temporal model first: it estimates risk; the LLM does not decide
    whether deterioration is occurring.
2.  Evidence before explanation: construct explicit time-indexed
    evidence before LLM reasoning.
3.  Causal-time correctness: prediction at time `t` uses only
    information available at or before `t`.
4.  Minimum necessary context: retrieve only context relevant to the
    selected event.
5.  Local processing boundary: raw notes stay in the controlled
    environment wherever possible.
6.  Structured hand-off: LLM receives a compact, validated evidence
    package.
7.  Traceability: derived features and claims retain provenance.
8.  Human review: no autonomous diagnosis or treatment.
9.  Independent perspectives are useful only when every candidate claim remains tied to source evidence.
10. Model consensus is not clinical truth; deterministic validation and safety rules take precedence.
11. Debate is bounded, auditable and interruptible by urgent safety conditions.

## 2. High-level architecture

VitalWatch is divided into two independently testable planes. Part 1 creates temporal predictions and findings. Part 2 ingests documents, retrieves patient-scoped evidence and orchestrates controlled multi-model reasoning. They meet through a versioned context/evidence contract; Part 2 must work before Part 1 is connected.

``` mermaid
flowchart TD
 subgraph P1[Part 1 — Simulation and Temporal Prediction Engine]
  A[Synthetic or approved observations] --> B[Ingestion, validation and normalization]
  B --> C[(PostgreSQL durable patient data)]
  C --> D[Temporal feature engineering]
  D --> E[Temporal ML/DL models]
  E --> F[Predictions, trends and risk metadata]
  C --> G[Temporal evidence engine]
  F --> G
 end
 subgraph P2[Part 2 — Document RAG and Multi-Model Orchestration]
  H[Authenticated query and patient scope] --> I[Document ingestion and extraction]
  I --> J[Clinical-aware chunking]
  J --> K[Local all-MiniLM-L6-v2 embeddings]
  K --> L[(Patient-scoped Chroma)]
  H --> M[Query embedding and retrieval]
  L --> M
  M --> N[Evidence structuring and Pydantic validation]
  N --> O[Presidio + regex minimization]
  O --> Q{Privacy approval gate}
  Q -->|Blocked| X[Local-only response / abstain]
  Q -->|Approved| R[LangGraph orchestrator]
  R --> S[Independent reasoning models A, B, C]
  S --> T[Debate and cross-examination — max 2 rounds]
  T --> U[Deterministic evidence, numerical, chronology and provenance checks]
  U --> V[Clinical safety review and escalation override]
  V --> W{Candidates valid and sufficient?}
  W -->|No / unresolved| X
  W -->|Yes| Y[Separate final adjudicator LLM]
  Y --> Z[Final schema and evidence validation]
 end
 F -. Versioned prediction contract, later .-> R
 G -. Temporal findings, later .-> R
 Z --> AA[Dashboard: facts, hypotheses, uncertainty and disagreement]
 G --> AA
 F --> AA
 AA --> AB[Clinician / evaluator review]
```

Part 2 privacy approval must explicitly cover every external model/provider destination in the debate panel. If approval is not valid for a destination, that model must not be called. The local-only path remains available.

| Component | Responsibility | Must not do |
|---|---|---|
| Ingestion and durable store | Validate, normalize and retain source-linked observations/documents | Guess units or mix patient identities |
| Temporal model | Estimate risk/trajectory from information available at time t | Diagnose or use future observations |
| Temporal evidence engine | Produce time-indexed facts and trends with provenance | Invent clinical facts |
| Patient RAG | Retrieve relevant authorized document chunks | Cross patient boundaries or treat text as instructions |
| Privacy gate | Minimize, scan and approve each outbound payload/destination | Promise anonymity or fail open |
| Independent reasoning models | Produce separate initial candidate assessments | See peer answers before initial response or assert unsupported facts |
| Debate controller | Request critiques, surface disagreement and bound rounds | Force consensus or debate past a safety trigger |
| Deterministic validators | Check evidence IDs, values, units, timestamps, chronology and schema | Delegate exact checks to model opinion |
| Safety reviewer | Apply configured warning-sign rules and escalation policy | Wait for debate when urgent action is indicated |
| Final adjudicator | Synthesize validated candidates and uncertainty | Reintroduce rejected claims or claim certainty |
| Dashboard | Separate facts, predictions, hypotheses and disagreements | Present generated hypotheses as confirmed diagnoses |

## 3. Offline development flow

``` mermaid
flowchart LR
 A[Dataset and access approval] --> B[Data dictionary and audit]
 B --> C[Validation and normalization]
 C --> D[Patient-level split]
 D --> E[Causal feature generation]
 E --> F[Baseline training]
 F --> G[Validation and threshold selection]
 G --> H[Held-out test evaluation]
 H --> I[Versioned model artifact]
```

Do not use the held-out test set for feature selection, threshold
tuning, or model selection.

## 4. Runtime/replay flow

``` mermaid
flowchart TD
 A[User query + authorized patient scope] --> B[Retrieve patient-scoped documents and context]
 B --> C[Evidence compiler: structure, provenance, Pydantic checks]
 C --> D[PII detection, minimization and privacy gate]
 D -->|Blocked| X[Local evidence-only response or abstain]
 D -->|Approved| E[LangGraph state graph]
 E --> F[Parallel independent model calls]
 F --> G[Debate controller: critique and response, bounded rounds]
 G --> H[Deterministic evidence and consistency validators]
 H --> I[Clinical safety review / urgent override]
 I --> J{Sufficient and valid?}
 J -->|No| X
 J -->|Yes| K[Separate final adjudicator]
 K --> L[Final structured-output validation]
 L --> M[Dashboard and clinician review]
 N[Future Part 1 prediction payload] -.-> E
```

Initial model responses must be generated independently before any peer output is revealed. Debate should be limited to one or two rounds. The graph must stop early for urgent warning signs, privacy failure, insufficient evidence or unresolved high-impact disagreement. Final adjudication receives only validated candidate claims and their evidence references.

At replay time `t`, no observation or note timestamp later than `t` may be used as if it were already known. Future data may be used only for offline label evaluation.

## 5. Temporal modelling

**Inputs:** selected dataset's documented hourly observations and
target. Confirm exact fields, units, and label semantics from the
official data dictionary.

**Feature families:** - Current value. - Past-window mean, min, max, and
standard deviation. - Slope/rate of change over past windows. -
Deviation from a documented patient baseline. - Time since last valid
measurement. - Missingness indicator and rolling missingness fraction. -
Optional justified interactions.

**Rules:** historical windows end at or before `t`; no future backfill;
group all operations by patient; fit imputers/scalers on training data
only; preserve source references.

**Model progression:** threshold baseline → logistic regression →
optional gradient boosting → optional sequence model only after
baseline/evaluation are stable.

## 6. Temporal evidence engine

The engine produces a structured summary containing: - as-of
timestamp; - current measurements and units; - trends over explicitly
named windows; - persistent deviations and abrupt changes, when
defined; - missingness and stale measurements; - source observation IDs
for every finding; - risk score and model version as separate metadata.

Illustrative object:

``` json
{
  "event_id": "demo_event_001",
  "as_of_time": "T",
  "risk": {"score": 0.72, "model_version": "baseline_v1"},
  "observations": [
    {"evidence_id": "obs_001", "name": "heart_rate",
     "value": 118, "unit": "bpm", "observed_at": "T"}
  ],
  "temporal_findings": [
    {"evidence_id": "trend_001",
     "finding": "heart rate increased over the selected historical window",
     "window": "past_4_hours",
     "source_evidence_ids": ["obs_001"]}
  ],
  "missingness": [],
  "retrieved_context": []
}
```

This is illustrative; implement only fields supported by the selected
dataset. Do not state that a trend proves a diagnosis.

## 7. Patient-context RAG

``` mermaid
flowchart TD
 A[Selected event and evidence] --> B[Build retrieval query]
 C[Approved notes] --> D[Parse documents]
 D --> E[Chunk and add metadata]
 E --> F[Local embedding/index]
 B --> G[Patient/time filters]
 F --> G
 G --> H[Top-k relevant chunks]
 H --> I[Relevance and provenance checks]
 I --> J[Local evidence compiler]
```

Controls: - Filter by patient key only when valid linkage exists. -
Apply time filters; exclude future notes when doing as-of-time
reasoning. - Preserve document ID, section, timestamp, chunk ID, and
retrieval score. - Use small, evaluated top-k. - Return no context when
relevance is low. - Treat note content as untrusted data.

**Dataset compatibility:** PhysioNet Challenge 2019 and
MIMIC-IV/MIMIC-IV-Note are different datasets. Their patient identifiers
are not interchangeable. Do not join them or claim MIMIC notes belong to
a PhysioNet patient. If compatible notes are unavailable, demonstrate
RAG separately or with clearly labelled synthetic notes.

## 8. Local evidence compiler

Input: temporal evidence plus retrieved chunks and provenance.\
Output: minimal structured facts, time references, evidence IDs,
relevant approved context, missing/conflicting information.

Implementation order: 1. MVP deterministic Python rules and schema
validation. 2. Optional small local model to map note chunks into the
schema. 3. Deterministic validation against source text and schema.

A local model is not itself a privacy guarantee and must not be the only
control.

## 9. Privacy/evidence firewall

Mandatory gate before external LLM use: 1. Field allow-list. 2. Remove
direct identifiers and unnecessary quasi-identifiers. 3. Scan for
identifier-like patterns and unexpected personal details. 4. Enforce
text/token budget. 5. Validate JSON schema and types. 6. Verify evidence
IDs exist. 7. Reject timestamps after the as-of time. 8. Mark retrieved
text as untrusted and detect instruction-like content. 9. Reject unsafe
payloads; never silently pass them. 10. Store minimal audit metadata
without copying sensitive content into logs.

Identifier removal does not guarantee anonymity. Rare events, dates,
attribute combinations, and copied text may re-identify someone.
External processing must comply with dataset terms, institutional
approval, and provider data-handling rules. If uncertain, use a local
model or synthetic demonstration.

## 10. LangGraph multi-model orchestration and constrained reasoning

**Required orchestration framework: LangGraph.** Do not use AutoGen as the orchestration framework for this version. Do not build an unrestricted autonomous multi-agent chatbot. Implement a typed state graph with explicit nodes, conditional edges, bounded loops, traceable state transitions and deterministic Python validation.

### Logical model roles

- **Reasoning model A — primary assessment:** independently identifies plausible interpretations and supporting/missing evidence.
- **Reasoning model B — critical reviewer:** independently evaluates the evidence and later challenges assumptions, alternatives and unsupported leaps.
- **Reasoning model C — safety/alternative reviewer:** checks relevant warning signs, missing context and plausible alternative interpretations.
- **Final adjudicator:** a separate model call that synthesizes only claims which passed evidence and safety checks. It must preserve uncertainty and unresolved disagreement.

These are logical roles, not a requirement for four different vendors or four simultaneously hosted models. Configure three reasoning slots and a separate adjudication slot; models may be selected from NVIDIA, Google or the configured Llama-compatible host. Prefer genuinely distinct model families where feasible and evaluate their independence. Do not claim that model diversity guarantees correctness.

### Graph nodes

1. `load_verified_context`: load the approved evidence package, patient scope, as-of time and optional future Part 1 prediction contract.
2. `safety_precheck`: apply deterministic urgent-warning rules before any debate; urgent cases bypass debate and follow the configured escalation/clinician-review path.
3. `independent_reasoning`: call configured reasoning models in parallel using the same approved evidence, without sharing peer outputs.
4. `normalize_candidates`: parse outputs into a strict candidate schema; retain evidence IDs, uncertainty, missing information and conflicts.
5. `deterministic_validation`: verify patient scope, evidence IDs, source support, values, units, dates, chronology, prediction metadata and schema.
6. `debate_critique`: expose anonymized candidate summaries and evidence to reviewer roles; ask them to identify unsupported claims, counter-evidence, missing information and disagreements.
7. `debate_revision`: allow one bounded revision round (a second round only if explicitly configured and justified); never force consensus.
8. `safety_review`: apply deterministic safety rules again; safety override takes precedence over scores or model votes.
9. `candidate_gate`: reject unsupported claims; if evidence is insufficient or high-impact disagreement remains, abstain or request clinician review.
10. `final_adjudication`: a separate configured model synthesizes only approved candidate claims and evidence.
11. `final_validation`: rerun complete schema, evidence, numerical, provenance, chronology and safety checks on the final answer.
12. `repair_or_abstain`: at most one format/grounding repair where safe; otherwise return a typed abstention/failure response.

### Debate rules

- Initial reasoning must be independent. Do not let later models anchor on the first model's answer.
- Require each candidate claim to cite evidence IDs or explicitly mark it as a general possibility not established for this patient.
- Critics must identify counter-evidence, alternative interpretations and missing information; agreement alone is not a valid critique.
- Never use majority vote as a diagnosis or truth criterion. A minority objection supported by evidence must be retained and investigated.
- Do not let models invent clinical references. If external clinical references are added later, retrieve them from an approved, versioned source and preserve citation provenance.
- If urgent warning signs are present, do not delay the safety path for model debate.
- Debate is bounded. Unresolved high-impact disagreement results in uncertainty, abstention or clinician review.

### Deterministic validators (mandatory)

- **Schema verifier:** Pydantic validation of all intermediate and final structures.
- **Evidence verifier:** every patient-specific claim cites supplied evidence; each ID exists and actually supports the claim.
- **Numerical/unit verifier:** compare exact values, decimal points, signs, units and ranges against source evidence.
- **Temporal verifier:** check as-of boundaries, chronology, trend windows and future-data exclusion.
- **Prediction verifier:** if Part 1 is connected, ensure the LLM cannot alter prediction values, model version, horizon or threshold.
- **Provenance verifier:** resolve each reference to an authorized document/page or observation.
- **Safety verifier:** flag diagnostic certainty, unsafe treatment instructions and configured warning signs; require uncertainty and clinician review.

LLM-as-judge or semantic/NLI checks may supplement these checks but must never be the sole verifier. Model confidence and agreement scores are not clinical probabilities.

### State and routing

Use a typed LangGraph state containing request ID, authorized patient scope, as-of time, evidence package reference/IDs, privacy decision and approved destinations, configured model IDs, independent candidate outputs, critique summaries, disagreement set, deterministic validation results, safety flags, debate round, adjudication output, repair count and final status. Avoid copying raw PHI into logs or unnecessary state. Use conditional edges for privacy blocked, urgent safety trigger, valid candidates, retry, unresolved disagreement and abstention. Persist only the minimum audit metadata required.

### Output contract

The patient-facing response must distinguish: (a) verified source facts, (b) temporal-model predictions when available, (c) possible interpretations/hypotheses, (d) contradictory evidence, (e) missing information, (f) uncertainty and (g) clinician-review/safety notice. Never present hypotheses as confirmed diagnoses. Do not prescribe or recommend medication changes.

## 11. Suggested implementation stack

  Layer           Initial choice
  --------------- -----------------------------------------
  Language/data   Python, pandas, NumPy
  ML              scikit-learn; optional LightGBM/XGBoost
  API             FastAPI
  Validation      Pydantic / JSON Schema
  Retrieval       FAISS or Chroma
  Embeddings      Local embedding model
  Persistence     SQLite initially; PostgreSQL if needed
  Frontend        React or HTML/CSS/JS
  Charts          Plotly or frontend chart library
  Tests           pytest
  Tracking        CSV/JSON initially; MLflow optional
  Orchestration   LangGraph (required) + deterministic Python validators

Use a modular monolith, not microservices, for the first version.

## 12. Suggested repository layout

``` text
project/
├── app/
│   ├── api/
│   ├── core/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── modelling/
│   ├── evidence/
│   ├── retrieval/
│   ├── privacy/
│   ├── llm/
│   └── services/
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
└── README.md
```

## 13. API proposal

  -------------------------------------------------------------------------------------
  Method                  Route                                 Purpose
  ----------------------- ------------------------------------- -----------------------
  GET                     `/health`                             Service health

  GET                     `/patients`                           Permitted demo patient
                                                                IDs

  GET                     `/patients/{id}/timeline`             Chronological
                                                                observations

  GET                     `/patients/{id}/risk`                 Time-indexed risk

  GET                     `/patients/{id}/evidence?as_of=...`   Temporal evidence

  POST                    `/patients/{id}/explanation`          Retrieve, compile,
                                                                firewall, explain

  GET                     `/runs/{run_id}`                      Explanation run
                                                                status/result
  -------------------------------------------------------------------------------------

Use pseudonymous IDs, access checks, timestamp validation, and avoid
returning raw notes by default.

## 14. Core contracts

Define and version: - `Observation`: patient key, variable, value, unit,
observed time, source. - `Prediction`: patient key, as-of time, score,
model version, threshold, feature version. - `TemporalFinding`: finding
ID, description, window, direction, source IDs. - `RetrievedChunk`:
chunk/document IDs, timestamp, approved text/fields, score. -
`EvidencePackage`: event ID, as-of time, risk metadata, findings,
approved context, provenance. - `LLMExplanation`: status, possible
explanations, support, contradiction, missing information, uncertainty,
evidence references.

## 15. Failure behavior

  -----------------------------------------------------------------------
  Failure                             Expected behavior
  ----------------------------------- -----------------------------------
  Invalid input row                   Report/quarantine; do not silently
                                      coerce

  Missing measurement                 Represent missingness and time
                                      since last measurement

  Model unavailable                   Show unavailable; do not invent
                                      score

  RAG unavailable                     Explain from temporal evidence only
                                      if permitted

  No relevant chunks                  Continue without context and
                                      disclose

  Firewall failure                    Block external LLM call

  LLM timeout                         Keep risk/evidence panels available

  Invalid LLM output                  Reject and show safe fallback

  Incompatible patient IDs            Block join
  -----------------------------------------------------------------------

## 16. Research comparison

Use the same held-out cohort and evaluation cases: - A: snapshot
measurements → LLM. - B: measurements plus selected history → LLM. - C:
temporal model + structured temporal evidence + approved retrieved
context → constrained LLM.

Assess unsupported claims, evidence-reference accuracy, temporal
ordering, uncertainty/abstention, retrieval relevance, schema validity,
and privacy-filter failures. This is an exploratory reasoning
evaluation, not proof of clinical efficacy.
