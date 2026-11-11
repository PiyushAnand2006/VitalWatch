# Product Requirements Document (PRD)

## Temporal Patient Deterioration Monitoring and Evidence-Grounded Reasoning

**Version:** 1.1 \| **Date:** 10 November 2026 \| **Status:** University minor-project draft\
**Working title:** Not finalized

## 1. Product summary

Build a software prototype that monitors patient observations over time,
estimates deterioration risk, explains the temporal evidence behind that
risk, and optionally uses relevant patient context to generate
constrained explanations for clinician review.

The system has two independently developable subsystems: **Part 1 — Simulation and Temporal Prediction Engine** (longitudinal data generation, temporal modelling, risk/trend outputs) and **Part 2 — Document RAG and Multi-Model Orchestration** (document ingestion, patient-scoped retrieval, privacy filtering, controlled model debate, evidence verification and explanation). They connect through a versioned evidence/prediction contract. The complete product also includes durable patient-data storage, a short-lived context cache, a temporal evidence engine and a clinician-review dashboard.

**One-line pitch:** A modular longitudinal healthcare research prototype that combines temporal prediction, patient-specific evidence retrieval and controlled multi-model reasoning to explain supported findings and uncertainty for clinician review.

## 2. Problem

A single measurement may hide a clinically important trajectory.
Persistent abnormality, rapid change, deviation from baseline,
combinations of observations, and missing measurements can all matter. A
risk model may produce a score without explaining the time pattern. A
general LLM may generate fluent but unsupported explanations when given
raw records.

This project separates detection from explanation: the temporal model
estimates risk; the evidence engine establishes what changed; the LLM
may explain only the approved evidence package.

## 3. Goals

-   Ingest an approved, de-identified time-series dataset.
-   Build reproducible preprocessing and patient-level data splits.
-   Generate causal features using only information available at
    prediction time.
-   Train and evaluate transparent baseline risk models.
-   Replay patient timelines and display risk trajectories.
-   Generate traceable temporal evidence summaries.
-   Demonstrate patient-context retrieval where a compatible notes
    dataset is available.
-   Compile retrieved material into a strict structured evidence
    package.
-   Minimize identifying information before any external LLM call.
-   Produce possible explanations, counter-evidence, missing
    information, and uncertainty.
-   Keep clinician review in the loop.
-   Use a bounded LangGraph workflow to obtain independent model perspectives, challenge unsupported conclusions and synthesize only evidence-validated findings.
-   Evaluate whether multi-model debate improves factual support and safety compared with a single-model baseline.

## 4. Non-goals and safety boundaries

The prototype does not diagnose, prescribe, recommend treatment,
autonomously escalate care, replace clinicians, or claim prospective
clinical effectiveness. It does not guarantee anonymity or privacy. It
must not send raw patient notes or direct identifiers to an external
LLM. It must not join unrelated datasets or use future observations in
earlier predictions. All outputs are research decision support for
review only.

## 5. Users and user stories

**Reviewer/evaluator:** selects a patient and time point; replays
observations; sees risk trajectory; inspects evidence; reviews possible
explanations and uncertainty.

**Student developer/researcher:** runs data preparation, training,
evaluation, replay, retrieval, and experiments reproducibly.

User stories: - As a reviewer, I can inspect observations and risk at a
selected time. - I can see which trends and missingness patterns
contributed to the evidence summary. - I can see supporting and
contradictory evidence for each proposed explanation. - I can see when
evidence is insufficient. - As a developer, I can reproduce a result
from a fixed dataset split and configuration.

## 6. Scope

### MVP --- must deliver

-   One approved time-series dataset.
-   Data dictionary, cleaning, and patient-level splits.
-   Causal rolling features and missingness indicators.
-   Threshold baseline and logistic regression; optional tree model.
-   Evaluation report including AUROC, AUPRC, sensitivity, calibration,
    false alerts per patient-day, and lead time where measurable.
-   Patient timeline replay dashboard.
-   Deterministic temporal evidence summary.
-   Clear separation of model risk and LLM explanation.

### Part 2 implementation (develop independently first)

-   Document ingestion, local embeddings and patient-scoped Chroma retrieval.
-   Deterministic evidence structuring and Presidio/regex privacy gate.
-   LangGraph multi-model reasoning workflow with independent initial responses, bounded debate, deterministic evidence/safety checks and a final adjudicator.
-   NVIDIA, Google and configurable Llama-compatible provider adapters.
-   Part 2 must operate without Part 1; use synthetic or separately approved records until compatible patient-linked data are available.

### Extension --- should deliver if time/data access allow

-   Integration of Part 1 prediction payloads into Part 2 context.
-   More advanced temporal models, external validation and robust clinical-reference retrieval.
-   Clinician feedback and expanded evaluation.

### Stretch

-   Second-dataset validation, local LLM compiler, advanced sequence
    model, robustness and subgroup analyses.

## 7. Data requirements

**Temporal MVP dataset:** PhysioNet Computing in Cardiology Challenge
2019: https://physionet.org/content/challenge-2019/1.0.0/\
Use its documented hourly records and label after reviewing official
terms, variable definitions, and label construction. Treat the label as
the dataset target, not a universal definition of deterioration.

**Notes/RAG candidate:** MIMIC-IV plus MIMIC-IV-Note: -
https://physionet.org/content/mimiciv/ -
https://physionet.org/content/mimic-iv-note/

Access may require credentialing, training, institutional
affiliation/supervision, and a data-use agreement. Do not assume
PhysioNet Challenge 2019 IDs match MIMIC IDs. If data are not linkable,
demonstrate RAG separately or with clearly labelled synthetic notes;
never present unrelated notes as belonging to a temporal-model patient.

Data governance: store data only in approved locations; do not commit
raw data or credentials; record dataset version, access conditions,
preprocessing version, and split strategy.

## 8. Functional requirements

  ------------------------------------------------------------------------------
  ID                Requirement        Priority          Acceptance condition
  ----------------- ------------------ ----------------- -----------------------
  FR-01             Load supported     Must              Schema and types
                    time-series data                     validated

  FR-02             Normalize          Must              Invalid rows and unit
                    timestamps/units                     issues reported

  FR-03             Split by patient   Must              No patient overlap
                                                         across partitions

  FR-04             Generate causal    Must              No future data used at
                    features                             time t

  FR-05             Train baseline     Must              Reproducible from
                    model                                config

  FR-06             Evaluate model     Must              Metrics and threshold
                                                         documented

  FR-07             Replay timeline    Must              Chronological replay
                                                         works

  FR-08             Display risk       Must              Risk updates with
                    trajectory                           selected time

  FR-09             Generate temporal  Must              Findings cite
                    evidence                             measurements/windows

  FR-10             Retrieve relevant  Should            Ranked chunks retain
                    notes                                provenance

  FR-11             Compile structured Should            Output validates
                    evidence                             against schema

  FR-12             Apply privacy      Should            Unsafe payload is
                    checks                               rejected

  FR-13             Generate           Should            Output conforms to
                    constrained                          schema
                    explanation

  FR-14             Abstain when       Should            Explicit
                    evidence is weak                     insufficient-evidence
                                                         state

  FR-15             Record run         Should            Model, prompt, and
                    metadata                             retrieval versions
                                                         recorded

  FR-16             Independent       Should            Reasoning models return
                    model perspectives                  separate initial outputs
                                                        before peer exposure

  FR-17             Bounded debate    Should            Critique/revision is
                    and critique                        limited to configured
                                                        rounds and logged

  FR-18             Final adjudication Should           Separate adjudicator
                                                        synthesizes only claims
                                                        passing evidence/safety
                                                        checks

  FR-19             Safety override   Must              Urgent warning signs
                    and abstention                       bypass debate; unresolved
                                                        cases abstain or request
                                                        clinician review
  ------------------------------------------------------------------------------

## 9. Non-functional requirements

-   Reproducible data split, preprocessing, and model training.
-   Risk score and LLM explanation shown separately.
-   Every temporal claim linked to timestamped source observations.
-   Missingness shown rather than silently filled.
-   Raw records kept inside the controlled environment wherever
    possible.
-   External LLM receives only firewall-approved minimal evidence.
-   Retrieved text is treated as untrusted data, not instructions.
-   API keys stored outside source control.
-   Dashboard remains usable if RAG or LLM is unavailable.

## 10. Product workflow

1.  Import approved data.
2.  Validate and normalize observations.
3.  Create patient-level splits and causal features.
4.  Train/evaluate risk model.
5.  Select patient and replay time in dashboard.
6.  Generate temporal evidence.
7.  Optionally retrieve compatible patient context.
8.  Compile a minimal structured evidence package locally.
9.  Run privacy/evidence firewall.
10. Pass only the approved, minimized evidence package to the explicitly configured debate-model panel; each destination must be covered by the privacy decision.
11. Obtain independent initial model assessments, then run a bounded LangGraph critique/debate round.
12. Run deterministic evidence, numerical, chronology, provenance and safety checks; urgent warning signs take the safety path immediately.
13. Send only validated candidate conclusions to a separate final adjudicator for structured synthesis.
14. Validate the final response and evidence references; abstain if unresolved.
15. Display supported facts, hypotheses, disagreement, uncertainty and review flags for clinician review.

## 11. LLM output contract

Required fields: status (`generated` or `insufficient_evidence`),
possible explanations, supporting evidence IDs, contradictory evidence
IDs, missing information, temporal summary, uncertainty, and safety
note.

Models must not invent observations, timestamps, history, or evidence IDs; diagnose with certainty; recommend treatment; or claim causality from correlation. Independent model agreement is not proof of correctness and must never replace source verification. The final adjudicator may synthesize only candidate claims that pass deterministic evidence and safety checks. It must preserve meaningful disagreement and abstain when evidence is inadequate.

## 12. Success metrics

**Risk model:** AUROC, AUPRC, sensitivity, specificity, precision,
calibration/Brier score, false alerts per patient-day, warning lead time
where label definition permits.

**Evidence/orchestration:** valid evidence-reference rate, unsupported-claim rate, factual consistency, contradiction detection, appropriate abstention, safety-escalation handling, debate disagreement resolution, latency and token/API cost. Compare single-model, multi-model without debate, and multi-model with bounded debate on the same test cases.

**RAG/privacy:** retrieval relevance, provenance accuracy, schema-valid rate, outbound privacy-filter rejection tests, and zero external calls when the privacy gate blocks.

**Demo acceptance:** reviewer can replay a patient, see risk change,
inspect the underlying temporal evidence, and distinguish the LLM's
hypothesis from the model's risk estimate.

## 13. Risks and mitigations

  -----------------------------------------------------------------------
  Risk                                Mitigation
  ----------------------------------- -----------------------------------
  Notes access delayed                Complete temporal MVP first; use
                                      clearly labelled synthetic notes
                                      for UI tests

  Dataset mismatch                    Never join unrelated cohorts;
                                      document valid linkage

  Imbalanced labels                   Report AUPRC, sensitivity,
                                      calibration, and alert burden

  Temporal leakage                    Causal windows and future-data
                                      tests

  Missing/irregular data              Missingness features and robustness
                                      analysis

  LLM hallucination                   Independent reasoning, source-linked
                                      claims, deterministic validators,
                                      bounded debate, abstention and review

  Model consensus error               No majority-vote diagnosis; preserve
                                      disagreement; final synthesis only
                                      after evidence and safety checks

  Debate latency/cost                 Limit panel size and debate rounds;
                                      parallelize independent calls; log
                                      usage metadata without PHI

  Privacy leakage                     Minimize fields, scan outbound
                                      payload, avoid raw notes

  Scope creep                         Freeze MVP; treat RAG/LLM as staged
                                      extension

  LLM outage                          Deterministic evidence-only
                                      fallback
  -----------------------------------------------------------------------

## 14. Open decisions

Final title; cohort and dataset release; target and prediction horizon;
observation window; model and threshold; notes access; local vs external
LLM; semester deadline; team roles and compute.

## 15. Definition of done

Repository setup is reproducible; data provenance and access rules are
documented; leakage tests pass; baseline training/evaluation works;
dashboard replays a patient; risk and evidence are traceable; LLM
outputs (if implemented) are validated and labelled as hypotheses;
failure paths are safe; limitations are explicit.
