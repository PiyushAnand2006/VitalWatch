# VitalWatch Part 2 Incremental Implementation Plan

## 1. Repository analysis

### Specification reviewed
- Read all 787 lines of `docs_pdf/VitalWatch_master_prompt_updated.md`; it is the controlling implementation specification.
- Cross-checked `docs_pdf/architecture.md`, `docs_pdf/implementation_plan.md`, and `docs_pdf/prd (1).md` for context.
- Where older documents mention temporal models, Redis, FAISS, SQLite, Chroma, or an internal SLM, the updated approved architecture takes precedence: Part 2 only, PostgreSQL + pgvector, LangGraph, no SLM, and no Part 1 implementation.

### Current repository and dependencies
- Root contents include `.gitignore`, `.mimosa/`, `architecture_png/`, `docs_pdf/`, `finetune/`, `ppt/`, `app/`, `tests/`, `alembic/`, `alembic.ini`, `pyproject.toml`, `.env.example`, and `README.md`.
- Python 3.12.10 is available and satisfies the Python 3.11+ requirement.
- Core runtime dependencies in `pyproject.toml` provide FastAPI, Pydantic v2, PostgreSQL (via `psycopg`), SQLAlchemy 2.x, Alembic, pgvector, and Uvicorn. Dev dependencies provide Pytest, HTTPX, Ruff, and strict Mypy.
- `finetune/` contains an SLM dataset builder, training notebooks, inference examples, adapter artifacts, and reports. These artifacts are explicitly excluded from Part 2 and remain isolated, unimported, and untracked.

### Approved technology stack

| Component | Technology | Role & Architecture Boundary |
|---|---|---|
| Primary database | **PostgreSQL** | Authoritative relational store for users, patient grants, document metadata, chunks/provenance, conversations, privacy approvals, audit events, and prediction metadata. |
| ORM & Data Layer | **SQLAlchemy 2.x** | Explicit typed models and repositories; async engine with connection pooling (`create_async_engine`), transaction scoping, and dependency injection outside route handlers. |
| Database migrations | **Alembic** | Version-controlled schema migrations; automated execution via `alembic upgrade head`; no manual schema creation. |
| Vector database functionality | **pgvector** | PostgreSQL extension enabling vector similarity search (384-dim embeddings) alongside relational metadata in patient-scoped queries; HNSW indexing. |
| Embedding model | **`sentence-transformers/all-MiniLM-L6-v2`** | Local embeddings (384 dimensions); cached locally; strictly zero remote embedding API calls. |
| Document storage | **Encrypted private local storage** | Local filesystem storage confined to private root directories with strict path-traversal prevention; clean abstraction (`DocumentStorage` ABC) enabling seamless future migration to S3-compatible object storage. |
| Future document storage | **S3-compatible object storage** | Deferred target for production document binaries; interface-compatible with `DocumentStorage`. |
| Cache / Queue | **Redis (Deferred)** | Strictly deferred; not part of the active implementation. |
| Orchestration | **LangGraph** | Multi-agent reasoning, bounded debate, deterministic verification, and adjudication. |
| Privacy Engine | **Microsoft Presidio + Custom Rules** | Mandatory fail-closed redaction and destination-specific cryptographic approval gate. |

### Component status

**Completed, but documentation/design only**
- Part 2 requirements, architecture, API proposal, safety boundaries, and acceptance criteria are documented.
- Existing diagrams and pitch materials are present; they are not executable implementation.

**Completed or partially completed outside this task’s scope**
- `finetune/` contains an SLM dataset builder, training notebooks, inference examples, adapter artifacts, and reports.
- These artifacts are explicitly excluded from Part 2 and will remain unchanged and unimported.
- `finetune/build_dataset.py` provides useful synthetic test-case ideas, and the inference scripts contain basic regex checks. They are references only, not reusable Part 2 privacy or evidence implementations: they lack Presidio, Pydantic v2 contracts, destination-bound approval, application integration, and the required test suite.

**Completed foundation (Stage 1)**
- `.gitignore` and `.env.example` configured to protect secrets, `.env`, private runtime databases, uploaded documents, local model caches, Python caches, and test artifacts.
- Application factory (`create_app`), typed environment settings (`Settings`) with strict validation, request ID propagation middleware (`RequestIDMiddleware`), safe public error handlers, redacted logging (`RedactingFormatter`), and minimal audit logging sink.
- System liveness (`/api/v1/health`) and readiness (`/api/v1/ready`) endpoints.
- PostgreSQL + SQLAlchemy 2.x session manager (`DatabaseSessionManager`), Alembic migration runner (`alembic/env.py`), initial pgvector migration (`0001_enable_pgvector.py`), and `DocumentStorage` abstraction with `LocalDocumentStorage`.
- Automated test suite (38 tests) passing 100% with Ruff and strict Mypy validation.

**Missing Part 2 implementation (Stages 2–11)**
- Authentication, patient authorization, PostgreSQL repositories, patient access grants, audit persistence, and rate limiting.
- Secure document lifecycle, extraction/OCR, chunking, and provenance.
- Local embeddings (`sentence-transformers/all-MiniLM-L6-v2`), pgvector storage, patient-scoped similarity search, document-level deletion, and re-indexing.
- Evidence schemas/compiler and deterministic validation.
- Presidio/regex minimization and fail-closed, destination-specific privacy approval.
- NVIDIA, Google, and configurable Llama-compatible provider adapters and routing.
- LangGraph independent reasoning, bounded debate, validators, adjudication, repair, and abstention.
- Query/retrieval APIs, conversation scoping, future prediction contract, tests, and setup documentation.

## 2. Approved-scope changes proposed

Create a modular monolith using the approved layout:
- `app/main.py`, `app/core/`, `app/api/v1/`, `app/schemas/`, `app/services/`, `app/repositories/`.
- `app/document_processing/`, `app/vectorstore/`, `app/privacy/`, `app/llm/`, `app/orchestration/`, and `app/integrations/`.
- `alembic/` and `alembic.ini` for versioned PostgreSQL and pgvector schema migrations.
- `tests/unit/`, `tests/integration/`, `tests/security/`, `tests/contract/`, and synthetic-only `tests/fixtures/`.
- Root `pyproject.toml`, `.env.example`, and `README.md`; expanded `.gitignore` for secrets and private runtime artifacts.
- Private runtime `data/documents/` (encrypted local storage) and PostgreSQL database storage, all excluded from source control.

Do not move or delete existing directories, initialize microservices, add Redis, create a frontend, implement temporal prediction, or integrate the SLM. No existing frontend exists, so the API/UI phase delivers FastAPI/OpenAPI only.

## 3. Fixed architectural decisions and invariants

- **Persistence (PostgreSQL & SQLAlchemy 2.x):** PostgreSQL is the single authoritative relational database. SQLAlchemy 2.x declarative models and repositories handle users, patient access grants, document metadata, chunk records, conversation sessions, privacy approval logs, and minimal audit events. Repository interfaces isolate database queries from services and API handlers.
- **Transactions & Connection Management:** Explicit transactions wrap all multi-record mutations; sessions are injected via dependency injection and closed cleanly. The application startup remains decoupled from a live database connection (checked via readiness checks).
- **Database Migrations (Alembic):** All schema changes, table definitions, indexes, and extensions are applied via version-controlled Alembic migrations (`alembic upgrade head`). No manual table creation is permitted.
- **Vector Storage & Similarity Search (pgvector):** Document chunk embeddings are stored directly in PostgreSQL using the `pgvector` extension.
  - Dimension: Exactly 384 dimensions matching `sentence-transformers/all-MiniLM-L6-v2`.
  - Compatibility metadata: Every vector entry and index stores model identity, revision, and dimension; mixing incompatible models in the same index is strictly prohibited and rejected.
  - Indexing: HNSW (Hierarchical Navigable Small World) index configured for similarity search with cosine or L2 distance.
  - Tenant & Patient Isolation: All similarity queries enforce mandatory SQL filtering by authorized patient ID before or during vector ranking; cross-patient vector leakage is structurally impossible.
  - Lifecycle: Document deletion triggers transactional deletion of all associated chunks and vector rows; re-indexing is atomic.
  - Testing: Filtered vector search queries must be verified through automated tests rather than assuming index defaults.
- **Authentication/authorization:** HTTP Bearer JWT authentication with validated configuration and server-side user-to-patient grants. A path/query patient ID never grants access. Tests mint synthetic tokens; production identity-provider integration remains a documented deployment concern.
- **Local data boundary:** Originals, OCR text, raw chunks, embeddings, vector tables, and full clinical evidence stay local. Only minimized, structured evidence enters an approved external request.
- **Document storage abstraction:** Document binaries are stored in private local directories with optional encryption at rest, managed behind a `DocumentStorage` interface that ensures path-traversal prevention and enables drop-in migration to S3-compatible object storage.
- **Embedding boundary:** Both document and query vectors use local `sentence-transformers/all-MiniLM-L6-v2`; no hosted embedding calls. Cache the model locally and record model identity/revision and dimension.
- **Exact-payload privacy approval:** Minimize and serialize an outbound request once, scan all non-secret text and metadata, and bind approval to its SHA-256 digest plus provider, endpoint/model destination, purpose, and request ID. Transport sends only that approved artifact; any mutation invalidates approval.
- **Every external call is gated:** Initial reasoning, critique, revision, adjudication, repair, and fallback each require approval for their exact payload and destination. A new debate round or fallback cannot reuse an approval for different content or another destination.
- **Fail closed:** `rejected`, `needs_review`, disabled external LLMs, Presidio failure, serialization mismatch, or gate failure result in zero provider calls. Local upload, retrieval, and evidence display remain available.
- **No unverified prose success:** Strict, versioned Pydantic candidate/final schemas distinguish source facts, possibilities, conflicts, missing information, uncertainty, and review flags. Patient-specific claims require evidence IDs and machine-checkable source fields; malformed output is rejected.
- **Exact clinical values:** Preserve source numeric lexemes and use Decimal-based comparisons where applicable. Do not silently convert units, infer missing data, or resolve source conflicts.
- **LangGraph control:** Typed state, explicit nodes/conditional edges, bounded loops and provider-call budgets. Initial A/B/C calls are parallel and receive identical approved evidence with no peer output. Default to one debate round; two only by explicit configuration. Never force consensus.
- **Safety and adjudication:** Deterministic urgent-warning rules bypass debate; deterministic schema, evidence, numeric/unit, temporal, consistency, safety, provenance, and later prediction checks gate candidates. A separate adjudicator call receives only validated claims, evidence IDs, disagreements, and uncertainty. Final validation permits at most one repair, then typed abstention.
- **Audit without PHI:** Record request/node/provider/model/latency/usage/status metadata where available, never raw prompts, completions, detected PII, documents, or secrets. Do not expose chain-of-thought.
- **Testing:** All records are fabricated and labelled synthetic. Automated provider tests use mocks and require no paid keys. Live provider smoke tests are separate, opt-in, and never required for a stage-pass claim.

## 4. Implementation stages and test gates

After each stage, run its focused tests and then the complete current suite with `python -m pytest`. Do not advance with failing required tests. If Tesseract, the local embedding model, or PostgreSQL is unavailable, report that acceptance item as blocked/incomplete rather than treating a skipped test as success.

### Stage 0 — Inspection and plan
- [x] Read the master prompt completely and inspect the repository.
- [x] Identify current code/dependencies, scope conflicts, and missing components.
- [x] Establish the no-SLM/no-Part-1 boundary and test-first incremental sequence.
- [x] Align database architecture with PostgreSQL + SQLAlchemy 2.x + Alembic + pgvector.
- [x] Obtain approval before creating the application structure.

### Stage 1 — Foundation
- [x] Add pinned, compatible project dependencies and pytest/ruff/mypy configuration in `pyproject.toml`.
- [x] Implement typed environment settings, secure defaults, startup validation, application factory, request IDs, safe exceptions, redacted logging, and minimal audit interfaces.
- [x] Add `/api/v1/health` and `/api/v1/ready`; readiness reports dependency status without paths or secrets.
- [x] Add `.env.example` placeholders and expand `.gitignore`; external LLMs default to disabled.
- [x] Add initial setup/limitations documentation and synthetic test configuration.
- [x] Establish PostgreSQL database configuration, SQLAlchemy 2.x DeclarativeBase, Alembic setup, initial pgvector migration, and document storage abstraction.
- **Tests:** settings validation, startup failure for unsafe configuration, request IDs, safe errors, secret/PHI log redaction, liveness, readiness, database manager instantiation, and local storage CRUD.
- **Gate:** report actual focused/full-suite results; health endpoints and database initialization pass through FastAPI test requests without requiring a live database.

### Stage 2 — Authorization and secure document lifecycle
- [ ] Add PostgreSQL models/repositories using SQLAlchemy 2.x: users, patient profiles, patient access grants, and document metadata.
- [ ] Add Alembic migration for Stage 2 database tables.
- [ ] Implement JWT authentication, server-side patient grants, authorization dependencies, pagination, and route audit events.
- [ ] Implement private upload storage using `DocumentStorage` abstraction with generated filenames, extension/MIME/signature/size checks, SHA-256 checksums, duplicate detection, and protected source metadata.
- [ ] Implement `uploaded`, `processing`, `processed`, `failed`, and `quarantined` transitions plus explicit reprocessing and retention/deletion metadata.
- [ ] Add document upload/list/status/reprocess/delete routes with consistent errors; wire vector deletion through a repository interface for Stage 4.
- [ ] Add configurable request/user rate limits and safe handling for malformed or active-content files.
- **Tests:** valid/invalid JWTs, cross-patient access, path traversal, spoofed extensions/MIMEs, oversized/malformed files, duplicates, lifecycle transitions, pagination, safe errors, audit metadata, and unauthorized deletion.
- **Gate:** all authorization and upload security tests pass; no real patient fixture is used.

### Stage 3 — Extraction, OCR, and chunking
- [ ] Implement PyMuPDF native PDF extraction with page/coordinate provenance and scanned-page detection.
- [ ] Add Tesseract OCR fallback, image extraction, TXT and practical DOCX parsing, extraction-quality flags, and no confident treatment of uncertain OCR.
- [ ] Preserve headings/tables where supported, decimals, signs, units, dates, and lab relationships; do not execute active content.
- [ ] Add configurable page/section/table-aware chunking and complete provenance/content hashes.
- [ ] Offload blocking extraction/OCR work from the async event loop.
- **Tests:** native and scanned PDF fixtures, PNG/JPEG, TXT/DOCX, OCR fallback and uncertainty, malformed/decompression-bomb inputs, tables, chunk boundaries, provenance, and content hashing.
- **Gate:** run real Tesseract against a synthetic scanned fixture when installed; otherwise explicitly report OCR acceptance as blocked.

### Stage 4 — Local embeddings and pgvector storage
- [ ] Implement a cached local Sentence Transformer service with configurable model directory/device, batching, clear load failures, and recorded model/revision/dimension (`sentence-transformers/all-MiniLM-L6-v2`, 384 dims).
- [ ] Implement PostgreSQL chunk embeddings table with pgvector column (`Vector(384)`) and HNSW index.
- [ ] Add Alembic migration for pgvector chunk embeddings schema and HNSW indexes.
- [ ] Implement patient-specific pgvector repositories with server-controlled add/query/delete/re-index operations.
- [ ] Store model compatibility metadata (model name, revision, dimension) and reject mixing incompatible vectors in the same table/index.
- [ ] Connect processed documents to indexing and ensure document deletion transactionally removes chunks and vectors.
- **Tests:** embedding interface and real model dimension (384), persistent reload, patient isolation in vector search, filtered queries, deletion, re-indexing, model mismatch rejection, and model-load failure.
- **Gate:** demonstrate actual local `all-MiniLM-L6-v2` embeddings and pgvector persistence using temporary synthetic data; mocks alone are insufficient for this gate.

### Stage 5 — Retrieval and deterministic evidence
- [ ] Implement local query embedding, authorized top-k retrieval via pgvector cosine distance, relevance thresholds, server-built date/document SQL filters, overlap deduplication, and typed no-evidence results.
- [ ] Add strict Pydantic v2 schemas for documents, chunks, retrieval, evidence, validation, and API contracts.
- [ ] Implement deterministic common-lab/date/number/unit/reference-range parsing without inferring missing values or units.
- [ ] Build normalized evidence packages preserving explicit/uncertain/missing/conflicting states, source provenance, content quality, and stable evidence IDs.
- [ ] Treat retrieved text solely as untrusted data and enforce as-of/future-data boundaries.
- **Tests:** relevance/no-result behavior, filters/deduplication, cross-patient attempts, future-note exclusion, exact decimals/signs/units/dates, missing and conflicting labs, provenance resolution, and malformed schema rejection.
- **Gate:** synthetic retrieval-to-evidence integration passes with real local embeddings/pgvector and no external LLM.

### Stage 6 — Presidio privacy pipeline and mandatory gate
- [ ] Configure Microsoft Presidio plus custom recognizers for names, contacts, addresses, government/medical/internal IDs, DOB, and identifying facilities.
- [ ] Implement question-specific field allow-listing, redaction/tokenization, safe temporal generalization, irrelevant-content exclusion, and quasi-identifier review flags.
- [ ] Implement `approved`, `rejected`, and `needs_review`; verify provenance/validation and fail closed on every service/error path.
- [ ] Bind approvals to exact serialized payload digests and individual destinations; store minimal privacy audit records in PostgreSQL without raw values.
- [ ] Add local-only mode and a dispatch guard that makes unapproved external transmission impossible through the application interface.
- **Tests:** varied/malformed identifiers, PII in text and metadata, internal IDs, date handling, rare combinations, prompt-injection text, Presidio unavailable, gate exceptions, payload mutation, wrong destination, fallback without re-approval, secret-free logs, and zero spy-provider calls in every blocked/local-only case.
- **Gate:** privacy/security tests pass before any live provider adapter is connected.

### Stage 7 — Providers, prompts, and routing
- [ ] Define normalized LLM request/response and prepared-payload interfaces plus versioned explanation/reasoning prompts.
- [ ] Verify current official API documentation, then implement isolated NVIDIA, Google, and configurable Llama-compatible adapters without inventing endpoints or model IDs.
- [ ] Add endpoint/model/auth configuration, TLS requirements for remote calls, timeouts, bounded transient retries, rate limits, safe errors, and response normalization.
- [ ] Configure three reasoner slots and one separate adjudicator slot; fallback remains disabled by default and always requires fresh approval.
- [ ] Add `/api/v1/providers` with enabled/configuration status only, never secrets.
- **Tests:** mocked request/response normalization, malformed responses, timeout/retry bounds, authentication failures, routing, disabled slots, destination approval, payload digest enforcement, fallback re-approval, and metadata/log redaction.
- **Gate:** all tests run without API keys. Live behavior is not claimed unless a separately approved synthetic-data smoke test succeeds.

### Stage 8 — LangGraph orchestration and deterministic verification
- [ ] Implement typed graph state and the required nodes: context loading, safety precheck, independent reasoning, normalization, deterministic validation, critique, bounded revision, safety review, candidate gate, final adjudication, final validation, and repair/abstain.
- [ ] Add explicit conditional edges for privacy blocked, urgent safety, insufficient evidence, debate bounds, unresolved disagreement, adjudication, one repair, and abstention.
- [ ] Implement independent async A/B/C first-pass calls with identical evidence and no peer visibility; cap rounds, total calls, tokens, and timeouts.
- [ ] Implement standalone schema, evidence, numerical/unit, temporal, consistency, safety, and provenance verifiers.
- [ ] Preserve supported minority objections and send only validated claims/disagreements to the separate adjudicator; never use majority vote as truth.
- [ ] Return structured verified facts/hypotheses/conflicts/missing information/uncertainty/review notices, never hidden chain-of-thought.
- **Tests:** each verifier independently; initial independence; urgent bypass; unknown/unsupported IDs; wrong decimal/value/date/unit; source conflicts; prompt injection; unsafe diagnosis/treatment text; unresolved disagreement; adjudicator input restrictions; malformed JSON repaired once; second failure abstains; and provider-call/round limits.
- **Gate:** execute the compiled LangGraph against mocked providers for both success and every failure route, and verify final answers cannot be marked verified before all checks pass.

### Stage 9 — Complete Part 2 API and vertical slice
- [ ] Add authorized `/retrieve` and `/query` endpoints plus response statuses, citations, verification summaries, model metadata, and clinician-review flags.
- [ ] Connect upload through final orchestration using dependency injection; keep business logic outside route handlers.
- [ ] Add minimal user/patient-scoped follow-up context stored in PostgreSQL with retention/deletion controls; every patient-specific follow-up re-retrieves evidence.
- [ ] Complete OpenAPI documentation, safe error models, pagination, and configuration/setup instructions.
- **Tests:** full synthetic upload → extraction/OCR → chunks → local embeddings → pgvector → retrieval → evidence → privacy → mocked panel/debate → adjudication → cited response; retrieval-only/local-only modes; insufficient evidence; provider outage; conversation isolation; and unauthorized source access.
- **Gate:** the independent document-only vertical slice passes with no prediction payload and no SLM/Part 1 imports.

### Stage 10 — Future Part 1 contract only
- [ ] Add versioned optional prediction-context Pydantic schemas and a context-builder interface with no prediction implementation.
- [ ] Store prediction metadata and outputs in PostgreSQL with foreign keys to patient records.
- [ ] Preserve model/version, generation time, values, units, horizon, threshold, uncertainty, and source observation references as immutable context.
- [ ] Add deterministic prediction verification to the existing graph without making the payload mandatory.
- **Tests:** valid/invalid contract versions, missing optional payload, source references, as-of boundaries, and attempted alteration of every immutable prediction field.
- **Gate:** document-only tests still pass unchanged and prediction-contract tests pass without any temporal model code.

### Stage 11 — Final verification and reporting
- [ ] Update README, `.env.example`, Mermaid architecture, API/schema/configuration documentation, re-indexing/deletion/retention procedures, and limitations.
- [ ] Review dependencies, storage permissions/encryption-at-rest deployment strategy, secrets, logs, error responses, and privacy call paths.
- [ ] Produce a final report separating implemented/tested behavior, mocked versus real integrations, skipped/blocked checks, known limitations, and deferred work.
- **Final checks:** run the full test suite, ruff, format check, mypy, and synthetic end-to-end/security/contract suites; inspect that no secrets or patient data were added.
- **Completion rule:** report exact commands, pass/fail/skip counts, and unresolved failures. Do not claim OCR, live providers, clinical validity, anonymity, regulatory compliance, or any unexecuted acceptance criterion as complete.

## 5. Stage-result reporting format

After every stage, provide:
1. Stage name and implementation status: passed, failed, or blocked.
2. Files/components added or changed.
3. Exact test commands and environment prerequisites used.
4. Actual passed/failed/skipped counts and relevant failure output.
5. Behaviors demonstrated by tests, distinguishing real local dependencies from mocks.
6. Acceptance items still incomplete or deferred.
7. Full-suite regression result before requesting/proceeding to the next stage.

A generated file or passing import is not evidence of completion. A stage is complete only when its required tests actually execute successfully and its report identifies any remaining limitations.
