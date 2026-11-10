# VitalWatch Part 2

Privacy-first document RAG and verified-explanation backend. This repository is being implemented incrementally from `docs_pdf/VitalWatch_master_prompt_updated.md`.

## Current implementation status

**Stages 2–6 — secure document lifecycle, local extraction, pgvector indexing, retrieval, and the mandatory privacy gate** are implemented on top of the foundation:

- typed environment configuration with secure defaults;
- FastAPI application factory and private runtime-directory initialization;
- request ID propagation, safe public errors, and redacted logging;
- a minimal structured audit interface with a logging sink;
- liveness and readiness endpoints;
- Bearer JWT authentication plus server-side PostgreSQL patient grants;
- private original storage with generated filenames, checksums, duplicate detection and lifecycle metadata;
- authorized document upload/list/status/reprocess/delete API routes.
- local PDF, image, TXT, and DOCX extraction; scanned pages use installed Tesseract OCR;
- page-provenance chunks and private structured extraction output; documents finish as `processed` or `failed`;
- a cached local `sentence-transformers/all-MiniLM-L6-v2` service (384 dimensions) that records model, revision, and dimension;
- patient-scoped chunk vectors in PostgreSQL/pgvector with a cosine HNSW index and model-identity compatibility checks;
- processed documents are embedded and indexed locally; document deletion removes chunks and vectors transactionally (explicit delete plus `ON DELETE CASCADE`);
- patient-scoped retrieval with an explicit no-evidence result and deterministic, provenance-checked evidence packages;
- Presidio-based minimization, question-specific field allow-listing, redaction/tokenization, temporal generalization, and review flags;
- single-use, destination-bound approvals with a PII-free PostgreSQL audit table and a fail-closed local-only dispatch guard.

HTTP routes for retrieval and privacy decisions, external provider adapters, LangGraph, and prediction contracts are not implemented yet. External dispatch is disabled by default; no SLM or Part 1 temporal prediction code is used.

## Requirements

- Python 3.11 or newer
- PowerShell on Windows for the commands below

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"
python -m spacy download en_core_web_sm
```

Copy the generated value into `APP_SECRET_KEY` in `.env`. The application rejects the example placeholder, secrets shorter than 32 characters, production debug mode, and attempts to disable the mandatory privacy gate.

`API_DOCS_ENABLED=true` in the example enables Swagger UI during local development. Its secure default is `false` when not explicitly configured.

## Run

```powershell
python -m uvicorn app.main:app --reload
```

With documentation enabled, Swagger UI is available at `http://127.0.0.1:8000/docs`.

Available endpoints:

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/api/v1/health` | Returns liveness and the request ID. |
| GET | `/api/v1/ready` | Checks validated configuration and private document storage directory; returns HTTP 503 if directory is unavailable. |
| POST | `/api/v1/patients/{patient_id}/documents` | Upload an authorized PDF, TXT, PNG/JPEG, or DOCX document. |
| GET | `/api/v1/patients/{patient_id}/documents` | List authorized document metadata with pagination. |
| GET/DELETE | `/api/v1/patients/{patient_id}/documents/{document_id}` | Read status/metadata or delete an authorized document. |
| POST | `/api/v1/patients/{patient_id}/documents/{document_id}/reprocess` | Return an authorized document to `uploaded` status. |

Readiness (`/api/v1/ready`) checks configuration, the document storage directory, a real `SELECT 1` against PostgreSQL, and the local embedding model. A failed database check returns HTTP 503. The embedding check reports `skipped` while the model is still lazy and unloaded, and `passed` once it is loaded; readiness requires that no check is `failed`, so an unloaded model does not by itself claim readiness of retrieval. OCR, Presidio availability, and external providers are not included in the readiness check.

Every response includes `X-Request-ID`. A syntactically valid incoming header is reused for correlation; an absent or invalid value is replaced with a generated UUID.

## Database and Migrations (PostgreSQL + pgvector)

The foundation uses PostgreSQL via SQLAlchemy 2.x and Alembic for migrations, with pgvector for vector storage:

```powershell
# Set DATABASE_URL in .env (e.g. postgresql+psycopg://user:password@localhost:5432/vitalwatch)
# Run database migrations:
alembic upgrade head
```

Migrations: `0001_enable_pgvector` enables the `vector` extension, `0002_grants_and_documents` adds patient grants and document metadata, `0003_chunk_embeddings` adds the `chunk_embeddings` table with a `vector(384)` column and a cosine HNSW index, and `0004_privacy_approvals` adds the PII-free approval audit table. Database connection is decoupled from application startup so that health checks and local testing run cleanly without requiring a live database.

## Local embeddings and vector index (Stage 4)

Embeddings are computed on this host with `sentence-transformers/all-MiniLM-L6-v2`; there is no embedding API, key, or remote endpoint. The first use downloads the model into the Hugging Face cache or `EMBEDDING_MODEL_DIRECTORY`; set `EMBEDDING_LOCAL_FILES_ONLY=true` afterwards to forbid any further download attempt. `EMBEDDING_DEVICE`, `EMBEDDING_BATCH_SIZE`, and `EMBEDDING_MODEL_REVISION` are configurable.

Every stored vector records `model_name`, `model_revision`, and `dimension`. The index accepts only 384-dimensional vectors, and a second model identity is rejected with `embedding_model_mismatch` so different embedding models are never mixed in the same table. The recorded revision is the resolved local snapshot commit when available; pin `EMBEDDING_MODEL_REVISION` to a commit to keep that identity stable. Changing the model or revision requires rebuilding the index rather than mixing identities.

Vector reads and writes always require a server-supplied `patient_id`; a route or caller cannot widen that scope. Each document's vectors are replaced within one transaction, so re-indexing never leaves a partial document, and deleting a document removes its vectors in the same transaction as its metadata.

To deliberately rebuild the whole index after changing the model or revision, run the local operator command in a maintenance window:

```powershell
python -m app.cli rebuild-index --yes
```

It clears every stored chunk vector first, then re-embeds each `processed` document from its private extraction output and prints a JSON report with counts and any failed document IDs. The command exits non-zero when any document fails, so a partial rebuild is visible. `--yes` is mandatory because the operation is destructive; nothing is exposed over HTTP.

## Retrieval and deterministic evidence (Stage 5)

`RetrievalService` embeds the query locally, searches only the authorized patient's chunks with pgvector cosine distance, forwards server-built document/date filters to SQL, applies a configurable relevance threshold (`RETRIEVAL_MIN_SCORE`, default `0.25`), and drops overlapping chunks from the same document and page. It returns a typed result: either `retrieved` with ranked items, or `no_evidence` with a reason (`no_matching_documents` or `below_relevance_threshold`). Retrieval never widens patient scope, and no route exposes it yet; Stage 9 adds the HTTP API.

`as_of` filters use `documents.created_at` as the server-controlled availability boundary. This is conservative: a note containing facts from an earlier date but ingested later is hidden before its ingestion time, and observations whose explicit in-text date is after `as_of` are excluded from the evidence package and listed in `excluded_after_as_of`.

`EvidenceCompiler` parses a small allow-listed set of common labs from retrieved chunks without inferring anything:

- numbers are kept as exact lexemes (`4.10` and `-0.50` do not become `4.1`/`-0.5`) and units are preserved, never converted;
- a line without a unit is marked `uncertain` and flagged `missing_unit`;
- OCR-derived text of uncertain quality is marked `uncertain`;
- reference ranges are only compared when the range carries the same explicit unit; otherwise the comparison is left unknown with a flag;
- two different values or units for the same lab are reported as a conflict, not resolved;
- pending labs become `missing_information`;
- ISO (`2026-01-05`) and `2026/01/05` dates are parsed; ambiguous or invalid dates are flagged and never guessed.

Each observation carries a stable evidence ID derived from its source chunk, line, lab, and last value, and every package resolves those IDs back to `document_id`, `chunk_id`, page, content hash, extraction method, quality, and score. Retrieved text is marked `untrusted_retrieved_text`; it is data for later validation, never instructions.

Configuration: `RETRIEVAL_TOP_K` (default `5`), `RETRIEVAL_MIN_SCORE` (default `0.25`), and `RETRIEVAL_MAX_QUERY_CHARS` (default `2000`).

## Privacy gate and approvals (Stage 6)

`PrivacyGate.prepare` is the mandatory boundary before any external model call. It verifies evidence provenance, classifies the question (`lab_value`, `trend`, `medication`, `diagnosis`, or `general`), and builds a payload from a question-specific field allow-list. Irrelevant labs are excluded, requested labs that are absent become `missing_information`, and no patient ID, document/chunk ID, filename, checksum, storage key, score, or raw retrieved text is included.

The question and evidence are processed locally:

- Presidio plus custom recognizers detect names, contact details, addresses, government and medical identifiers, internal IDs, dates of birth, facilities, and ages;
- detected values are replaced with stable per-request tokens (for example `[PERSON_1]`, `[EMAIL_1]`, `[MEDICAL_ID_1]`); the token vault exists only in memory;
- dates of birth become age bands (`[AGE_BAND_70_79]`, `[AGE_BAND_90_plus]`), observation dates are reduced to `YYYY-MM`, and malformed or ambiguous dates fail closed;
- prompt-injection text and risky identifier combinations (person plus context, facility plus date, age 90+, residual PII) are flagged for review;
- the payload digest is a canonical JSON SHA-256, and the destination digest binds provider, endpoint, model, and purpose.

Every decision is one of `approved`, `rejected`, or `needs_review`. Only an approved decision exposes a payload. Rejections cover local-only mode, empty/oversized questions, provenance failures, Presidio unavailability or failure, unexpected gate errors, and audit-store failure. An approved payload expires after `PRIVACY_APPROVAL_TTL_SECONDS` (default 300) and can be dispatched once; `ApprovedDispatcher` rechecks status, payload digest, destination digest, and expiry, then atomically consumes the approval before calling the transport. Mutation, a wrong destination, reuse, or a fallback route without a fresh approval is blocked. The PostgreSQL `privacy_approvals` row stores hashes, counts, status, and reason/review codes only, never the question, payload, raw patient/actor ID, or PII values.

Presidio uses the local spaCy model from `PRIVACY_NLP_MODEL` (default `en_core_web_sm`). If Presidio or that model is missing, the gate rejects with `presidio_unavailable` instead of sending data. Keep `PRIVACY_LOCAL_ONLY=true`; setting it to `false` also requires `EXTERNAL_LLM_ENABLED=true`, and no real provider adapter is implemented in this stage.

## Tests and static checks

```powershell
python -m pytest tests/unit
python -m pytest -m "not integration"
python -m pytest -m integration
python -m pytest
python -m ruff check app tests alembic
python -m mypy app
```

Tests use only synthetic values and temporary directories. Paid provider keys and live external calls are not required.

Unit and security tests load real Presidio and `en_core_web_sm`; install the model with `python -m spacy download en_core_web_sm`. Integration tests start a temporary embedded PostgreSQL 16 with pgvector through the `pgserver` development dependency, run the real Alembic migrations against it (including `privacy_approvals`), and load the real `all-MiniLM-L6-v2` model. The first integration run needs network access to download the models; later runs work from the local cache. If `pgserver`, Presidio, or a model is unavailable the tests report an error rather than passing, so a skipped privacy or Stage 4 check is not evidence of success.

## Configuration

Settings are loaded from environment variables and `.env` using Pydantic v2 settings. See `.env.example` for all configuration variables, including the local embedding settings. Runtime documents, `.env`, virtual environments, logs, caches, and build artifacts are excluded by `.gitignore`.

Logging redaction covers common secret, bearer-token, email, and identifier patterns, including rendered exception tracebacks. It is defense in depth, not proof of anonymization: application code must still avoid logging PHI, raw prompts, completions, document contents, or credentials.

The Stage 1 audit interface accepts only structured metadata and emits allow-listed fields to the log. Durable patient-scoped audit persistence belongs to a later stage.

## Safety and limitations

### Windows event loop

`psycopg` cannot run async work on the default Windows `ProactorEventLoop`; the default policy raises `InterfaceError`. The test suite selects `WindowsSelectorEventLoopPolicy` automatically. Start the API locally with the selector loop:

```powershell
python -c "import asyncio; asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()); import uvicorn; uvicorn.run('app.main:app', reload=True)"
```

### Index limitations

The HNSW index serves cosine ranking, while patient, document, and date filters are applied in SQL. pgvector versions before 0.8 scan and filter after the graph walk, so recall under very selective filters can fall below exact search. Stage 5 measures this on a small synthetic index and the integration checks compare filtered rankings with exact cosine scores, but this is not a production-scale recall benchmark: measure recall and tune `hnsw.ef_search` on realistic corpus sizes before relying on filtered retrieval.

VitalWatch is a research prototype and does not provide autonomous diagnosis or treatment advice. No clinical validity, anonymity, regulatory compliance, or production readiness is claimed. Use synthetic records only until the required privacy, authorization, and security controls are implemented and tested.
