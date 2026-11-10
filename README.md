# VitalWatch Part 2

Privacy-first document RAG and verified-explanation backend. This repository is being implemented incrementally from `docs_pdf/VitalWatch_master_prompt_updated.md`.

## Current implementation status

Only **Stage 1 — Foundation** is present:

- typed environment configuration with secure defaults;
- FastAPI application factory and private runtime-directory initialization;
- request ID propagation, safe public errors, and redacted logging;
- a minimal structured audit interface with a logging sink;
- liveness and readiness endpoints;
- Stage 1 automated tests.

Document upload, authentication, patient authorization, extraction/OCR, embeddings, pgvector retrieval, evidence compilation, Presidio, external providers, LangGraph, and prediction contracts are not implemented yet. No SLM or Part 1 temporal prediction code is used.

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
```

Copy the generated value into `APP_SECRET_KEY` in `.env`. The application rejects the example placeholder, secrets shorter than 32 characters, production debug mode, and attempts to disable the mandatory privacy gate.

`API_DOCS_ENABLED=true` in the example enables Swagger UI during local development. Its secure default is `false` when not explicitly configured.

## Run

```powershell
python -m uvicorn app.main:app --reload
```

With documentation enabled, Swagger UI is available at `http://127.0.0.1:8000/docs`.

Stage 1 endpoints:

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/api/v1/health` | Returns liveness and the request ID. |
| GET | `/api/v1/ready` | Checks validated configuration and private document storage directory; returns HTTP 503 if directory is unavailable. |

Readiness currently checks only dependencies that exist in Stage 1. It does not claim that a database, embedding model, vector store, OCR engine, privacy service, or external provider is operational.

Every response includes `X-Request-ID`. A syntactically valid incoming header is reused for correlation; an absent or invalid value is replaced with a generated UUID.

## Database and Migrations (PostgreSQL + pgvector)

The foundation uses PostgreSQL via SQLAlchemy 2.x and Alembic for migrations, with pgvector for vector storage:

```powershell
# Set DATABASE_URL in .env (e.g. postgresql+psycopg://user:password@localhost:5432/vitalwatch)
# Run database migrations:
alembic upgrade head
```

The initial migration (`0001_enable_pgvector`) enables the `vector` extension on PostgreSQL. Database connection is decoupled from application startup so that health checks and local testing run cleanly without requiring a live database.

## Tests and static checks

```powershell
python -m pytest tests/unit tests/integration/test_health.py tests/integration/test_ready.py
python -m pytest
python -m ruff check app tests alembic
python -m mypy app
```

Tests use only synthetic values and temporary directories. Paid provider keys and live external calls are not required.

## Configuration

Settings are loaded from environment variables and `.env` using Pydantic v2 settings. See `.env.example` for the Stage 1 variables. Runtime documents, `.env`, virtual environments, logs, caches, and build artifacts are excluded by `.gitignore`.

Logging redaction covers common secret, bearer-token, email, and identifier patterns, including rendered exception tracebacks. It is defense in depth, not proof of anonymization: application code must still avoid logging PHI, raw prompts, completions, document contents, or credentials.

The Stage 1 audit interface accepts only structured metadata and emits allow-listed fields to the log. Durable patient-scoped audit persistence belongs to a later stage.

## Safety and limitations

VitalWatch is a research prototype and does not provide autonomous diagnosis or treatment advice. No clinical validity, anonymity, regulatory compliance, or production readiness is claimed. Use synthetic records only until the required privacy, authorization, and security controls are implemented and tested.
