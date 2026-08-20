# Company Researcher

Company Researcher is an evidence-driven system for controlled, reproducible
investigations over UK company filings.

This is **not a Companies House chatbot**. Companies House provides a public,
reproducible stand-in for the private APIs, databases, documents, and audit
histories commonly found in enterprise AI projects. The long-term value of the
project is its retrieval, evidence, evaluation, and human-review architecture;
the public data source should remain replaceable.

The project is currently at its Day 2 foundation. It can:

- run a local PostgreSQL 17 database with the `pgvector` extension available;
- manage the database schema with SQLAlchemy and Alembic;
- run a minimal FastAPI application;
- retrieve a real company profile and complete filing history from Companies
  House;
- inspect that source data through a small command-line interface; and
- persist a company's profile and filing history in PostgreSQL, with source
  provenance and retrieval timestamps, idempotently.

Advanced RAG, embeddings, LangGraph, temporal analysis, evaluation, and
human-in-the-loop workflows are deliberately deferred until later phases.

## Prerequisites

- Python 3.13
- [`uv`](https://docs.astral.sh/uv/)
- Docker with Docker Compose (Docker Desktop or OrbStack both work)
- Tesseract OCR 5 with English language data
- A Companies House REST API key

Install Tesseract on macOS with Homebrew:

```bash
brew install tesseract
```

The application Docker image installs Tesseract automatically.

Create a REST API key through the Companies House developer portal. Do not
commit the key to Git.

## Initial setup

Clone the repository and enter its directory, then install the exact locked
dependencies:

```bash
uv sync
```

`uv` reads `pyproject.toml`, creates `.venv` if necessary, and installs the
versions recorded in `uv.lock`. You do not need to activate the virtual
environment when using `uv run`.

Create your local environment file from the committed template:

```bash
cp .env.example .env
```

Open `.env` and replace this placeholder with your real REST API key:

```dotenv
COMPANIES_HOUSE_API_KEY=replace-with-your-api-key
```

`.env` is ignored by Git. `.env.example` documents the required settings but
contains no secrets.

## Start PostgreSQL

Start the database container in the background:

```bash
docker compose up -d db
```

Check that it is running and healthy:

```bash
docker compose ps
```

The Compose configuration uses PostgreSQL 17 with pgvector included. Local
development defaults expose it on `localhost:5432` with the database, username,
and password `company_researcher`. These are development credentials only.

Apply every database migration:

```bash
uv run alembic upgrade head
```

The initial migration enables PostgreSQL's `vector` extension so the database
is ready for a later retrieval phase. The application does not use embeddings
or vector search yet.

To stop PostgreSQL without deleting its data:

```bash
docker compose down
```

The named Docker volume preserves the database between restarts.

## Run the complete stack in Docker

Build the application image and start PostgreSQL, the migration task, and the
FastAPI service:

```bash
docker compose up --build -d
```

Compose starts the services in dependency order:

1. `db` starts and passes its PostgreSQL health check.
2. `migrate` applies `alembic upgrade head` and exits successfully.
3. `api` starts and passes its HTTP health check.

Inspect the state of all three services:

```bash
docker compose ps -a
```

The `db` and `api` services should be healthy, while `migrate` should show that
it exited with status `0`. The API is then available at
<http://127.0.0.1:8000> by default.

Run the company inspection command inside the application container:

```bash
docker compose exec api company-researcher inspect 00000006
```

`DATABASE_URL` uses `localhost` when Python runs directly on the host. Compose
overrides it inside the application containers so they connect to PostgreSQL at
the Docker service hostname `db`.

Stop the complete stack without deleting PostgreSQL data:

```bash
docker compose down
```

## Run the API

Start FastAPI locally with Uvicorn:

```bash
uv run uvicorn company_researcher.main:app --reload
```

In another terminal, request the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

This endpoint currently confirms that the application process can serve a
request. It is not yet a database readiness check.

FastAPI's interactive API documentation is available at
<http://127.0.0.1:8000/docs> while the server is running.

In VS Code, select **Run and Debug → Company Researcher API** to launch the same
application under the debugger. The launch configuration loads `.env`.

## Inspect a real company

Fetch a company profile and its complete filing history using its Companies
House company number:

```bash
uv run company-researcher inspect 00000006
```

The command prints formatted JSON with two top-level fields:

- `profile`: the current Companies House company profile;
- `filing_history`: all filing-history pages retrieved from the API.

Company numbers are identifiers, not integers. Preserve leading zeroes when
entering them.

In VS Code, select **Run and Debug → CLI: Inspect Company**. VS Code will prompt
for the company number before starting the debugger.

The client handles authentication failures, missing companies, rate limiting,
connection failures, response errors, pagination, and response validation. API
responses remain external, untrusted input.

## Persist a real company

Fetch a company profile and filing history and store them in PostgreSQL:

```bash
uv run company-researcher ingest 00000006
```

Each row records `source`, `retrieved_at`, and the raw source payload
alongside its structured fields, so ingested data stays auditable back to
where and when it came from. Re-running `ingest` for the same company number
updates the existing rows in place rather than creating duplicates.

In VS Code, select **Run and Debug → CLI: Ingest Company**.

## Persist a filing document

After ingesting a company, download and persist one filing document by its
Companies House transaction ID:

```bash
uv run company-researcher ingest-document 08130873 TRANSACTION_ID
```

The command requires an existing filing row with a Companies House document
reference. It downloads the PDF, stores the original bytes under the configured
`ARTIFACT_ROOT`, and records the filing relationship, source metadata, checksum,
storage key, and retrieval timestamps in PostgreSQL. Repeating the command for
unchanged content refreshes its retrieval metadata without creating a duplicate
document version.

Documents default to content-addressed paths beneath `data/artifacts`. The
directory is ignored by Git and can be changed with `ARTIFACT_ROOT`.

## Extract a filing document

Run page-aware OCR for a downloaded filing document using its PostgreSQL ID:

```bash
uv run company-researcher extract-document FILING_DOCUMENT_ID
```

The command verifies the stored PDF against its recorded SHA-256 checksum,
extracts every page with Tesseract, and persists the page text together with
the exact OCR and renderer configuration. Repeating the command with the same
document and configuration reuses the successful extraction.

## Quality checks

Run the test suite:

```bash
uv run pytest
```

Check linting and formatting:

```bash
uv run ruff check .
uv run ruff format --check .
```

Run strict static type checking:

```bash
uv run mypy
```

To apply Ruff's formatter after editing Python files:

```bash
uv run ruff format .
```

## Project structure

```text
.
├── compose.yaml                       # Local PostgreSQL service
├── migrations/                        # Alembic schema revisions
├── src/company_researcher/
│   ├── api/                            # FastAPI routes
│   ├── companies_house/                # Replaceable source integration
│   ├── db/                             # SQLAlchemy engine, sessions, and models
│   ├── cli.py                          # Company inspection and ingestion commands
│   ├── config.py                       # Environment-backed settings
│   ├── ingestion.py                    # Idempotent persistence of source data
│   └── main.py                         # FastAPI application factory
├── tests/                              # Focused unit and API tests
├── .env.example                        # Safe configuration template
├── alembic.ini                         # Alembic configuration
├── pyproject.toml                      # Package, tools, and dependencies
└── uv.lock                             # Reproducible dependency lock
```

The full product direction and intended later phases are described in
[`docs/project-brief.md`](docs/project-brief.md).
