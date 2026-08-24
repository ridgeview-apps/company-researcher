# Company Researcher

Company Researcher is an evidence-driven system for controlled, reproducible
investigations over UK company filings.

This is **not a Companies House chatbot**. Companies House provides a public,
reproducible stand-in for the private APIs, databases, documents, and audit
histories commonly found in enterprise AI projects. The long-term value of the
project is its retrieval, evidence, evaluation, and human-review architecture;
the public data source should remain replaceable.

The project has completed its ingestion and document-processing foundation. It
can:

- run a local PostgreSQL 17 database with the `pgvector` extension available;
- manage the database schema with SQLAlchemy and Alembic;
- run a minimal FastAPI application;
- retrieve a real company profile and complete filing history from Companies
  House;
- inspect that source data through a small command-line interface;
- persist a company's profile and filing history in PostgreSQL, with source
  provenance and retrieval timestamps, idempotently;
- download filing PDFs and preserve their metadata and immutable original bytes
  in content-addressed storage;
- verify stored artifacts against their recorded SHA-256 checksums;
- extract image-only PDFs page by page with local Tesseract OCR;
- persist page text and exact extraction provenance, reusing an existing
  successful extraction when its document and configuration are unchanged; and
- measure a deterministic PostgreSQL lexical-search baseline against a small,
  manually labelled retrieval evaluation corpus using Recall@K and MRR (see
  [Measure the lexical-search retrieval baseline](#measure-the-lexical-search-retrieval-baseline)
  below for the measured results).

LLM generation, embeddings, hybrid retrieval, LangGraph, temporal analysis, and
human-in-the-loop workflows remain deliberately deferred until their evidence
and retrieval foundations exist.

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

## Measure the lexical-search retrieval baseline

Score PostgreSQL full-text search against a labelled evaluation dataset:

```bash
uv run company-researcher evaluate-retrieval
```

Without an argument, the command evaluates
[`evaluation/gymshark_retrieval_questions.json`](evaluation/gymshark_retrieval_questions.json):
a small, manually labelled set of retrieval questions over Gymshark Ltd's
persisted accounts filings, including an original/amended filing pair. Each
question's text is matched against `document_pages` using a
PostgreSQL-native, deterministic ranking (`ts_rank` over an OR-combined,
stemmed `tsquery`, accelerated by a GIN expression index) with no embeddings,
vector search, or LLM involved. The command reports Recall@K and Mean
Reciprocal Rank per question and averaged across the dataset.

By default the command issues each question's hand-picked `query` field. Pass
`--query-source derived` to instead ignore `query` and derive one from `text`
with `derive_query()` — a fixed, deterministic stopword-removal rule defined
in [`query_construction.py`](src/company_researcher/query_construction.py)
that depends only on the question text, never on which pages are known to be
relevant, so it cannot be tuned to a specific answer. Pass `--query-source
derived-idf` for a second deterministic strategy,
[`derive_discriminative_query()`](src/company_researcher/discriminative_query.py),
which further ranks `derive_query()`'s content words by document frequency
across every persisted document page and keeps only the rarest few:

```bash
uv run company-researcher evaluate-retrieval --query-source derived
uv run company-researcher evaluate-retrieval --query-source derived-idf
```

### Measured results

Four lexical query strategies have been measured against the same 6-question
Gymshark evaluation set, using the same `ts_rank`/GIN-indexed search underneath
every time — only the text sent as the query changed.

**Full question sentence as the query** (each question's `text` field — the
first thing tried):

| Question | Recall@5 | Recall@10 | Reciprocal rank |
| --- | --- | --- | --- |
| q1-fy2025-turnover | 0.00 | 0.00 | 0.04 |
| q2-turnover-trend-fy2021-fy2025 | 0.00 | 0.00 | 0.00 |
| q3-fy2022-amendment-comparison | 0.00 | 0.00 | 0.00 |
| q4-directors-fy2021-fy2025 | 0.00 | 0.00 | 0.00 |
| q5-dividends-fy2022-vs-fy2025 | 0.00 | 0.00 | 0.06 |
| q6-going-concern-fy2023 | 0.00 | 0.00 | 0.08 |
| **Mean** | **0.000** | **0.000** | **0.030** |

Matching a whole natural-language question against single pages by lexical
term overlap is a weak signal, especially for questions whose relevant pages
are spread across several documents (Q2, Q4) or that ask about the *absence*
of a difference (Q3).

**Short keyword query as the query** (each question's `query` field — a few
words chosen to reflect what a person would actually type into a search box,
tuned by measuring against the persisted corpus rather than guessed; this is
the query the CLI command actually issues; see the caveat below the results):

| Question | Recall@5 | Recall@10 | Reciprocal rank |
| --- | --- | --- | --- |
| q1-fy2025-turnover | 1.00 | 1.00 | 0.50 |
| q2-turnover-trend-fy2021-fy2025 | 0.75 | 1.00 | 0.50 |
| q3-fy2022-amendment-comparison | 0.00 | 0.50 | 0.14 |
| q4-directors-fy2021-fy2025 | 1.00 | 1.00 | 1.00 |
| q5-dividends-fy2022-vs-fy2025 | 0.50 | 1.00 | 0.33 |
| q6-going-concern-fy2023 | 0.50 | 0.50 | 0.20 |
| **Mean** | **0.625** | **0.833** | **0.446** |

Query wording, not the ranking mechanism, was the dominant cause of the first
result: the same OR-combined `ts_rank` search goes from finding nothing to
finding most relevant pages once it is given a short, targeted query instead
of a full sentence. Q3 remains the hardest case even with a good query — its
relevant pages span two different vocabularies (profit-and-loss language and
balance-sheet language, for both the original and amended documents), which a
single short query struggles to boost simultaneously. That is a genuine,
still-open limitation of pure lexical search on this corpus, not a bug.

**Caveat on the short-query result above**: each `query` string was
hand-tuned by trial and error directly against these same 6 questions'
known-relevant pages, not chosen blind or validated against held-out
questions. So that result shows that a human who already knows the correct
answer can hand-craft a query that finds it — it does not show that this
approach generalizes to a genuinely new, unseen question.

**Derived query, from `derive_query(text)`** (a fixed stopword-removal rule,
applied identically to all six questions with no knowledge of which pages are
relevant — see [`query_construction.py`](src/company_researcher/query_construction.py)):

| Question | Recall@5 | Recall@10 | Reciprocal rank |
| --- | --- | --- | --- |
| q1-fy2025-turnover | 0.00 | 0.00 | 0.04 |
| q2-turnover-trend-fy2021-fy2025 | 0.00 | 0.00 | 0.00 |
| q3-fy2022-amendment-comparison | 0.00 | 0.00 | 0.00 |
| q4-directors-fy2021-fy2025 | 0.00 | 0.00 | 0.00 |
| q5-dividends-fy2022-vs-fy2025 | 0.00 | 0.00 | 0.06 |
| q6-going-concern-fy2023 | 0.00 | 0.00 | 0.08 |
| **Mean** | **0.000** | **0.000** | **0.030** |

This result is a genuine negative finding, not a bug: it scores no better
than matching the full sentence. Stopword removal alone still leaves 7–13
content words per question (for example Q3 derives to `amended FY2022
accounts AAMD change reported turnover profit balance sheet figures compared
original FY2022 accounts`), including common, non-discriminative terms —
"accounts", "compared", "figures", "position" — that recur across most pages
of an accounts filing. Under OR-combined `ts_rank`, this dilutes ranking the
same way a full sentence does. What made the hand-tuned queries work was not
mainly their brevity but that a human who already knew the answer selected
*rare, discriminative* terms for that specific page (several lifted directly
from the answer itself, e.g. "revolving credit facility", "statement of
financial position") — which is exactly the tuning bias the caveat above
describes, now visible from the other direction: removing that bias while
keeping only a generic, corpus-blind heuristic reproduces the original
sentence-matching failure. Query *length/term-discriminativeness*, not
sentence-vs-keyword phrasing, is the operative variable, and a naïve
stopword-only rule does not control for it.

**Discriminative query, from `derive_discriminative_query(text)`** (ranks
`derive_query()`'s content words by document frequency across all persisted
document pages — computed corpus-wide, never from a question's known-relevant
pages — and keeps the 4 rarest):

| Question | Recall@5 | Recall@10 | Reciprocal rank |
| --- | --- | --- | --- |
| q1-fy2025-turnover | 0.00 | 1.00 | 0.14 |
| q2-turnover-trend-fy2021-fy2025 | 0.00 | 0.00 | 0.00 |
| q3-fy2022-amendment-comparison | 0.00 | 0.00 | 0.06 |
| q4-directors-fy2021-fy2025 | 0.00 | 0.00 | 0.02 |
| q5-dividends-fy2022-vs-fy2025 | 0.00 | 0.00 | 0.06 |
| q6-going-concern-fy2023 | 0.50 | 0.50 | 0.50 |
| **Mean** | **0.083** | **0.250** | **0.130** |

A real, if partial, improvement over stopword-only (Mean Recall@10 0.0 →
0.25, MRR 0.03 → 0.13) — ranking by corpus rarity recovers some signal a
generic stopword rule can't. But it falls well short of the hand-tuned
result, and inspecting the actual queries shows two concrete, corpus-specific
reasons why:

1. **Boilerplate repetition defeats page-level document frequency.** Q4
   derives to `secretary annual set according`, dropping both `directors`
   (document frequency 110 pages) and `Gymshark` (234 pages) — the two most
   relevant terms for "who were the directors" — because accounts filings
   repeat director-related and company-identifying language across dozens of
   pages (directors' report, statement of directors' responsibilities,
   company information page, each repeated per filing year). A term being
   common across *many pages* doesn't mean it's a poor filter for *the one
   right page*; page-level IDF conflates the two. This is a single-company
   corpus of 5 filings, so `Gymshark` in particular provides no discriminating
   power here — it would likely behave very differently across a
   multi-company corpus.
2. **Literal-token mismatch.** `derive_query()` preserves `FY2021`/`FY2025`
   verbatim from the question text, but the filings themselves say "2021"/
   "2025" without the `FY` prefix, so those terms have document frequency 0
   and are dropped by both derived strategies — not a ranking failure, a
   vocabulary mismatch between how the question was phrased and how the
   source text is written.

Q6 (`concern going identify position`, Recall@5 = 0.50) is this strategy's
best result and the counter-example: "going concern" is genuinely rare in
this corpus (it's a specific accounting disclosure, not boilerplate), so
IDF-based selection worked exactly as intended there.

Together, the four results show that closing the gap to the hand-tuned
baseline through smarter *deterministic lexical term selection* alone has
diminishing returns on a corpus this size: page-level document frequency
can't distinguish "rare because unimportant" from "common because it's
structural boilerplate repeated within every filing." That is a materially
different, better-evidenced case for exploring semantic (embedding-based)
retrieval than "the schedule said so" — a semantic representation of a page
doesn't depend on literal token match or corpus-wide term frequency the same
way, so it isn't vulnerable to either failure mode diagnosed above. It is not
evidence that hybrid retrieval would beat lexical search outright, only that
lexical search's remaining deterministic query-construction options have
been reasonably explored on this corpus first, rather than skipped.

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
├── evaluation/                         # Labelled retrieval evaluation datasets
├── migrations/                        # Alembic schema revisions
├── src/company_researcher/
│   ├── api/                            # FastAPI routes
│   ├── companies_house/                # Replaceable source integration
│   ├── db/                             # SQLAlchemy engine, sessions, and models
│   ├── artifact_store.py               # Content-addressed source artifacts
│   ├── cli.py                          # Inspection, ingestion, extraction, and evaluation CLI
│   ├── config.py                       # Environment-backed settings
│   ├── document_ingestion.py           # Filing-document acquisition and persistence
│   ├── extraction_persistence.py       # Idempotent page-extraction persistence
│   ├── ingestion.py                    # Idempotent persistence of source data
│   ├── lexical_search.py               # PostgreSQL full-text page search
│   ├── main.py                         # FastAPI application factory
│   ├── pdf_extraction.py               # Page-aware local PDF OCR
│   └── retrieval_evaluation.py         # Recall@K / MRR scoring against labelled data
├── tests/                              # Focused unit and API tests
├── .env.example                        # Safe configuration template
├── alembic.ini                         # Alembic configuration
├── pyproject.toml                      # Package, tools, and dependencies
└── uv.lock                             # Reproducible dependency lock
```

The full product direction and intended later phases are described in
[`docs/project-brief.md`](docs/project-brief.md).
