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
  below for the measured results);
- embed filing pages with an OpenAI-compatible embeddings API and measure a
  pgvector cosine-distance vector-only retrieval baseline against the same
  evaluation corpus (see
  [Measure the vector-only retrieval baseline](#measure-the-vector-only-retrieval-baseline)
  below); and
- combine the lexical and vector rankings with Reciprocal Rank Fusion and
  measure the resulting hybrid retrieval baseline against the same evaluation
  corpus (see
  [Measure the hybrid retrieval baseline](#measure-the-hybrid-retrieval-baseline)
  below);
- persist a second company, Nothing Technology Ltd, and measure the same
  lexical-search baseline against a second, independently hand-labelled
  evaluation corpus covering both its accounts and registered-charge
  filings (see
  [Measure the second-company retrieval baseline: Nothing Technology](#measure-the-second-company-retrieval-baseline-nothing-technology)
  below); and
- answer one natural-language investigation question at a time with a small
  LangGraph agent that generates its own lexical search query, retrieves
  evidence pages, and produces a structured, citation-grounded finding whose
  citations are validated against the evidence actually retrieved (see
  [Run the investigation agent](#run-the-investigation-agent) below).

Temporal analysis and human-in-the-loop workflows remain deliberately
deferred until the relevant project phase.

## Prerequisites

- Python 3.13
- [`uv`](https://docs.astral.sh/uv/)
- Docker with Docker Compose (Docker Desktop or OrbStack both work)
- Tesseract OCR 5 with English language data
- A Companies House REST API key
- An OpenAI API key (only needed for `embed-document`, `evaluate-retrieval
  --retrieval-method vector|hybrid`, and `investigate`)

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

If you intend to run `embed-document`, also replace the OpenAI placeholder:

```dotenv
OPENAI_API_KEY=replace-with-your-api-key
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
the exact OCR and renderer configuration. Repeating the command reports one
of three outcomes: "Created" (no tracking row existed yet), "Retried" (a row
existed but its previous run had failed — real OCR work happens on this
call), or "Reused" (a row already completed successfully — no new work
happens, the true no-op case).

## Embed a filing document

Embed a succeeded document extraction's pages using its PostgreSQL ID:

```bash
uv run company-researcher embed-document DOCUMENT_EXTRACTION_ID
```

The command calls the configured OpenAI-compatible embeddings API
(`OPENAI_EMBEDDING_MODEL`, default `text-embedding-3-small`) once per
extraction, in a single batched request covering every persisted page, and
stores one vector per page alongside the exact provider, model, and
dimensionality used. Repeating the command reports one of three outcomes:
"Created" (no tracking row existed yet), "Retried" (a row existed but its
previous run had failed, e.g. an exhausted API quota — real embedding work
happens on this call), or "Reused" (a row already completed successfully —
no new work happens, the true no-op case). See
[Measure the vector-only retrieval baseline](#measure-the-vector-only-retrieval-baseline)
for how these embeddings are actually searched and evaluated.

Because the stored `vector` column has a fixed width (1536, matching
`text-embedding-3-small`'s native output), switching to a differently-sized
model later would need a new migration, not just a different configuration
value; `dimensions` on `document_embeddings` records what was actually used
for provenance, but does not itself control the column's width.

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
| q6-going-concern-fy2023 | 0.50 | 0.50 | 0.33 |
| **Mean** | **0.625** | **0.833** | **0.468** |

(Q6's reciprocal rank and the mean MRR were originally measured as 0.20
and 0.446; see
[Scoping retrieval to one company](#scoping-retrieval-to-one-company)
below for why they changed — a ranking-tiebreak fix, not a change in
retrieval quality.)

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
| q3-fy2022-amendment-comparison | 0.00 | 0.00 | 0.02 |
| q4-directors-fy2021-fy2025 | 0.00 | 0.00 | 0.03 |
| q5-dividends-fy2022-vs-fy2025 | 0.00 | 0.00 | 0.06 |
| q6-going-concern-fy2023 | 0.50 | 0.50 | 0.50 |
| **Mean** | **0.083** | **0.250** | **0.125** |

(Q3 and Q4's reciprocal ranks, and the mean MRR, were originally measured
as 0.06/0.02 and 0.130. This strategy's document-frequency statistics are
computed across *all* persisted document pages by design — see
[Scoping retrieval to one company](#scoping-retrieval-to-one-company)
below for why ingesting a second company genuinely changed its input
statistics, unlike the hand-tuned queries above, which are fixed strings
and unaffected by corpus size.)

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

## Measure the vector-only retrieval baseline

Each Gymshark document extraction's pages were embedded with
`text-embedding-3-small` via `embed-document` (see
[Embed a filing document](#embed-a-filing-document)), then scored against the
same 6-question set with:

```bash
uv run company-researcher evaluate-retrieval --retrieval-method vector
```

This embeds each question's full natural-language `text` (not a short
keyword query — unlike lexical `ts_rank`, embeddings are not diluted by
extra context, so there is no reason to shorten it) and ranks pages by
cosine distance against the persisted page embeddings.

### Measured result

| Question | Recall@5 | Recall@10 | Reciprocal rank |
| --- | --- | --- | --- |
| q1-fy2025-turnover | 0.00 | 0.00 | 0.00 |
| q2-turnover-trend-fy2021-fy2025 | 0.00 | 0.00 | 0.03 |
| q3-fy2022-amendment-comparison | 0.00 | 0.00 | 0.07 |
| q4-directors-fy2021-fy2025 | 0.00 | 0.00 | 0.02 |
| q5-dividends-fy2022-vs-fy2025 | 0.00 | 0.00 | 0.00 |
| q6-going-concern-fy2023 | 0.00 | 0.50 | 0.14 |
| **Mean** | **0.000** | **0.083** | **0.044** |

This is worse than every lexical strategy measured above, including the
weakest one (matching the full sentence). As predicted, vector search is not
vulnerable to the two lexical failure modes diagnosed above — but inspecting
individual results shows it has a different, more serious failure mode on
this specific corpus: **it confuses near-identical disclosures across
different fiscal years.**

For Q1 ("turnover for the year ended 31 July 2025"), the actually-relevant
page — Gymshark's FY2025 profit and loss account, containing `Turnover 3
490,142 458,624` — ranked outside the top 50 entirely, at cosine distance
0.443. Ranked *first* instead, at distance 0.255, was a FY2023 strategic
report's KPI table headed "FOR THE YEAR ENDED 31 JULY 2023" that also
happens to state a `Turnover` figure. The KPI table's embedding is closer to
the question than the actual FY2025 P&L page's embedding, because the KPI
table is short and almost entirely about the concept "turnover," while the
full P&L page dilutes that signal across a dozen other line items (cost of
sales, distribution costs, administrative expenses, interest, ...) — and
critically, the embedding barely distinguishes *which year's* turnover table
it is looking at.

The same pattern explains Q6, the only question that scored above zero: its
two relevant pages are the FY2023 accounts' going-concern note, but the
page ranked closest to the question is the *FY2021* accounts' going-concern
note — nearly word-for-word identical boilerplate ("The directors are
required to assess the Company's ability to continue as a going concern for
a period of at least 12 months from the date of signing...") that UK
statutory accounts reuse verbatim, year after year, by regulatory
convention. The FY2023 page does rank in the top 10 (position 7), which is
why Q6 gets partial credit — but a near-duplicate from the wrong year still
ranks above it.

This is the reverse of the lexical corpus's problem. Lexical search's
literal `2025`/`2023` token match trivially disambiguates fiscal years —
that specific mechanism is exactly why the hand-tuned query `Gymshark
turnover 2025` scored Recall@5 = 1.00 on Q1. Dense embeddings capture *topic*
well (this is unmistakably "a turnover KPI table," "a going-concern note")
but are comparatively weak at the fine-grained, almost incidental detail of
*which year's instance* of a heavily templated annual disclosure it is —
exactly the kind of precise, low-semantic-weight token embeddings are known
to underweight. On a single-company, multi-year corpus built from
regulation-driven boilerplate, that is a real and severe limitation, not an
artifact of a misconfigured search.

This result is a concrete, evidenced case for **hybrid retrieval** as the
next step — not because the schedule always said so, but because lexical and
vector search have now been shown to fail in complementary, opposite ways on
this exact corpus: lexical is weak at bridging vocabulary/paraphrase gaps
(the caveat that motivated embeddings in the first place) while vector is
weak at fine-grained temporal disambiguation between near-duplicate
boilerplate (the failure just measured). Neither dominates the other here;
combining literal term matching with semantic similarity is a plausible way
to get both kinds of precision the other search alone lacks.

## Measure the hybrid retrieval baseline

[`hybrid_search.py`](src/company_researcher/hybrid_search.py) combines the
lexical and vector rankings above with Reciprocal Rank Fusion (RRF), scoring
each page `sum(1 / (k + rank))` (k=60) across whichever ranking(s) it appears
in. RRF combines by rank position rather than raw score deliberately:
`ts_rank` and cosine distance are on incomparable, oppositely-oriented
scales, so combining them by value would need an unvalidated normalization
step first. Each method's already-established query input is reused
unchanged — the lexical component uses whatever `--query-source` selects,
the vector component always embeds the full question text:

```bash
uv run company-researcher evaluate-retrieval --retrieval-method hybrid
```

### Measured result

Measured against the same 6 Gymshark questions, combining the hand-tuned
lexical baseline (`--query-source dataset`, the CLI default) with the
vector-only baseline above:

| Question | Recall@5 | Recall@10 | Reciprocal rank |
| --- | --- | --- | --- |
| q1-fy2025-turnover | 0.00 | 0.00 | 0.05 |
| q2-turnover-trend-fy2021-fy2025 | 0.00 | 0.00 | 0.08 |
| q3-fy2022-amendment-comparison | 0.00 | 0.00 | 0.04 |
| q4-directors-fy2021-fy2025 | 0.00 | 0.25 | 0.17 |
| q5-dividends-fy2022-vs-fy2025 | 0.00 | 0.00 | 0.06 |
| q6-going-concern-fy2023 | 0.50 | 0.50 | 0.20 |
| **Mean** | **0.083** | **0.125** | **0.099** |

This is a genuine negative result. Hybrid scores worse than hand-tuned
lexical alone (Mean Recall@5 = 0.625, Recall@10 = 0.833, MRR = 0.468) on
*every single question*, and only marginally better than vector alone (Mean
Recall@5 = 0.000, Recall@10 = 0.083, MRR = 0.044). Combining the two
rankings did not split the difference between them — it pulled the strong
lexical signal down almost to vector's level.

Q1 shows why, and it isn't a bug. Lexical search alone ranks the actually
relevant FY2025 turnover page (document extraction 33, page 20) at position
2 — this is why hand-tuned lexical scored RR=0.50 there. But vector search's
diagnosed weakness above means that same page doesn't appear anywhere in
vector's top 50 at all. Under RRF's `1/(60 + rank)` scoring, that page's one
strong lexical placement (`1/62 ≈ 0.0161`) loses to several irrelevant pages
that rank only moderately in *both* lists and so accumulate two smaller
contributions each — for example, extraction 44 page 35 (lexical rank 7,
vector rank 3, fused score 0.0308) and extraction 43 page 35 (lexical rank
6, vector rank 9, fused score 0.0296) both outscore it, pushing the actually
relevant page out of the fused top 10 entirely.

Equal-weighted RRF implicitly assumes both rankers place the correct page
somewhere reasonably near the top of *each* list, even if not first — that a
weaker ranker is merely noisy, not blind. This corpus violates that
assumption specifically on the questions where lexical is strongest: vector
search's diagnosed failure isn't "ranks the right page a bit lower," it's
"misses it past position 50 entirely" on exactly the year-disambiguation
questions (see the vector-only section above). Fusing with a ranker that
fails that completely, rather than merely imprecisely, doesn't average out
the error — it lets several distractors that are mediocre in both lists
outscore a page one method already found confidently.

This does not mean hybrid retrieval is a dead end on this corpus, only that
naive, equal-weighted Reciprocal Rank Fusion over these two specific
rankings, at this depth, is not competitive with hand-tuned lexical search
alone here. Weighting the two rankings unevenly, filtering out a clearly
weaker method before fusing, or a different combination strategy entirely
remain open, deliberately unexplored questions rather than assumed next
steps.

## Measure the second-company retrieval baseline: Nothing Technology

A second company, Nothing Technology Ltd (`12984564`), is now persisted alongside
Gymshark, chosen per `docs/project-brief.md`'s suggested use case
(financing-related investigation, distinguishing evidence from speculation) --
see [Scoping retrieval to one company](#scoping-retrieval-to-one-company)
below for how it was ingested and the retrieval-evaluation bugs that
ingesting it surfaced and fixed. Its filing history includes 3 accounts
filings (accounting periods ended 2021-10-31, 2022-12-31, and 2023-12-31)
and 6 registered-charge (`MR01`) filings in two batches: three charges
created 18 December 2024 naming Banco Santander, S.A. as security agent,
and three more created 1 July 2026 naming a different security agent,
Ocean II PLO LLC.

[`evaluation/nothing_technology_retrieval_questions.json`](evaluation/nothing_technology_retrieval_questions.json)
is a second hand-labelled evaluation dataset, built with the identical
methodology as Gymshark's: relevant pages identified manually by reading
the real, persisted OCR page text, hand-tuned queries chosen by measuring
against the real corpus (the same hand-tuning caveat the Gymshark dataset
already documents applies here too -- this does not show generalization
to an unseen question), documents identified by stable transaction ID.
Unlike Gymshark's single-topic accounts corpus, three of its six questions
(q2, q3, q6) span both accounts and charge filings, and q3 was deliberately
designed as this dataset's hardest question -- correctly grounding it
requires bridging the accounts' prose date format ("18 December 2024")
with the charge filings' numeric one ("18/12/2024"), the same role q3
plays in the Gymshark dataset. Score it the same way as the Gymshark
dataset, by passing its path:

```bash
uv run company-researcher evaluate-retrieval evaluation/nothing_technology_retrieval_questions.json
```

### Measured result

| Question | Recall@5 | Recall@10 | Reciprocal rank |
| --- | --- | --- | --- |
| q1-fy2023-revenue-loss | 0.67 | 1.00 | 1.00 |
| q2-registered-charges-2024-2026 | 0.83 | 1.00 | 1.00 |
| q3-december-2024-facility-evidence | 0.17 | 0.50 | 1.00 |
| q4-directors-fy2021-fy2023 | 1.00 | 1.00 | 1.00 |
| q5-revenue-trend-fy2021-fy2023 | 1.00 | 1.00 | 1.00 |
| q6-going-concern-fy2023 | 1.00 | 1.00 | 1.00 |
| **Mean** | **0.778** | **0.917** | **1.000** |

Every question's top-ranked page is relevant (RR = 1.00 throughout), even
q3, the deliberately hardest question -- its accounts-side relevant page
(the post-reporting-date-events disclosure) ranks first by itself, but the
charge filings' MR01 summary pages it also needs don't appear until much
lower, which is why its Recall@5/@10 are this dataset's weakest by a wide
margin, the same role Q3 plays in the Gymshark results above. This is a
stronger hand-tuned baseline than Gymshark's own (0.625/0.833/0.468), which
is a property of this specific 6-question set (its charges question, q2,
has an unusually clean one-page-per-document match against six near-
identically-formatted MR01 summary pages) rather than evidence that
lexical search performs better on this company's filings in general.

### Comparing the deterministic query strategies across both companies

The same two corpus-blind, deterministic query-construction strategies
measured against Gymshark were also run against this dataset:

```bash
uv run company-researcher evaluate-retrieval evaluation/nothing_technology_retrieval_questions.json --query-source derived
uv run company-researcher evaluate-retrieval evaluation/nothing_technology_retrieval_questions.json --query-source derived-idf
```

| Strategy | Recall@5 | Recall@10 | MRR | (Gymshark's own result) |
| --- | --- | --- | --- | --- |
| `derived` (stopword-only) | 0.278 | 0.417 | 0.230 | 0.000 / 0.000 / 0.030 |
| `derived-idf` (corpus-rarity ranked) | 0.250 | 0.306 | 0.193 | 0.083 / 0.250 / 0.130 |

Both strategies score meaningfully better here than they did on Gymshark's
corpus -- not a contradiction of the earlier findings, but a real,
corpus-dependent difference worth understanding rather than just noting.
Inspecting the actual derived queries surfaced a genuine, new failure
mode for `derive_discriminative_query()`, distinct from the boilerplate-
repetition problem diagnosed on Gymshark: `derive_query()`'s stopword-only
pass correctly keeps "Nothing Technology" verbatim in every question (e.g.
q4 derives to `Nothing Technology directors according set accounts FY2021
FY2023`), but `derive_discriminative_query()` drops both words from every
single question, because "nothing" is an ordinary English word that
recurs across the corpus regardless of company -- it has a corpus-wide
document frequency of 246 out of 588 persisted pages (confirmed directly
against the database, not assumed), the highest of any term checked,
because Gymshark's own filings use the word "nothing" in ordinary prose
(e.g. auditor boilerplate: "we have nothing to report in this regard").
Document-frequency-based selection has no way to distinguish "rare
because it's this company's own name" from "common because it's an
everyday word that happens to double as this company's name" -- a
different blind spot than Gymshark's "common because it's repeated
company-identifying boilerplate within its own filings" one, but the same
underlying limitation: page-level document frequency is a proxy for
discriminative power, not the thing itself, and it fails in different,
company-specific ways depending on what makes a term rare or common for
that specific corpus. Despite dropping the company name, both strategies
still score better here than on Gymshark overall, most visibly on
q6-going-concern-fy2023 (`derived-idf` scores this question a clean
1.00/1.00, because "going concern" remains genuinely rare in this smaller,
mixed-document-type corpus) -- consistent with the existing, evidenced
account of *why* IDF-based selection sometimes works (rare accounting
disclosures) and sometimes doesn't (terms that are common for reasons
IDF can't see), now shown on a second, independently measured corpus
rather than asserted to generalize from one.

## Run the investigation agent

Answer one natural-language investigation question over the persisted
corpus:

```bash
uv run company-researcher investigate "What did the directors identify as Gymshark's going-concern position in the FY2023 accounts, and does the evidence support that?"
```

Run with no argument to use that same question as the default. The command
runs a small [LangGraph](https://github.com/langchain-ai/langgraph)
`StateGraph` ([`investigation_agent.py`](src/company_researcher/investigation_agent.py))
with three linear nodes:

1. **`generate_query`** — an LLM call (via
   [`llm_client.py`](src/company_researcher/llm_client.py), an async
   OpenAI-compatible chat-completion client mirroring
   `embeddings_client.py`'s shape) turns the question into a short lexical
   search query, the same role a human played when hand-tuning the
   evaluation dataset's `query` field — except now produced at run time from
   the question alone, with no access to the correct answer.
2. **`retrieve_evidence`** — the generated query is issued to lexical
   `search_pages` only. Vector and hybrid search stay unwired here
   deliberately: the measured results above show lexical outperforming both
   on this corpus, so lexical is what the agent calls, not because it was
   the only retrieval method built.
3. **`synthesize_finding`** — a second LLM call answers the question from
   only the retrieved pages, using the provider's strict JSON-schema
   structured-output mode to return a Pydantic `Finding` (a claim, an
   `evidence_sufficient` flag, and a list of citations). Its system prompt
   also instructs the model to distinguish a filing's different voices —
   the directors' own report and notes versus the independent auditor's
   report — and rely only on the party the question actually asks about
   (see *Observed result* below for why). Every citation is then checked
   deterministically (not by an LLM judge) against the pages actually
   retrieved for that run; a citation to any other page raises
   `InvestigationAgentError` rather than silently passing through.

### Observed result

The first version of `synthesize_finding`'s prompt did not distinguish a
filing's different voices. A real run against the FY2023 going-concern
question above found the correct page (the same page eval question q6
identified as relevant) via the agent's own LLM-generated query, and the
citation-provenance check passed — but the synthesized claim also cited the
*auditor's* report page alongside the directors' own going-concern note,
attributing the auditor's opinion to the directors, the question actually
asked about. That was recorded as a genuine, uncorrected limitation rather
than smoothed over.

The system prompt was then tightened with the voice-distinction instruction
described above. Re-running the same question confirmed the fix: across
several runs, citations no longer referenced the auditor's report.

That same round of re-testing surfaced a different, still-open limitation:
intermittently, an otherwise-correct answer also cited a page from a
different filing than the one the question named. In one run,
`document_extraction_id=44` was cited — the *amended FY2022* accounts
(transaction `MzQwMTE2OTc4MmFkaXF6a2N4`), not FY2023. Two other runs of the
identical question cited only pages from the correct FY2023 filing. The
likely cause: `generate_query`'s system prompt asks for a short,
discriminative query but does not force the fiscal year into it, so when
the LLM's own query omits "2023", lexical search's literal-year-token
disambiguation — the exact mechanism that made the evaluation dataset's
hand-tuned queries score well on year disambiguation (e.g. `"Gymshark
turnover 2025"`) — does not reliably apply. Gymshark's amended FY2022
accounts reuse near-identical going-concern boilerplate to FY2023's, so an
under-specified query matches both filings, and which one
`synthesize_finding` draws from becomes a matter of LLM sampling. This is
the same "near-duplicate boilerplate across fiscal years" failure mode
already diagnosed for vector search above, now showing up via a different
path — an LLM-generated query that does not reliably include the year —
and is left open rather than prompt-patched, since it deserves the same
deliberate design pass as everything else in this milestone rather than a
rushed fix.

### Fixing the fiscal-year-disambiguation leak

The cross-fiscal-year limitation above was addressed, and the fix is
measured, not assumed to have worked. `fiscal_year_extraction.py` adds
`extract_fiscal_years()`, a deterministic regex-based function that pulls
plain 4-digit years out of a question's text (normalising an "FY" prefix
away, since filing text never uses one). `investigation_agent.py`'s
`_force_unambiguous_fiscal_year()` then appends the question's year to
`generate_query`'s LLM-generated query whenever the question names exactly
one year and the query doesn't already contain it as a literal token —
deliberately *not* applied when a question names zero years or more than
one, since the evaluation dataset's hand-tuned queries for genuine
multi-year range questions (e.g. q2 and q4, "FY2021 through FY2025")
omit any year token at all, and forcing one in for those would diverge
from that established, measured-good behaviour rather than fix anything.

This closes the originally diagnosed gap — the generated query now
reliably contains the literal year token — but real-corpus testing
surfaced a second, distinct mechanism behind the same symptom that this
fix does not close. Across 8 runs of the FY2023 going-concern question (5
via the CLI, 3 via a diagnostic script that also inspected the graph's
intermediate `retrieved_pages` state), the generated query included
"2023" in every run, confirming the query-generation gap is fixed. But
the year is only one of roughly five OR-combined terms in the query, and
near-duplicate going-concern boilerplate pages from the amended FY2022
filing (`document_extraction_id=44`) and the original FY2022 filing
(`document_extraction_id=43`) still matched enough of the *other* terms
to enter the top-5 retrieved context in every one of the 8 runs. From
there, whether `synthesize_finding` actually cited one of those
wrong-year pages alongside the correct FY2023 page came down to LLM
sampling: it happened in 2 of the 8 runs — a leak rate not clearly
better than the roughly 1-in-3 rate originally reported above, on a
sample this small.

So that first change alone was a partial fix: query generation was fixed
and retrieval ranking of the correct page was measurably strengthened,
but cross-year evidence-mixing was not eliminated, because the residual
leak happens at a different point — which near-duplicate pages survive
into `context_pages` — not at query term selection.

### Closing the residual leak with structured filing metadata

The natural next idea — filter retrieved candidates by whether their page
text literally contains the target year — was checked against the real
corpus before being built, and rejected: querying the two leaking
documents directly showed that pages from *both* the original and
amended FY2022 filings already contain the literal string "2023",
because Gymshark's amended FY2022 accounts were signed and filed in
November 2023, even though they report the year ended 31 July 2022. A
page-text filter would not have excluded them at all. This is recorded
as a genuine negative finding about the approach originally agreed, not
smoothed over.

Instead, `fiscal_year_lookup.py` adds
`document_extraction_ids_for_fiscal_year()`, which resolves which
document extractions belong to a filing whose *actual accounting
period* — Companies House's `made_up_date` field (the date accounts are
"made up to"), already persisted in each filing's `raw_filing` JSON from
ingestion — falls in a given year. This is a structured, authoritative
fact rather than an inference from OCR text, matching this project's
general principle of keeping structured facts in PostgreSQL rather than
inferring them. `search_pages()` in
`lexical_search.py` gained an optional `document_extraction_ids`
parameter (defaulting to no restriction, so retrieval evaluation's
measured baseline is provably unaffected — re-running `evaluate-retrieval`
after this change reproduced the exact same Mean Recall@5/@10/MRR as
before); `retrieve_evidence_node` now passes it whenever
`generate_query_node` determined the question names exactly one fiscal
year, restricting candidates to only that year's filing(s) before
ranking, rather than merely nudging their rank.

Measured result: re-running the FY2023 going-concern question 8 times
against the real persisted corpus, every single run cited only pages
from the correct FY2023 filing (`document_extraction_id=42`) — zero
cross-year leaks, down from 2 of 8 with the query-forcing change alone
and the roughly 1-in-3 rate originally observed. A genuine multi-year
range question (the FY2021–FY2025 directors question) was also re-run to
confirm this change doesn't affect it: `generate_query_node` correctly
determined it names more than one year, so no extraction-id restriction
was applied, and its retrieval behaviour is unchanged from before this
fix — including its pre-existing limitation that a single `context_pages`
retrieval pass struggles to gather evidence spanning five separate
filings, which is a distinct, already-known gap belonging to the
multi-step investigation milestone, not something this change was meant
to address.

This first slice was deliberately the smallest useful slice: one question
in, one finding out, no multi-step planning or looping across
sub-questions, no human-in-the-loop review, no LLM-as-judge, and no
persisted/checkpointed graph state. `search_pages` was also still not
scoped by company at this point (see
[Measure the lexical-search retrieval baseline](#measure-the-lexical-search-retrieval-baseline)) —
with only Gymshark persisted this did not yet matter in practice, but a
second company's filings would have competed unfiltered in the same
search. This has since been fixed; see
[Scoping retrieval to one company](#scoping-retrieval-to-one-company)
below.

### Multi-year investigation questions

The single-pass graph above has one structural limitation for a question
naming several fiscal years at once (e.g. "how did turnover change from
FY2021 through FY2025?"): `retrieve_evidence_node` runs one `search_pages`
call and keeps only `context_pages` (default 5) pages total, so evidence
for a question spanning 4–5 filings has to compete for those same 5 slots
— one filing's pages can crowd out another's entirely.

`investigation_agent.py` now branches on how many fiscal years a question
names. `generate_query_node` still runs first and computes, deterministically
from `extract_fiscal_years()`, either a single `fiscal_year` (unchanged,
existing behaviour) or — when 2 or more years are named — an inclusive
`fiscal_year_range` spanning the earliest to the latest named year. This
range-filling matters: `extract_fiscal_years("FY2021 through FY2025")`
returns only the literal boundary tokens `["2021", "2025"]`, but the
evaluation dataset's own q4 ("directors from FY2021 to FY2025") needs
evidence from every year in between too, not just the endpoints — checked
against the dataset's own answer key before building this, not assumed.

A question naming 0 or 1 years still goes through the original, completely
unchanged `retrieve_evidence → synthesize_finding` pass. A question naming
2+ years instead goes through two new nodes:

- **`gather_year_findings`** — sequentially, for each year in
  `fiscal_year_range`: looks up that year's filings with
  `document_extraction_ids_for_fiscal_year` (a year with no filing, e.g.
  Gymshark's FY2024, simply gets an empty restriction rather than being
  skipped), runs `search_pages` restricted to that year alone with its own
  `context_pages` budget (not shared across years — the actual fix for the
  crowding problem above), and makes one `complete_structured(Finding)` call
  scoped to only that year's evidence and reusing the same
  voice-distinguishing system prompt as the single-year path. Every sub-
  finding's citations are validated with the existing `_validate_citations`
  against only that year's own retrieved pages — the same discipline that
  fixed the single-question cross-fiscal-year citation leak, now applied
  per year instead of relying on one shared, mixed-year context window. A
  year with zero retrieved pages (the FY2024 gap case) still goes through
  this same path and naturally produces `evidence_sufficient=False`, so the
  gap is reported rather than silently dropped.
- **`aggregate_findings`** — one final `complete_structured(Finding)` call
  given each year's already-grounded claim, sufficiency flag, and citations
  (not the raw OCR page text again — grounding already happened once per
  year, so this step is a narrative/comparison layer over already-validated
  facts, not a second pass over page text). Its system prompt instructs it
  to copy citations exactly from what it was given rather than invent new
  ones, and its citations are validated with the same `_validate_citations`
  against the union of every year's retrieved pages.

`investigate()`'s return type is unchanged — still a single `Finding` (the
aggregate, for a multi-year question) — so the CLI's output contract does
not change in this slice; the per-year `YearEvidence` breakdown exists only
as internal graph state, inspectable in tests but not currently surfaced by
`company-researcher investigate`.

This was tested with four new unit tests in `test_investigation_agent.py`
against the real local Postgres instance (a fake chat client, since no
real LLM call is involved in proving the graph's routing, retrieval
scoping, or citation-validation logic): decomposition into one isolated
pass per year, a year with no filing still getting its own pass, a
per-year citation validated against only that year's own pages, and a
fabricated aggregate citation being rejected.

### Observed real-run result

The multi-year path was then run against the real LLM and the persisted
Gymshark corpus with two questions matching q2 and q4's shape. Both
completed successfully with no `InvestigationAgentError` — every citation
across both runs (4 per-year passes plus 1 aggregation, twice) validated
against the pages actually retrieved for its year, and no citation crossed
into another year's filing.

The turnover question ("How did Gymshark's turnover change year-over-year
from FY2021 through FY2025?") returned the correct figure for every year
that has a filing (FY2021 GBP437,629k, FY2022 GBP349,054k, FY2023
GBP403,818k, FY2025 GBP490,142k — all matching the evaluation dataset's
manually-verified answer key) and correctly reported no FY2024 figure,
matching the known gap documented above. One genuine nuance, recorded
rather than smoothed over: the FY2022 sub-finding's own citation (from its
own filing, `document_extraction_id=44`) quotes a geographical turnover
breakdown that does not itself state the £349,054k headline total; that
number is instead corroborated by a second, separately listed citation —
the FY2023 filing's comparative column (`document_extraction_id=42`, "2023
2022 ... Turnover 403,818 349,054"). Both citations are real, valid, and
independently verified, so this is not a validation failure, but it shows
the aggregator can lean on an adjacent year's comparative-column citation
to supply a year's headline number rather than that year's own filing
citing it as cleanly.

The directors question ("Who were Gymshark's directors and company
secretary according to each set of annual accounts from FY2021 to
FY2025?") also completed cleanly and named materially correct
directors for every year with more granular resignation/appointment
detail than the evaluation dataset's hand-picked answer (which draws from
each filing's company-information page rather than its directors'
report), and correctly reported no FY2024 information. It surfaced one
genuine, still-open limitation, though: it never found the company
secretary (present as "C Reed" on every filing's company-information
page, per the evaluation dataset), because the one query shared across
all years retrieved each filing's directors'-report page instead of its
company-information page. Both pages are real and on-topic; this is a
retrieval-precision gap in sharing a single generated query across years,
not a citation or validation bug, and is left open rather than
prompt-patched around one observed run.

A known, deliberately accepted gap: FY2024 has no filing of its own in the
persisted corpus — its only figure lives as a comparative column inside
the FY2025 filing's page. Because retrieval is restricted per-year by
`made_up_date`, an FY2024 sub-pass will correctly find nothing even though
a number for it technically exists elsewhere in the corpus. Extracting
comparative-column data from an adjacent year's filing is a distinct,
unaddressed problem, not something this milestone attempted.

### Verifying citation quotes

Inspecting the FY2022 citation from the turnover run above (see the
nuance recorded just above) surfaced a genuine, previously invisible gap:
`_validate_citations` only ever checked that a citation's
`(document_extraction_id, page_number)` was part of the retrieved
evidence — it never checked that a citation's `supporting_text` was
actually real text from that page. The FY2022 citation's quote spliced
together the header of one table with the total line of a different table
further down the same page, joined by an inserted "…" — a real page, but
a fabricated excerpt of it, which nothing in the existing evidence
contract would have caught.

`investigation_agent.py` closes this gap with `_find_quote_mismatches`, a
deterministic check (no LLM judge) run alongside `_validate_citations`:
each citation's `supporting_text` is normalized and checked for
containment in the real, equally-normalized `DocumentPage.text` it cites.
A failed check no longer fails closed immediately - `_synthesize_and_validate`
(a new helper shared by all three synthesis call sites: the single-year
path, each per-year pass, and the final aggregation) retries the
synthesis once, telling the model exactly which quote didn't match and
asking it to requote verbatim, before raising `InvestigationAgentError` if
the retry also fails. `_FINDING_SYSTEM_PROMPT` was also tightened to
require an exact, contiguous quote up front, rather than relying on the
retry alone to teach that contract.

This was first verified with unit tests (13 covering the check itself,
self-correction succeeding and failing, and the retry firing correctly in
both the single-year and multi-year paths), then run repeatedly against
the real LLM and corpus - which surfaced that a naive verbatim check was
initially too strict, and refined the design through several real
failures rather than assuming it would work:

- **OCR renders a thousands separator as "." instead of ","** (e.g.
  "437.629" for "437,629") and adds stray underscore "leader" characters
  (e.g. "__260.674") around it.
- **OCR pairs a mismatched bracket character** (e.g. "{Appointed 9
  January 2023)" for "(Appointed 9 January 2023)").
- **OCR drops a space inside a name** (e.g. "N AMcElhinney" for "N A
  McElhinney").
- **The model itself naturally reformats a page's newline-separated list**
  (e.g. one director name per line) **into a comma-separated, period-terminated
  prose sentence** when quoting it - a real quote, just not
  whitespace-for-whitespace identical to the source.

None of these involve a different word or digit sequence, only
whitespace or punctuation, so `_normalize_for_quote_check` strips commas,
periods, and underscores entirely, canonicalizes curly braces to
parentheses, and removes whitespace completely (not just collapses it)
before comparing. This is a deliberate trade-off, made explicit rather
than silently accepted: it makes the check slightly more permissive (two
different numbers, or two unrelated adjacent words, could in principle
collide once the characters between them are stripped), in exchange for
no longer rejecting a genuinely real quote purely for not reproducing
scanner noise or reflowing a list into prose. A dedicated test confirms
it does **not** become so permissive that a genuinely different fabricated
figure is missed (`437,629` and `500,000` still normalize differently).

After that refinement, re-running both the turnover and directors
questions repeatedly against the real corpus still produced occasional
`InvestigationAgentError`s - and inspecting those failures directly
(bypassing the validator to print the model's raw, rejected quote)
confirmed they are the check working correctly, not a further
normalization gap:

- One rejected citation pointed at a real page discussing FY2025 going
  concern - entirely unrelated to directors or a company secretary - when
  the corpus most likely doesn't actually contain the secretary's name in
  a form retrieval could surface for this multi-year query (see the
  directors run's "still-open limitation" recorded above); the model
  reached for irrelevant evidence and fabricated a quote from it, and this
  was correctly rejected even after a retry.
- Another rejected citation reordered content from a real page - placing
  a subsection heading before the table-header line the page actually
  states it after - which is exactly the "splice together text from
  different parts of the page" failure the finding prompt explicitly
  warns against, correctly rejected as non-contiguous even though every
  individual word was genuinely on the page.

Deliberately not pursued: normalizing further OCR character
substitutions (e.g. an apostrophe in "£'000" rendered as a degree sign)
as they turn up one at a time, or loosening the check to tolerate
reordered/non-contiguous text. Both would keep chasing individual OCR
quirks indefinitely, and the latter would gut the check's actual
purpose - the two remaining failure modes above are correct rejections
of genuinely unfaithful evidence, not false positives, so `investigate`
occasionally raising `InvestigationAgentError` on a real run is this
system refusing to serve a fabricated citation rather than a defect to
paper over.

### A reverted attempt at citation entailment checking

Quote verification proves a citation's `supporting_text` is real,
verbatim text from its page. It does not prove that text actually
substantiates the specific fact the `claim` attributes to it. A real
run surfaced exactly this gap: a citation verbatim-quoted "External D2C
sales 253,893," and the claim asserted that figure as the year's *total*
turnover - a real quote, cited for the wrong fact.

An LLM-judge entailment check was built to catch this:
`EntailmentJudgment`/`EntailmentIssue` structured models, a
`_check_entailment` call integrated into `_synthesize_and_validate`
sharing the existing quote-check's single retry budget, and 13 new tests
(all passing, all using a fake chat client). This deliberately crossed
AGENTS.md's gate on adding an LLM judge, done only after explicitly
agreeing to it as this milestone.

Real-corpus verification found a genuine, reproducible problem, not a
false alarm to shrug off. Against the FY2021 turnover citation
(`document_extraction_id=45`, page 28 - the same page whose class-of-business
table sums "External sales 398,627" + "intercompany sales 39,002" to a
"437,629" total line the page states outright), the judge flagged the
citation as unsupported on 3 of 3 runs, reasoning that intercompany
sales "should not be added" to represent total turnover - simply wrong
about what this specific page does. The system prompt was tightened to
explicitly trust a computation the source page performs itself, and
re-run: the arithmetic complaint disappeared, but two new problems
appeared in its place across another 3 runs - the judge complaining the
short quote didn't repeat a year/heading label that was genuinely
present a few lines above it on the full page, and, more seriously, a
verdict that contradicted its own stated reason (literally writing "so
it does support the claim entirely" or "which is correct" as the
*reason* for flagging a citation as unsupported). Showing the judge the
full cited page alongside the quote (not just the isolated excerpt) was
tried next, specifically to fix the missing-label complaints - but the
self-contradictory verdicts persisted across a further 3 runs.

Two rounds of prompt tuning (6 real runs total) did not fix a judge that
sometimes writes a reason affirming support and flags the citation as
unsupported anyway - that is a reliability defect, not a wording
problem, and continuing to iterate on prompt text against it would have
been guessing against LLM sampling noise rather than fixing a diagnosed
cause, the opposite of how every other result in this project was
reached. Because the check fails closed, shipping it as-is would have
made `investigate` error out on a real, correct answer to one of this
project's own two canonical multi-year regression questions more often
than not - a net reliability regression, not a rough edge worth
documenting and shipping anyway.

The entailment-checking code and its tests were reverted rather than
merged. This is a genuine negative result, recorded rather than
smoothed over, in the same spirit as the vector-only and naive hybrid
retrieval baselines earlier in this project: built, measured, found to
underperform on a directly relevant real case, and deliberately not
made part of the active system. `investigate` remains at the citation
quote-verification milestone documented above. The project brief's own
"LLM-as-a-judge" section calls for calibrating a judge against human
judgement before trusting it; that calibration work is real, separate,
and has not been done - a prompt tightened against 3-6 observed runs is
not a substitute for it, and revisiting entailment checking should wait
for that dedicated effort rather than another ad hoc prompt pass.

### Running the agent against a second company: Nothing Technology

With Nothing Technology ingested, company-scoped, and covered by its own
evaluation dataset (see
[Measure the second-company retrieval baseline: Nothing Technology](#measure-the-second-company-retrieval-baseline-nothing-technology)),
`investigate` was run against it for the first time with a real
financing/charges question:

```bash
uv run company-researcher investigate "What significant financing-related events has Nothing Technology been involved in, and does the evidence support interpreting the charges registered in December 2024 as a sign of financial distress?" --company-number 12984564
```

The first real run returned `evidence_sufficient=false` with zero
citations, despite the corpus containing strong, directly relevant
evidence (this exact topic is what the evaluation dataset's own
`q2-registered-charges-2024-2026` question, measured just above, scores
Recall@5=0.83 on). Diagnosed rather than assumed, with two direct
checks: `extract_fiscal_years()` pulls `"2024"` out of the question's
"registered in December 2024" phrase, and
`document_extraction_ids_for_fiscal_year(session, "2024")` resolves to
an empty list, confirmed directly against the database - no filing for
either persisted company has an accounting period ending in 2024
(Nothing Technology's own periods are FY2021/FY2022/FY2023, and its
`MR01` charge filings, unlike its accounts filings, have no accounting
period at all). `search_pages` treats an empty
`document_extraction_ids` list as "match nothing," not "no
restriction," so this single-year fiscal-year-forcing mechanism -
built and validated entirely against Gymshark's accounts-only corpus,
where every year named in a question really does refer to an accounting
period - silently zeroed out retrieval before the generated query was
even issued. This is a new failure mode, not a repeat of the company-
scoping bugs above: it is specifically about a year in the question
referring to something other than an accounting period (an event/charge-
creation date), which Gymshark's single-document-type corpus never
exercised.

`retrieve_evidence_node` (`investigation_agent.py`) now falls back to no
restriction when `document_extraction_ids_for_fiscal_year` returns
empty, deliberately scoped to only the single-year path: the multi-year
`gather_year_findings_node` path is untouched, so a genuinely absent
year within a named range (e.g. Gymshark's FY2024 comparative-column
gap, see above) still correctly reports `evidence_sufficient=false` for
that year rather than silently widening to every year's filings. A new
regression test
(`test_investigate_falls_back_to_unrestricted_search_when_named_year_matches_no_filing`)
covers this. Re-running the same question after the fix retrieved real
evidence, but surfaced a second, distinct real-run failure: a citation
to the correct page (the FY2023 directors' report's post-reporting-date
disclosure) was rejected by `_find_quote_mismatches` even though the
model's quote was accurate. Diagnosed by pulling the actual page text
and diffing it against the rejected quote character-by-character (the
same method used for every prior quote-verification failure): the page
reads `"a £30m debt ©\n-fundraising in order"`, where OCR inserted a
stray "©" character and a line-wrap hyphenation break exactly where the
model's clean quote says `"debt fundraising"` - genuine content, an OCR
scanning artifact from a PDF carrying DocuSign watermark elements
Gymshark's filings never had, not a fabrication.

This directly revisits a decision already recorded above: after fixing
four OCR-noise patterns for Gymshark's corpus, this project deliberately
stopped chasing further individual quirks rather than treat quote
verification as open-ended whack-a-mole against one corpus's scanner
noise. Ingesting a second, independently-scanned company's filings is a
different situation - it surfaces that corpus's *own* real, distinct OCR
artifacts, not another quirk of the same corpus - so
`_normalize_for_quote_check` now also strips "©" and "-" (hyphens),
alongside the existing comma/period/underscore/brace handling. A new
regression test
(`test_normalize_for_quote_check_tolerates_a_stray_symbol_at_a_linewrap_hyphen`)
uses this exact real page/quote pair. Re-running the question three more
times against the real LLM and corpus after both fixes completed with
zero `InvestigationAgentError`s across all three runs, each producing a
distinct but consistently well-grounded, appropriately hedged claim
(e.g. "the evidence does not support interpreting these charges as a
sign of financial distress" - correctly declining to over-interpret
routine facility security as a distress signal), and the existing
default Gymshark investigation was re-run and confirmed unaffected by
either fix.

### A known limitation: a filing that structurally lacks the requested fact

Testing the multi-year investigation path (see
[Multi-year investigation questions](#multi-year-investigation-questions))
against Nothing Technology with `"How did Nothing Technology's revenue
change from FY2021 through FY2023?"` surfaced a genuine, reproducible
limitation, distinct from both fixes above -- diagnosed but deliberately
left undocumented-and-unfixed for now rather than papered over with
another prompt tweak.

Nothing Technology's FY2021 accounts took a small-company audit exemption
that explicitly excludes the Profit and Loss account (the filing itself
states "the option not to file the Profit and Loss Account has been
taken"), so the FY2021 per-year retrieval pass's only available evidence
-- a balance sheet, with no revenue figure anywhere in it -- structurally
cannot answer a revenue question. Across three real runs, rather than
setting `evidence_sufficient=false` as `_FINDING_SYSTEM_PROMPT` instructs
when the evidence is insufficient, the model instead fabricated a
citation each time: once citing "Trade debtors 1,218,206" as though it
were revenue, another time splicing a real 2022 revenue-breakdown figure
together with an unrelated 2021 exchange-losses figure from a different
table further down the same page. Both were confirmed as genuine
non-contiguous splices by pulling the real page text and checking
directly, not assumed -- exactly the failure mode `_find_quote_mismatches`
is designed to catch, and it caught both, even after a self-correction
retry, correctly raising `InvestigationAgentError` rather than serving a
fabricated citation.

This is a different category of problem than the two fixes above. Those
were deterministic pipeline bugs (a filtering gap, a normalizer gap).
This is the model's own reliability at following its "report insufficient
evidence rather than guess" instruction when *partial-but-wrong* evidence
is present -- the same category of problem the reverted citation-
entailment-checking milestone already ran into and found unreliable to
chase with prompt tuning across a handful of observed runs (see "A
reverted attempt at citation entailment checking" above). Rather than
repeat that mistake, this is recorded as a genuine, diagnosed, currently
unresolved limitation, the same way Gymshark's own FY2024 comparative-
column gap is recorded above -- except this is a materially different
case: FY2024 there had *no filing at all* for that year, which the system
already handles gracefully (an empty per-year retrieval pass correctly
reports `evidence_sufficient=false` with no fabrication). Here, a filing
*does* exist for the named year, but structurally lacks the specific
disclosure the question asks about -- a case the system does not
currently distinguish from "the answer just wasn't in the top
`context_pages` retrieved," and the model does not reliably recognize on
its own. A deterministic (non-LLM-judge) way to detect "this year's only
retrieved evidence cannot structurally answer this question" before
synthesis is a real, open design question, not a quick patch, and is left
for a future, deliberately scoped pass rather than solved here.

## Scoping retrieval to one company

`search_pages()` in [`lexical_search.py`](src/company_researcher/lexical_search.py)
now takes an optional `company_number` parameter, joining
`DocumentPage -> DocumentExtraction -> FilingDocument -> Filing` to filter on
`Filing.company_number` (this join path was verified against
[`db/models.py`](src/company_researcher/db/models.py) before writing the
query, not assumed). It defaults to `None` (no restriction), the same
no-op-by-default pattern the fiscal-year restriction established. At the
point this parameter was added, only Gymshark was persisted, so
`retrieval_evaluation.py`'s two `search_pages` call sites — which did not
pass it yet — were provably unaffected: re-running `evaluate-retrieval`
immediately after this change reproduced the exact same Mean
Recall@5/@10/MRR reported above. That call sites *not yet* passing it
turned out to matter for real the moment a second company was ingested —
see below.

`investigate()` in
[`investigation_agent.py`](src/company_researcher/investigation_agent.py)
now requires a `company_number` argument, threaded through
`InvestigationState` to both `retrieve_evidence_node` and
`gather_year_findings_node`, which pass it to `search_pages` alongside
whatever fiscal-year restriction already applies. Unlike the fiscal-year
restriction — genuinely optional, since a question may or may not name a
year — company scope is not treated as optional here: an investigation is
always about exactly one company, so every caller must be explicit about
which one rather than risk silently searching across every persisted
company's filings once a second company exists. The CLI's `investigate`
command gained a `--company-number` flag defaulting to Gymshark's
`08130873`, so `company-researcher investigate` with no arguments keeps
working exactly as before.

This was verified against the real corpus, not just in tests: a new
`test_search_pages_restricts_to_given_company_number` test in
`test_lexical_search.py` persists two companies' pages under the same
query terms and confirms only the requested company's pages are returned;
all 15 existing `investigate()` calls in `test_investigation_agent.py`
were updated to pass an explicit `company_number`; and a real run of
`company-researcher investigate` (no arguments) against the persisted
Gymshark corpus completed successfully end-to-end with the new default
wired through the CLI, `investigate()`, and `search_pages`.

### Ingesting a second company: Nothing Technology Ltd

Nothing Technology Ltd (company number `12984564`) is now ingested as the
second company — the project brief's suggested first step toward an
unseen holdout evaluation set. Its company number was looked up against
the live Companies House website and confirmed against the real
Companies House API via `inspect` before ingesting, not guessed. It was
chosen over Made.com Design Ltd (the brief's other suggestion) because it
is usable with what this project has already built: its filing history
includes 6 registered-charge (`MR01`) filings across two creation batches
alongside its accounts filings, matching the brief's "financing-related
investigation, distinguishing evidence from speculation" framing, whereas
Made.com's main intended value — point-in-time, hindsight-leakage
analysis — needs an as-of retrieval constraint this project has not built
yet.

`company-researcher ingest 12984564` persisted its profile and 46 filing
items. `company-researcher ingest-document` and `extract-document` then
downloaded and OCR'd 9 of those filings: its 3 accounts filings (covering
accounting periods ended 2021-10-31, 2022-12-31, and 2023-12-31) and all
6 charge-creation filings. One administrative filing in the same window
(`AA01`, a pure accounting-reference-date change with no narrative
content) was deliberately skipped.

No evaluation dataset, retrieval evaluation, or investigation-agent run
has been built for Nothing Technology yet — this step only ingests and
OCR-extracts its filings. Constructing an evaluation dataset for it and
running the agent-vs-general-LLM baseline comparison the project brief
calls for remain separate, not-yet-started work.

### Cross-company contamination in retrieval evaluation, and a latent ranking-tie bug

Ingesting a second company immediately surfaced a real, measured
consequence of the company-scoping work above, not a hypothetical one.
`retrieval_evaluation.py`'s lexical `search_pages` calls were still
unscoped by company, so once Nothing Technology's pages shared the same
`document_pages` table as Gymshark's, they began competing in Gymshark's
evaluation rankings — exactly the risk this README and AGENTS.md have
flagged as "not yet mattering in practice" since the very first
retrieval-evaluation milestone. Measured effect, re-running
`evaluate-retrieval` right after ingestion: Gymshark's hand-tuned lexical
MRR moved from 0.446 to 0.427 (Recall@5/@10 unchanged) purely from this
cross-contamination — nothing else about the evaluation dataset, corpus,
or code had changed.

Since `EvaluationDataset` already carries `company_number`, and it was
already threaded through `evaluate_question` and
`evaluate_question_hybrid`'s signatures, the fix was small: pass it to
their `search_pages` calls the same way `investigation_agent.py` already
does. `vector_search.py`'s `search_pages_by_embedding` has no equivalent
company-scoping parameter and is a real, still-open gap — currently
latent only because Nothing Technology's pages have not been embedded;
the moment they are, vector and hybrid evaluation would be exposed to the
same cross-contamination. Closing that gap is deliberately left as
unstarted follow-up work here, not silently bundled into this fix.

Re-measuring after the `retrieval_evaluation.py` fix surfaced a second,
more interesting issue — a genuine latent bug the fix exposed, not a
regression it introduced. Q2's MRR did not return to its original value
even though the correct pages were once again the only candidates.
Comparing `search_pages`'s raw output for the same query, scoped vs.
unscoped, showed exactly why: three Gymshark pages tie exactly on
`ts_rank`, and `search_pages`'s `ORDER BY rank DESC` had no secondary
sort key, so PostgreSQL was free to return those tied rows in whatever
order its query plan happened to produce — an order that silently changed
the moment the company-scoping join altered that plan. This was a
pre-existing gap in a project that calls its lexical baseline
"deterministic": ties were always implicitly order-dependent on query
plan, it just never surfaced before this join existed to change the plan.
Fixed by adding `document_extraction_id, page_number` as a secondary
`ORDER BY` key in `search_pages`, so tie order is canonical regardless of
query plan; confirmed stable across three repeated `evaluate-retrieval`
runs afterward.

The net effect of both fixes, re-measured against the real corpus and
reflected in the results tables above: hand-tuned lexical Mean
Recall@5/@10 unchanged (0.625/0.833), MRR 0.446 → 0.468 (one Gymshark
tie now breaks differently under the new canonical order — a
methodology correction, not a retrieval-quality change); vector-only and
naive hybrid unaffected, since Nothing Technology has no embeddings yet;
and `derived-idf`'s MRR moved 0.130 → 0.125, which *is* a genuine,
expected change rather than a bug, since that strategy's document-
frequency statistics are explicitly computed across all persisted
document pages by design, and Nothing Technology's 350 pages are
now part of that corpus.

## Quality checks

Most of this project's tests exercise a real local PostgreSQL instance (the
`db` service from [Start PostgreSQL](#start-postgresql)) rather than mocking
the database. That's a deliberate choice, not an oversight: a meaningful
share of this codebase's correctness lives in Postgres-specific behavior —
`ts_rank` full-text search ranking, GIN index usage, constraint enforcement,
migrations — that a mock cannot faithfully reproduce. The known limitation is
that these tests currently share the same development database as manually
ingested evaluation data, rather than an isolated or ephemeral test database;
they rely on scoped cleanup and deliberately distinctive fixture data (see
[`test_lexical_search.py`](tests/test_lexical_search.py)) to avoid colliding
with it. A dedicated test database, or wrapping each test in a transaction
that is rolled back afterwards, would be the more rigorous fix. Start the
database before running the suite:

```bash
docker compose up -d db
```

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
│   ├── cli.py                          # Inspection, ingestion, extraction, embedding, evaluation, and investigation CLI
│   ├── config.py                       # Environment-backed settings
│   ├── discriminative_query.py         # Corpus document-frequency query ranking
│   ├── document_ingestion.py           # Filing-document acquisition and persistence
│   ├── embedding_persistence.py        # Idempotent page-embedding persistence
│   ├── embeddings_client.py            # Async client for the embeddings provider
│   ├── extraction_persistence.py       # Idempotent page-extraction persistence
│   ├── fiscal_year_extraction.py       # Deterministic fiscal-year extraction from question text
│   ├── fiscal_year_lookup.py           # Filing lookup by accounting period (made_up_date)
│   ├── hybrid_search.py                # Reciprocal Rank Fusion of lexical and vector rankings
│   ├── ingestion.py                    # Idempotent persistence of source data
│   ├── investigation_agent.py          # LangGraph investigation agent and citation validation
│   ├── lexical_search.py               # PostgreSQL full-text page search
│   ├── llm_client.py                   # Async client for the chat completions provider
│   ├── main.py                         # FastAPI application factory
│   ├── pdf_extraction.py               # Page-aware local PDF OCR
│   ├── query_construction.py           # Deterministic stopword-removal query derivation
│   ├── retrieval_evaluation.py         # Recall@K / MRR scoring against labelled data
│   └── vector_search.py                # pgvector cosine-distance page search
├── tests/                              # Focused unit and API tests
├── .env.example                        # Safe configuration template
├── alembic.ini                         # Alembic configuration
├── pyproject.toml                      # Package, tools, and dependencies
└── uv.lock                             # Reproducible dependency lock
```

The full product direction and intended later phases are described in
[`docs/project-brief.md`](docs/project-brief.md).
