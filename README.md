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
lexical alone (Mean Recall@5 = 0.625, Recall@10 = 0.833, MRR = 0.446) on
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

So the honest result is a partial fix: query generation is fixed and
retrieval ranking of the correct page is measurably strengthened, but
cross-year evidence-mixing is not eliminated, because the residual leak
happens at a different point — which near-duplicate pages survive into
`context_pages` — not at query term selection. Filtering retrieved
candidates by literal year match, or some other content-level mechanism,
remains open and deliberately deferred rather than folded into this
change, since it is a larger design decision (and one that could
wrongly exclude a genuinely relevant page that doesn't happen to restate
the year) deserving its own agreed-upon design pass, not a same-session
follow-on patch.

This first slice is deliberately the smallest useful slice: one question
in, one finding out, no multi-step planning or looping across
sub-questions, no human-in-the-loop review, no LLM-as-judge, and no
persisted/checkpointed graph state. `search_pages` is also still not
scoped by company (see
[Measure the lexical-search retrieval baseline](#measure-the-lexical-search-retrieval-baseline)) —
with only Gymshark persisted this does not yet matter in practice, but a
second company's filings would compete unfiltered in the same search.

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
