# Repository guidance

## Project purpose

This repository implements an evidence-driven AI agent for controlled,
reproducible investigations over UK company filings. Companies House is a
public proxy for proprietary enterprise data, not the product's defining
architecture. Keep the source-specific integration separate enough that an
internal API, database, and document store could credibly replace it.

Read `docs/project-brief.md` before making architectural or scope decisions.

## Current scope

Days 1–4 are complete. The repository can ingest company data and filing PDFs,
store immutable source artifacts by checksum, extract image-only PDFs with
page-aware OCR, persist reproducible extraction provenance and page text, and
measure a deterministic PostgreSQL lexical-search baseline against a small,
manually labelled Gymshark retrieval evaluation corpus
(`evaluation/gymshark_retrieval_questions.json`). Measured results are in
`README.md`.

Matching a full natural-language question against single pages scored poorly
(Mean Recall@5 = Recall@10 = 0.0, MRR = 0.03). Using a short, targeted keyword
query instead — same `ts_rank` ranking, only the query text changed — raised
this to Mean Recall@5 = 0.625, Recall@10 = 0.833, MRR = 0.446, but those
queries were hand-tuned by trial and error directly against the 6 questions
being scored, so that result does not show generalization to an unseen
question. A deterministic, corpus-blind query-construction function,
`derive_query()` in `query_construction.py` (stopword removal only, no
knowledge of relevant pages), was added to test that generalization honestly
and scored no better than the full-sentence baseline (Mean Recall@5 =
Recall@10 = 0.0, MRR = 0.03) — a genuine negative result, not a bug. It shows
that query *brevity* was not what made the hand-tuned queries work; what
mattered was a human selecting *rare, corpus-discriminative* terms for a
specific page, which a generic stopword rule cannot replicate. All three
results are recorded in `README.md`. A corpus-aware term-selection rule
(e.g. IDF-weighted) is a more promising deterministic next step than
stopword removal, and establishing a trustworthy, non-leaked lexical
baseline this way remains a prerequisite for any honest comparison against
hybrid retrieval.

Work incrementally. The next milestone has not yet been agreed. Challenge and
refine any proposed next milestone against the actual codebase and persisted
data before implementing it, the same way the retrieval evaluation milestone
was refined before any schema or code was added.

Do not add LLM generation, LangGraph, HITL, LLM judges, embeddings, vector or
hybrid retrieval, reranking, advanced RAG, or hard-coded historical as-of
behavior until the relevant project phase and until deliberately agreed as the
next milestone. Keep evaluation work limited to the small dataset and
deterministic retrieval metrics needed for the baseline.

## Engineering conventions

- Use Python 3.13 and `uv` for environments, dependencies, and locking.
- Keep importable code under `src/company_researcher/` and tests under `tests/`.
- Prefer small modules with explicit responsibilities over speculative generic
  abstractions.
- Use async interfaces for network and database I/O where appropriate.
- Keep configuration in environment variables and never commit secrets.
- Treat Companies House responses as external, untrusted input. Handle HTTP
  errors, timeouts, pagination, and response validation explicitly.
- Keep structured authoritative facts in PostgreSQL. Do not route ordinary
  structured lookup or deterministic calculations through an LLM.
- Preserve source identifiers, filing dates, retrieval timestamps, and raw or
  reproducible source references so later evidence and temporal claims can be
  audited.
- Do not invent evaluation results, evidence, citations, or performance claims.

## Quality checks

Add or update focused tests with each behavior change. Before considering a
change complete, run the relevant formatter, linter, type checker, and tests
once those tools are configured. Keep documentation and `.env.example` aligned
with configuration changes.
