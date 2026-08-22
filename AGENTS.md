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

The naive lexical baseline scores poorly against the labelled corpus (Mean
Recall@5 = Recall@10 = 0.0, MRR = 0.03) — matching a whole natural-language
question against single pages by term overlap is a weak signal. That result
should inform, but not by itself decide, the next milestone: whether it is
worth first improving the deterministic lexical query before introducing
embeddings or hybrid retrieval.

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
