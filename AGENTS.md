# Repository guidance

## Project purpose

This repository implements an evidence-driven AI agent for controlled,
reproducible investigations over UK company filings. Companies House is a
public proxy for proprietary enterprise data, not the product's defining
architecture. Keep the source-specific integration separate enough that an
internal API, database, and document store could credibly replace it.

Read `docs/project-brief.md` before making architectural or scope decisions.

## Current scope

Days 1–3 are complete. The repository can ingest company data and filing PDFs,
store immutable source artifacts by checksum, extract image-only PDFs with
page-aware OCR, and persist reproducible extraction provenance and page text.

Work incrementally. The immediate milestone is to establish a small retrieval
evaluation corpus and a deterministic PostgreSQL lexical-search baseline:

- inventory and deliberately select a small set of development/evaluation
  filings, initially focused on Gymshark;
- ingest and extract additional documents one at a time;
- define retrieval questions and manually identify relevant pages before
  tuning retrieval; and
- measure the lexical baseline with deterministic metrics such as Recall@K and
  MRR.

Do not add LLM generation, LangGraph, HITL, LLM judges, embeddings, vector or
hybrid retrieval, reranking, advanced RAG, or hard-coded historical as-of
behavior until the relevant project phase. Keep evaluation work limited to the
small dataset and deterministic retrieval metrics needed for the baseline.

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
