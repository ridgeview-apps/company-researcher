# Repository guidance

## Project purpose

This repository implements an evidence-driven AI agent for controlled,
reproducible investigations over UK company filings. Companies House is a
public proxy for proprietary enterprise data, not the product's defining
architecture. Keep the source-specific integration separate enough that an
internal API, database, and document store could credibly replace it.

Read `docs/project-brief.md` before making architectural or scope decisions.

## Current scope

The full chronological build log — every milestone, what was tried, what
was measured, and why, negative results included — lives in
[`docs/build-log.md`](docs/build-log.md). This section is a short,
current-state summary; read the build log for the reasoning and measured
numbers behind any of it.

Built and measured:

- Ingestion, checksummed artifact storage, and page-aware OCR extraction
  with reproducible provenance, against real Companies House data (Gymshark
  Ltd and Nothing Technology Ltd persisted).
- Lexical (`ts_rank`), vector (pgvector cosine), and hybrid (RRF) retrieval
  baselines, each measured against hand-labelled evaluation corpora —
  lexical wins on this corpus; vector-only and naive hybrid are documented
  negative results, not omissions.
- A LangGraph investigation agent (`generate_query → retrieve_evidence →
  synthesize_finding → validate_citations → verify_quotes →
  human_review_gate`), scoped by company, fiscal year, and as-of date, with
  multi-step multi-year decomposition and self-correcting citation-quote
  verification. The agent itself uses lexical retrieval only — vector/hybrid
  remain evaluation-only baselines, per the rule below.
- A no-retrieval general-LLM baseline and a tool-using ("General LLM +
  Companies House") baseline, both compared against the specialized agent
  on cost, latency, and hand-verified factual accuracy.
- Human-in-the-loop review (`claim_type`/`evidence_sufficient` gate,
  persisted decisions, a FastAPI + React analyst review UI).
- An offline-calibrated citation-entailment LLM judge, deliberately not
  wired into the live pipeline (see build log for why).
- Deterministic and real-LLM adversarial/prompt-injection testing (7/7
  hand-built cases now resist, after two rounds of fixes).
- GitHub Actions CI (lint/format/type-check/test) for both `backend/` and
  `web/`.

What was deliberately not built, and why, is consolidated in the root
[README's "Known limitations and deliberately deferred work"](../README.md#known-limitations-and-deliberately-deferred-work)
section (a filing structurally lacking a disclosure, one retrieval-precision
gap, the entailment judge's live reintegration, one surviving adversarial
case, Made.com, a second general-LLM baseline, AWS deployment, and the
review UI's narrower scope).

Work incrementally. Challenge and refine each step of an agreed milestone
against the actual codebase and persisted data before implementing it, the
same way the retrieval evaluation milestone was refined before any schema or
code was added.

Do not add HITL, LLM judges, reranking, advanced RAG, vector/hybrid
retrieval in the agent, or hard-coded historical as-of behavior until the
relevant project phase and until deliberately agreed as the next
milestone. Multi-step planning/looping in the agent is no longer gated —
it was the explicitly agreed milestone above — but further extensions to
it should still be deliberately agreed first, the same as any other
milestone. Keep evaluation work limited to the small dataset and
deterministic retrieval metrics needed for the baseline.

## Engineering conventions

- Use Python 3.13 and `uv` for environments, dependencies, and locking.
- Keep importable code under `backend/src/company_researcher/` and tests
  under `backend/tests/`. `web/` (TypeScript/React) is a separate sibling
  toolchain for the analyst review UI, gated on the Python backend workflow
  it reviews already existing.
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
