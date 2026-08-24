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
specific page, which a generic stopword rule cannot replicate. A second
deterministic strategy, `derive_discriminative_query()` in
`discriminative_query.py`, ranks `derive_query()`'s content words by
document frequency across all persisted document pages (still corpus-blind,
still no knowledge of relevant pages) and keeps only the rarest few. It
improved on stopword-only (Mean Recall@10 0.0 → 0.25, MRR 0.03 → 0.13) but
remained well below hand-tuned, because page-level document frequency
conflates "rare because unimportant" with "common because it's structural
boilerplate repeated within every filing" — e.g. it drops `directors` and
`Gymshark` from the directors question despite both being exactly the right
terms, because they recur across dozens of pages per filing. All four
results are recorded in `README.md`. This is now a reasonably explored,
non-leaked lexical baseline: closing the remaining gap through smarter
deterministic *lexical* term selection alone has diminishing returns on a
corpus this size, which is a better-evidenced motivation for exploring
semantic (embedding-based) retrieval next than the original schedule's
assumption — though not evidence that it would outperform lexical search
outright.

The embeddings and vector-only retrieval milestone is now complete and
measured. Embedding generation/persistence (`embeddings_client.py`,
`embedding_persistence.py`, `document_embeddings`/`page_embeddings` tables,
`embed-document` CLI) mirrors the OCR extraction phase's conventions
throughout; see earlier history for that detail. `vector_search.py` ranks
persisted page embeddings by cosine distance (pgvector `<=>`, via
`PageEmbedding.embedding.cosine_distance()`), and `evaluate-retrieval
--retrieval-method vector` embeds each question's full text (not a keyword
query — embeddings are not diluted by extra context the way `ts_rank` is)
and scores it the same way as the lexical strategies.

Measured against the same 6 Gymshark questions, using the real persisted
corpus: Mean Recall@5 = 0.0, Recall@10 = 0.083, MRR = 0.044 — worse than
every lexical strategy, including the full-sentence baseline. Diagnosed
cause (see `README.md` for the worked examples): dense embeddings capture
*topic* well but are comparatively weak at distinguishing *which year's*
instance of a heavily templated, regulation-driven annual disclosure they
are looking at — e.g. the FY2025 turnover page ranked outside the top 50
entirely, beaten by an FY2023 KPI table that also mentions "turnover"; the
FY2023 going-concern note lost to the FY2021 going-concern note's
near-identical boilerplate. This is the mirror image of the lexical corpus's
weakness: lexical's literal year-token match trivially disambiguates fiscal
years (why hand-tuned querying scored so well) but struggles with
vocabulary/paraphrase gaps; vector search handles paraphrase well but
struggles with fine-grained temporal disambiguation between near-duplicate
template text. Neither dominates the other on this corpus. That is now a
concrete, evidenced case for hybrid retrieval as the next milestone — not
an assumed one.

The hybrid retrieval milestone is now complete and measured, and it
produced a genuine negative result. `hybrid_search.py`'s
`reciprocal_rank_fusion()` combines `search_pages()` and
`search_pages_by_embedding()` by rank position (`sum(1 / (k + rank))`,
k=60) rather than raw score, since `ts_rank` and cosine distance are on
incomparable, oppositely-oriented scales that a value-based combination
would need to normalize first. `evaluate-retrieval --retrieval-method
hybrid` runs both rankings — the lexical component using whatever
`--query-source` selects, the vector component always the full question
text, matching each method's own established convention — and fuses them
before scoring.

Measured against the same 6 Gymshark questions, combining hand-tuned
lexical (`--query-source dataset`, the CLI default) with vector: Mean
Recall@5 = 0.083, Recall@10 = 0.125, MRR = 0.099 — worse than hand-tuned
lexical alone (0.625 / 0.833 / 0.446) on every question, and only
marginally better than vector alone (0.000 / 0.083 / 0.044). Diagnosed
cause (see `README.md` for the worked Q1 example): equal-weighted RRF
implicitly assumes both rankers place the correct page somewhere reasonably
near the top of *each* list, even if not first. Vector search's diagnosed
weakness on this corpus breaks that assumption on exactly the questions
where lexical is strongest — it doesn't rank the correct year-specific page
a bit lower, it misses it past position 50 entirely — so fusing lets
several distractors that are merely mediocre in both lists accumulate two
contributions and outscore a page lexical search already found confidently
with one. This is not evidence that hybrid retrieval is a dead end here,
only that naive equal-weighted RRF over these two rankings, at this depth,
underperforms hand-tuned lexical search alone on this corpus. Weighting the
rankings unevenly, filtering out a clearly weaker method before fusing, or
a different combination strategy remain open, deliberately unexplored
questions rather than assumed next steps.

Further tuning of that fusion (asymmetric weighting, filtering out the
weaker method, a different combination strategy) was considered and
deliberately deferred rather than pursued now: with only 6 evaluation
questions, tuning fusion parameters against them risks the same
hand-tuning bias already flagged for the lexical query, with no evidence
it would generalize. The LangGraph investigation-agent milestone is the
deliberately agreed next step instead, not an assumed one; it will inherit
retrieval's current limitations, particularly the temporal-disambiguation
gap diagnosed above, rather than block on further retrieval tuning.

The LangGraph investigation-agent milestone's first slice is now built and
has been run against the real persisted Gymshark corpus, not just in tests.
`llm_client.py` adds an async, OpenAI-compatible chat-completion client
(mirroring `embeddings_client.py`'s shape, including strict-mode structured
output for Pydantic response models) and `investigation_agent.py` adds a
three-node LangGraph `StateGraph` — `generate_query` (LLM, from the
natural-language question) → `retrieve_evidence` (lexical `search_pages`
only, per the measured retrieval results above) → `synthesize_finding`
(LLM, structured `Finding` output) — exposed via `company-researcher
investigate [question]`. Unlike the evaluation dataset's hand-tuned
queries, `generate_query` is produced by the LLM at run time from the
question alone; retrieval stays lexical-only because that is what the
measured results above show winning on this corpus, not because it was the
only option built. Every citation the LLM emits is validated
deterministically (not by an LLM judge) against the pages actually
retrieved for that run; a citation to any other page raises
`InvestigationAgentError` rather than silently passing through.

A real run against Gymshark's FY2023 going-concern question found the
correct evidence (the same page eval question q6 identified as relevant)
via its own LLM-generated query, but also surfaced a genuine limitation:
the synthesized claim conflated the *auditor's* opinion on going concern
with what the *directors* identified, citing the auditor's report page
alongside the directors' own note. `synthesize_finding`'s system prompt was
then tightened to explicitly distinguish a filing's different voices
(directors' own statements versus the independent auditor's report), and
re-running the same question across several repeats confirmed the fix — no
run since has cited the auditor's report.

That same re-testing surfaced a different, still-open limitation:
intermittently (one run out of three), the answer also cited a page from
the *amended FY2022* accounts rather than the FY2023 filing the question
named, because `generate_query` does not reliably force the fiscal year
into its generated query — when it's omitted, lexical search's literal-
year-token disambiguation (the exact mechanism that made the evaluation
dataset's hand-tuned queries score well on year disambiguation) doesn't
reliably apply, and Gymshark's amended FY2022 accounts reuse near-identical
going-concern boilerplate to FY2023's. This is the same "near-duplicate
boilerplate across fiscal years" failure mode already diagnosed for vector
search, now showing up via a different path.

That fiscal-year gap has since been addressed and measured, and the
result is a genuine but partial fix, not a closed issue.
`fiscal_year_extraction.py`'s `extract_fiscal_years()` deterministically
pulls plain 4-digit years out of a question's text, and
`investigation_agent.py`'s `_force_unambiguous_fiscal_year()` appends the
question's year to `generate_query`'s output whenever the question names
exactly one year and the query doesn't already contain it — deliberately
skipped for questions naming zero or multiple years, since the evaluation
dataset's hand-tuned queries for genuine multi-year range questions (q2,
q4) omit any year token too, and forcing one in there would diverge from
that established, measured-good behaviour. Across 8 real runs of the
FY2023 going-concern question (5 via the CLI, 3 via a diagnostic script
inspecting intermediate graph state), the generated query reliably
included "2023" every time — the originally diagnosed query-generation
gap is closed. But near-duplicate going-concern pages from the amended
and original FY2022 filings still entered the retrieved top-5 context in
every run regardless (the year is only one of several OR-combined terms),
and `synthesize_finding` still cited one of those wrong-year pages in 2 of
the 8 runs — a leak rate not clearly better than the roughly 1-in-3 rate
originally observed, on a small sample. The residual mechanism is
different from the one fixed: which near-duplicate pages survive into
`context_pages`, not what terms the query contains. Filtering retrieved
candidates by literal year match, or another content-level mechanism,
remains open and deliberately deferred as its own design decision rather
than folded into this fix. See README.md's "Run the investigation agent"
section for the full detail.

This first slice remains deliberately narrow: one natural-language question
in, one structured `Finding` out, no multi-step planning/looping, no HITL,
no LLM-judge, and no persisted/checkpointed graph state. `search_pages` is
also still not scoped by company (a pre-existing limitation, now directly
relevant here too, not just to retrieval evaluation) — with only Gymshark
persisted this does not yet matter in practice, but a second company's
filings would compete in the same lexical search unfiltered.

Work incrementally. Challenge and refine each step of an agreed milestone
against the actual codebase and persisted data before implementing it, the
same way the retrieval evaluation milestone was refined before any schema or
code was added.

Do not add HITL, LLM judges, reranking, advanced RAG, multi-step
planning/looping, vector/hybrid retrieval in the agent, or hard-coded
historical as-of behavior until the relevant project phase and until
deliberately agreed as the next milestone. Keep evaluation work limited to
the small dataset and deterministic retrieval metrics needed for the
baseline.

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
