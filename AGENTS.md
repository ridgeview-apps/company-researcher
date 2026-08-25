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
this to Mean Recall@5 = 0.625, Recall@10 = 0.833, MRR = 0.468 (originally
measured as 0.446; see the company-scoping note near the end of this file
for why the figure moved), but those
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
lexical alone (0.625 / 0.833 / 0.468) on every question, and only
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

That fiscal-year gap has since been closed in two steps, both measured
against the real corpus rather than assumed to work.
`fiscal_year_extraction.py`'s `extract_fiscal_years()` deterministically
pulls plain 4-digit years out of a question's text, and
`investigation_agent.py`'s `_force_unambiguous_fiscal_year()` appends the
question's year to `generate_query`'s output whenever the question names
exactly one year and the query doesn't already contain it — deliberately
skipped for questions naming zero or multiple years, since the evaluation
dataset's hand-tuned queries for genuine multi-year range questions (q2,
q4) omit any year token too. That first change alone was a genuine but
partial fix: it reliably fixed query generation (confirmed across 8 real
runs), but near-duplicate going-concern pages from the amended and
original FY2022 filings still entered the retrieved top-5 context
regardless, and `synthesize_finding` still cited a wrong-year page in 2
of those 8 runs.

The obvious next idea — filter candidates by whether their page text
literally contains the target year — was checked against the corpus
before being built and found to not work: pages from *both* leaking
FY2022 filings already contain the literal string "2023", because the
amended accounts were signed and filed in November 2023 despite
reporting the year ended 31 July 2022. Instead, `fiscal_year_lookup.py`'s
`document_extraction_ids_for_fiscal_year()` resolves which document
extractions belong to a filing whose *actual accounting period*
(Companies House's `made_up_date` — the date accounts are "made up
to" — already persisted in `raw_filing` from ingestion) falls in a
given year — a structured, authoritative fact rather than a text
inference. `search_pages()` gained an optional
`document_extraction_ids` restriction (a no-op by default; re-running
`evaluate-retrieval` after this change reproduced the exact same
baseline numbers, confirming no effect on evaluation), and
`retrieve_evidence_node` now applies it whenever exactly one fiscal year
is named, excluding other years' filings from candidates entirely rather
than merely deprioritizing them.

Re-running the FY2023 going-concern question 8 more times after this
change: every run cited only the correct FY2023 filing, zero cross-year
leaks. A multi-year range question was also re-run to confirm this
doesn't affect that path (no single year is named, so no restriction is
applied); its retrieval behaviour, including its own pre-existing
limitation gathering evidence spread across five filings in one
`context_pages` pass, is unchanged and belongs to the future multi-step
investigation milestone, not this fix. See README.md's "Run the
investigation agent" section for the full detail.

This first slice was deliberately narrow: one natural-language question
in, one structured `Finding` out, no multi-step planning/looping, no HITL,
no LLM-judge, and no persisted/checkpointed graph state. `search_pages`
was also still not scoped by company at this point (a pre-existing
limitation, directly relevant here too, not just to retrieval
evaluation) — with only Gymshark persisted this did not yet matter in
practice, but a second company's filings would have competed in the same
lexical search unfiltered. This has since been fixed — see below.

The multi-step investigation milestone (handling a question that names
several fiscal years at once, e.g. a turnover or directors comparison
across FY2021–FY2025) is now built. `generate_query_node` computes,
deterministically from `extract_fiscal_years()`, an inclusive
`fiscal_year_range` whenever a question names 2+ years — filling the range
between the earliest and latest named year, not just those literal
endpoint tokens, because `extract_fiscal_years("FY2021 through FY2025")`
only returns `["2021", "2025"]`, while the evaluation dataset's own q4
needs evidence from every intervening year too (checked against the
dataset's answer key before building this). A question naming 0 or 1 years
still goes through the original, byte-for-byte unchanged
`retrieve_evidence → synthesize_finding` pass. A question naming 2+ years
instead goes through two new nodes: `gather_year_findings`, which runs one
isolated `search_pages` + `complete_structured(Finding)` pass per year
(each with its own `context_pages` budget, restricted to only that year's
filings, and its citations validated with the existing
`_validate_citations` against only that year's own retrieved pages — the
same discipline that fixed the single-question cross-fiscal-year leak, now
applied per year instead of relying on one shared, mixed-year context
window), and `aggregate_findings`, which makes one final
`complete_structured(Finding)` call over each year's already-grounded claim
(not raw OCR text again) and validates its citations against the union of
every year's retrieved pages. `investigate()`'s return type is unchanged —
still a single `Finding` — so per-year `YearEvidence` stays internal graph
state, not part of the CLI/JSON contract; this was an explicit, agreed
product decision, not an oversight. A year with no filing (e.g. Gymshark's
FY2024, whose only figure lives as a comparative column inside the FY2025
filing) still gets its own pass and reports `evidence_sufficient=False`
rather than being silently skipped — extracting that comparative-column
data is a distinct, unaddressed gap. This was verified with four new unit
tests against real Postgres (a fake chat client, since none of what they
prove — graph routing, retrieval scoping, cross-sub-result citation
validation — requires a real LLM call), then re-verified with two real
runs against the real LLM and the persisted Gymshark corpus (a q2-shaped
turnover trend question and a q4-shaped directors question). Both
completed with zero `InvestigationAgentError`s and zero cross-year
citation leaks across all 10 real LLM calls between them, and the
turnover run's per-year figures matched the evaluation dataset's answer
key exactly, including correctly reporting no FY2024 figure. The
directors run surfaced one genuine, still-open limitation: it never
found the company secretary, because the one query shared across all
years retrieved each filing's directors'-report page rather than its
company-information page (where the secretary is recorded) — a
retrieval-precision gap, not a citation/validation bug, left open rather
than prompt-patched around a single observed run. See README.md's
"Multi-year investigation questions" and "Observed real-run result"
sections for the full detail.

A citation-quote-verification and self-correction milestone is also now
built and measured, motivated by a real gap that same turnover run
surfaced: `_validate_citations` only ever confirmed a citation's page was
retrieved, never that its `supporting_text` was real text from that page
— one citation had spliced together two different tables' text with an
inserted "…". `_find_quote_mismatches` (deterministic, no LLM judge) now
checks that too, and `_synthesize_and_validate` (a new helper shared by
the single-year, per-year, and aggregation synthesis calls) retries once
with feedback before raising `InvestigationAgentError`. Built with 13 new
tests, then measured against the real LLM and corpus repeatedly, which
found the first version too strict and drove three rounds of refining
`_normalize_for_quote_check` against real, observed OCR/formatting noise
(a "." for "," thousands separator, mismatched OCR brackets, a missing
space inside a name, a newline-separated list quoted as comma-separated
prose) rather than assumed ones. After that refinement, remaining
real-run `InvestigationAgentError`s were confirmed — by inspecting the
model's raw rejected quotes directly — to be the check correctly
rejecting genuine fabrication (an unrelated page, or page content quoted
out of its real order), not further false positives, so they were left
as-is rather than chased with more normalization. See README.md's
"Verifying citation quotes" section for the full detail.

A citation-entailment-checking milestone was attempted next — an
LLM-judge check for whether a citation's (already quote-verified)
`supporting_text` actually substantiates the specific fact the claim
attributes to it, deliberately crossing AGENTS.md's LLM-judge gate after
explicit agreement — and was reverted after real-corpus verification, not
shipped. It was motivated by a real gap: a citation had verbatim-quoted
"External D2C sales 253,893" while the claim asserted that figure as the
year's *total* turnover. The built check (`EntailmentJudgment`,
`_check_entailment`, integrated into `_synthesize_and_validate`'s
existing retry budget, 13 new tests) worked in unit tests, but 6 real
runs across two rounds of prompt tightening found the judge sometimes
writes a reason affirming a citation is correct and still flags it as
unsupported in the same response — a reliability defect, not a wording
problem prompt tuning could fix. Because the check fails closed, shipping
it would have made `investigate` error on a real, correct answer to one
of this project's own two canonical multi-year regression questions more
often than not — a net reliability regression, not a documentable rough
edge. The code and tests were reverted; `investigate` remains at the
quote-verification milestone. This is a genuine negative result, recorded
the same way the vector-only and naive hybrid retrieval baselines were —
built, measured, found not to earn its place, and deliberately kept out
of the active system. See README.md's "A reverted attempt at citation
entailment checking" section for the full detail. Revisiting this needs
the dedicated LLM-judge calibration work the project brief already calls
for, not another ad hoc prompt pass.

`search_pages` (`lexical_search.py`) is now scoped by company. It gained
an optional `company_number` parameter, joining `DocumentPage ->
DocumentExtraction -> FilingDocument -> Filing` to filter on
`Filing.company_number`, verified against `db/models.py` before
implementing; it defaults to no restriction, matching the fiscal-year
restriction's no-op-by-default pattern, so `retrieval_evaluation.py`'s
call sites are unaffected — re-running `evaluate-retrieval` reproduced
the exact same measured baseline. `investigate()` now requires a
`company_number` argument (deliberately required, not optional like the
fiscal-year restriction, since an investigation is always about exactly
one company), threaded to both `retrieve_evidence_node` and
`gather_year_findings_node`. The CLI's `investigate` command gained a
`--company-number` flag defaulting to Gymshark's `08130873`, so the
zero-argument CLI invocation is unchanged. This was verified with a new
company-scoping test in `test_lexical_search.py`, updated calls across
`test_investigation_agent.py`, and a real end-to-end run of
`company-researcher investigate` against the persisted Gymshark corpus.
See README.md's "Scoping retrieval to one company" section for the full
detail. This closes the standing limitation flagged since the first
retrieval-evaluation milestone; it does not itself ingest a second
company or build the holdout evaluation set or LLM-baseline comparison
the project brief calls for — those remain separate, not-yet-started
work.

A second company, Nothing Technology Ltd (`12984564`), is now ingested —
the project brief's suggested first step toward an unseen holdout
evaluation set, chosen (over Made.com) because it is usable with what is
already built: its filing history includes registered charges (6 `MR01`
charge-creation filings, in two batches) alongside its accounts filings,
matching the project brief's "financing-related investigation,
distinguishing evidence from speculation" framing, whereas Made.com's
main value (point-in-time/hindsight-leakage analysis) needs an as-of
retrieval constraint this project has not built. Its company number was
looked up against the live Companies House website, not guessed, then
confirmed against the real Companies House API via `inspect` before
ingesting. Profile, filing history (46 filings), and 9 filing documents —
its 3 accounts filings (FY ending 2021-10-31, 2022-12-31, 2023-12-31) and
all 6 charge-creation filings — were downloaded and OCR-extracted; the
one purely administrative filing in this set (an `AA01` accounting-
reference-date change, with no narrative content) was deliberately
skipped.

Ingesting it immediately surfaced a real, measured consequence of the
company-scoping work above, rather than a hypothetical one:
`retrieval_evaluation.py`'s lexical `search_pages` calls were still
unscoped by company, so once Nothing Technology's pages shared the same
`document_pages` table, they began competing in Gymshark's evaluation
rankings — exactly the risk flagged as "not yet mattering in practice"
since the very first retrieval-evaluation milestone. Measured effect:
Gymshark's hand-tuned lexical MRR moved from 0.446 to 0.427 (Recall@5/@10
unchanged) purely from this cross-contamination. Since
`EvaluationDataset` already carries `company_number` and it was already
threaded through `evaluate_question`/`evaluate_question_hybrid`, the fix
was small: pass it to their `search_pages` calls the same way
`investigation_agent.py` does. (`vector_search.py`'s
`search_pages_by_embedding` has no equivalent company-scoping parameter
and remains an open gap — currently latent only because Nothing
Technology's pages have not been embedded; the moment they are, vector
and hybrid evaluation would be exposed to the same cross-contamination,
and closing that gap is deliberately left as unstarted follow-up work,
not silently bundled into this fix.)

Re-measuring after that fix surfaced a second, more interesting latent
issue, not a regression in the fix itself: Q2's MRR did not return to its
original value, because `search_pages`'s `ORDER BY rank DESC` had no
secondary sort key. `ts_rank` produces exact ties reasonably often (three
Gymshark pages tied for Q2), and without a deterministic tiebreak,
PostgreSQL is free to return tied rows in whatever order its query plan
produces — which silently changed the moment the company-scoping join
altered that plan, confirmed by directly comparing scoped vs. unscoped
`search_pages` output for the same query and finding the same three pages
in a different order. This was a pre-existing gap in a project that calls
its lexical baseline "deterministic," just never exposed before this
join existed. Fixed by adding `document_extraction_id, page_number` as a
secondary `ORDER BY` key, making tie order canonical regardless of query
plan; confirmed stable across repeated runs afterward.

The net effect of both fixes, re-measured against the real corpus: hand-
tuned lexical Mean Recall@5/@10 unchanged (0.625/0.833), MRR 0.446 →
0.468 (one Gymshark tie now breaks differently under the canonical
order); vector-only and naive hybrid unaffected (Nothing Technology has
no embeddings yet); `derived-idf`'s MRR moved 0.130 → 0.125, expected and
correct rather than a bug, since that strategy's document-frequency
statistics are explicitly computed corpus-wide and Nothing Technology's
350 pages are now part of that corpus. See README.md's "Scoping
retrieval to one company" section for the full detail and updated
tables.

A hand-labelled retrieval evaluation dataset for Nothing Technology is
now built and measured too — `evaluation/nothing_technology_retrieval_questions.json`,
built with the identical methodology as Gymshark's dataset (relevant
pages identified by reading real persisted OCR text, hand-tuned queries
measured against the real corpus, same hand-tuning caveat). Six
questions, three of which (charges, a December 2024 facility, going
concern) span both accounts and the six registered-charge (`MR01`)
filings — chosen because Nothing Technology's charges split into two
batches naming different security agents (Banco Santander, S.A. for
three charges created 18 December 2024; Ocean II PLO LLC for three more
created 1 July 2026), a real, filing-established fact matching the
project brief's "financing-related investigation, distinguishing
evidence from speculation" framing for this company. Measured hand-tuned
result: Mean Recall@5 = 0.778, Recall@10 = 0.917, MRR = 1.000 — stronger
than Gymshark's own hand-tuned baseline, attributed to this specific
6-question set (its charges question matches six near-identically
formatted MR01 summary pages cleanly) rather than asserted as evidence
lexical search performs better on this company's filings generally. The
deterministic `derived` and `derived-idf` strategies were also re-run
against this dataset and scored meaningfully better here than on
Gymshark (derived: 0.278/0.417/0.230 vs 0.000/0.000/0.030;
derived-idf: 0.250/0.306/0.193 vs 0.083/0.250/0.130) — investigated
rather than left unexplained: inspecting the actual derived queries
showed `derive_discriminative_query()` drops "Nothing"/"Technology" from
every question because "nothing" is an ordinary English word with the
highest corpus-wide document frequency of any term checked (246/588
pages, confirmed directly against the database), including appearing in
Gymshark's own auditor boilerplate ("we have nothing to report") — a
different blind spot than the boilerplate-repetition one diagnosed on
Gymshark (there, a term was common because it recurred within one
company's own filings; here, a term is common because it is ordinary
English that happens to double as a company name), but the same
underlying limitation: document frequency is a proxy for discriminative
power, not the thing itself. See README.md's "Measure the second-company
retrieval baseline: Nothing Technology" section for the full detail,
tables, and worked example (q6-going-concern-fy2023, where derived-idf
scores a clean 1.00/1.00 because "going concern" remains genuinely rare
even in this smaller, mixed-document-type corpus). Two new tests
(`test_load_evaluation_dataset_parses_nothing_technology_fixture`,
`test_run_evaluation_resolves_the_nothing_technology_fixture_against_real_data`)
mirror the existing Gymshark dataset tests. Evaluation-dataset
construction for Nothing Technology is now done; the agent-vs-general-LLM
baseline comparison remains separate, not-yet-started work.

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
