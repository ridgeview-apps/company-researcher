I’m an experienced software developer with Python experience who wants to transition into **AI Engineering**. I recently built a production-style **FastAPI + PostgreSQL backend** using async Python, migrations, CI and Docker, powering a published open-source iOS app.

I now want to build a **~2-week portfolio project** that demonstrates production AI-engineering skills rather than simply being another chatbot/RAG demo.

## Project direction

We have decided to build an **evidence-driven AI agent for investigating UK company filings**, using **Companies House** as the public dataset.

However, the important framing is:

> **This is NOT a Companies House chatbot.**

Companies House is being used as a **public, reproducible proxy for proprietary enterprise data**.

The architecture should demonstrate how the same AI system could operate over an organisation's private/internal data by swapping:

* Companies House API → internal REST APIs
* Companies House metadata → operational/internal databases
* Annual accounts/filings → internal PDFs/documents
* Filing history → audit/event history
* Company number → internal customer/supplier/entity ID
* Companies House analyst workflow → internal analyst/domain-expert workflow

I want this explanation to feature prominently in the project/README because it explains why this is a serious enterprise-AI portfolio project rather than a toy Companies House application.

The domain-specific data layer should therefore be reasonably separated from the reusable AI architecture. Don't over-engineer abstractions purely for their own sake, but make the replaceability of the public data source architecturally credible.

## Core product concept

The system should perform **controlled, reproducible investigations over Companies House data**.

Example headline task:

> "Investigate Company X between 2022 and 2025. Identify material financial, governance, financing and filing-related changes, support every finding with primary-source evidence, distinguish facts from interpretation, and flag anything requiring human review."

Another important capability is **point-in-time ("as-of") analysis**:

> "Assess Company X using only information publicly available on 31 December 2022."

The retrieval/data layer must actually enforce that date constraint so the LLM cannot see future filings. This allows us to demonstrate temporal correctness and avoid hindsight leakage.

The system should favour **exhaustive/systematic investigations** over trivial questions a normal GPT can already answer.

For example, "Who are the directors?" isn't an interesting AI task and should probably just use the structured API.

More interesting tasks are:

* Identify all material changes during a period.
* Compare several reporting periods and explain significant changes.
* Determine what changed between two point-in-time assessments.
* Investigate whether filings contain evidence explaining a financial/corporate event.
* Identify facts that warrant further human investigation.
* Determine whether the evidence actually supports a proposed interpretation.
* State explicitly when available evidence is insufficient rather than inventing an explanation.

## Important architectural principle

Use the appropriate mechanism for each type of information:

**Structured authoritative facts**
→ Companies House API / PostgreSQL

**Calculations / aggregations**
→ SQL / Python

**Unstructured information inside filings**
→ RAG

**Multi-step investigation/research**
→ LangGraph agent

**Semantic judgement**
→ LLM

**Uncertain/high-impact judgement**
→ Human-in-the-loop

The project should explicitly demonstrate that **not every problem should be solved using an LLM or vector search**.

## AI-engineering skills I want to demonstrate

The project should cover as many of these as realistically possible within two weeks:

* RAG
* embeddings
* PostgreSQL + pgvector
* lexical search
* hybrid retrieval
* metadata-aware retrieval
* reranking
* chunking strategy
* retrieval evaluation
* structured LLM outputs using Pydantic
* tool/function calling
* LangChain where genuinely useful
* LangGraph for stateful/multi-step workflows
* agent planning/research loops
* persistent agent state
* human-in-the-loop (HITL)
* approve/edit/reject/request-more-research workflows
* LLM-as-a-judge
* human-labelled evaluation data
* calibration of LLM judges against human judgement
* deterministic evals where appropriate
* offline evaluation
* regression evaluation in CI
* potentially online evaluation
* observability/tracing (e.g. LangSmith)
* token/cost/latency measurement
* model comparison
* potentially model routing
* claim/evidence verification
* citation/provenance validation
* prompt-injection/adversarial testing
* graceful handling of insufficient evidence
* temporal correctness / prevention of future-information leakage
* FastAPI
* async Python
* PostgreSQL
* migrations
* Docker
* pytest
* GitHub Actions
* potentially AWS deployment

Do NOT add technologies merely to tick boxes. Each component should have a defensible reason to exist.

## RAG/retrieval

I want to establish a simple baseline first and then experimentally improve it.

For example:

1. vector-only retrieval
2. lexical retrieval
3. hybrid retrieval
4. metadata filtering
5. hybrid + reranking

Measure whether the improvements actually help using metrics such as:

* Recall@K
* MRR
* nDCG

The README should eventually contain **real measured results**, not invented claims.

## Human-in-the-loop

HITL should be a genuine part of the workflow.

For example:

AI finding:

> "Three directors resigned within 14 months."

That fact may be well supported.

But an interpretation such as:

> "This indicates governance instability."

may not be supported by Companies House evidence.

The system should distinguish:

**Fact → evidence**

from:

**Interpretation → model judgement**

from:

**Significance → potentially human judgement**

Low-confidence, weakly supported or consequential interpretations can pause the LangGraph workflow for a human analyst to:

* approve
* edit
* reject
* request further research

Human decisions should be stored and potentially become evaluation data.

### Later analyst-review interface

Once the backend HITL workflow exists, consider adding a small TypeScript
analyst interface for reviewing findings and their cited filing pages. It could
let an analyst approve, edit, reject, or request more research while keeping
the investigation and review state in the Python backend.

This should be introduced only when it serves the real review workflow, not as
an additional technology for its own sake. Python remains the core language for
ingestion, retrieval, investigation, and evaluation; TypeScript would cover the
user-facing interaction layer.

## LLM-as-a-judge

Do not implement this merely as "score the answer from 1–10."

Use specialised structured evaluators for things such as:

* factual correctness
* groundedness
* evidence entailment
* completeness
* citation correctness
* unsupported inference
* whether human review is required

Ideally create a manually labelled subset and compare LLM-judge results against human judgement so that we demonstrate **evaluation of the evaluator**.

Use deterministic evaluation instead of LLM judges where possible.

## Important evaluation idea

One of the project's central research questions should be:

> **Does a specialised evidence-driven agent produce more complete, grounded and auditable company investigations than simply asking a frontier LLM with web access?**

Potential baselines:

1. General LLM
2. General LLM + web, instructed to use Companies House
3. Our specialised system

Possible metrics:

* factual accuracy
* material-event recall
* temporal accuracy
* future-information leakage
* citation accuracy
* unsupported claims
* completeness
* latency
* cost

Do NOT assume our system will win. Measuring where specialised architecture does and doesn't add value is itself valuable AI-engineering work.

## Candidate development companies

We discussed initially developing against only a few companies rather than trying to support dozens.

Potential development cases:

### Gymshark Ltd

Useful for:

* multi-year accounts
* temporal RAG
* amended/superseded document handling
* document versioning

Possible questions:

* What materially changed between reporting periods?
* What changed between the original and amended accounts?

### Nothing Technology Ltd

Useful for:

* structured corporate events
* registered charges
* financing-related investigation
* distinguishing evidence from speculation

Possible questions:

* What significant financing-related events occurred?
* What do the filings establish about those events?
* Does the evidence justify interpreting increased charges as financial distress?

### Made.com Design Ltd

Useful for:

* historical failure
* point-in-time analysis
* hindsight leakage
* grounded risk investigation

Possible questions:

* What could an analyst reasonably have known as of 31 December 2021?
* What concerns were visible then?
* What additional evidence became available later?

Other previously discussed potential companies include Monzo, Revolut, Octopus Energy, Britishvolt, BrewDog, Bulb Energy, Deliveroo and others.

A good strategy may be to develop/tune against a small group and reserve other companies as an **unseen holdout evaluation set**.

## Existing engineering background

I already know how to build production-style Python backend infrastructure, so I don't want the two weeks dominated by basic FastAPI/Postgres work.

The backend can use familiar technologies such as:

* Python
* FastAPI
* PostgreSQL
* SQLAlchemy
* Alembic
* asyncio
* Docker
* pytest
* GitHub Actions

The majority of the learning effort should go into the **AI-engineering layer**.

## Time constraint

I want to build this over approximately **two weeks**.

A previous rough schedule was:

**Days 1–2:** ingestion + baseline RAG
**Day 3:** initial evaluation dataset
**Day 4:** baseline retrieval/generation metrics
**Days 5–6:** hybrid retrieval + reranking
**Days 7–8:** LangGraph investigation agent + tools
**Day 9:** claim/evidence verification + self-correction
**Day 10:** HITL
**Day 11:** LLM judges + human calibration
**Day 12:** adversarial evals + regression CI
**Day 13:** tracing, latency/cost measurement + model comparison
**Day 14:** deployment/documentation/portfolio polish

This schedule is NOT fixed. I want us to challenge and refine it based on the actual architecture and scope.

## What I want to do next

Starting from this context, help me turn the project into a **concrete implementation plan**.

I want us to work out, in practical detail:

1. The exact v1 product scope.
2. Exactly which Companies House APIs/data/documents we'll use.
3. Which companies and filings we'll use for development and evaluation.
4. The architecture and component boundaries.
5. PostgreSQL/pgvector data model.
6. Document ingestion/parsing/chunking strategy.
7. Baseline RAG implementation.
8. Retrieval improvements and how we'll evaluate them.
9. LangGraph state, nodes, tools and transitions.
10. HITL workflow.
11. Evaluation dataset design.
12. LLM-as-a-judge implementation and calibration.
13. Observability.
14. Testing/CI.
15. Deployment.
16. A realistic day-by-day two-week build plan.

Please stay **very concrete** throughout this discussion. Prefer exact datasets, APIs, schemas, example questions, tools, evaluation cases and implementation decisions over generic descriptions of what an AI agent "could" do.

Also keep reminding us of the enterprise analogue where relevant:

> Companies House is the replaceable public data source; the reusable AI/retrieval/evaluation/HITL architecture is the real portfolio project.
