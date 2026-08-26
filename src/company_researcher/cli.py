import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from sqlalchemy import select

from company_researcher.adversarial_injection import (
    AdversarialInjectionError,
    load_injection_dataset,
    run_injection_dataset,
)
from company_researcher.artifact_store import ArtifactStoreError, LocalArtifactStore
from company_researcher.baseline_comparison import QuestionComparison, run_comparison
from company_researcher.companies_house import (
    CompaniesHouseClient,
    CompaniesHouseDocumentClient,
    normalize_company_number,
)
from company_researcher.companies_house.exceptions import (
    CompaniesHouseAuthenticationError,
    CompaniesHouseConfigurationError,
    CompaniesHouseConnectionError,
    CompaniesHouseNotFoundError,
    CompaniesHouseRateLimitError,
    CompaniesHouseResponseError,
)
from company_researcher.config import Settings
from company_researcher.db.models import (
    EMBEDDING_DIMENSIONS,
    DocumentExtraction,
    Filing,
    FilingDocument,
    HumanReview,
)
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.document_ingestion import (
    DocumentIngestionError,
    ingest_filing_document,
)
from company_researcher.embedding_persistence import embed_document_extraction
from company_researcher.embeddings_client import EmbeddingsClient, EmbeddingsError
from company_researcher.extraction_persistence import extract_filing_document
from company_researcher.human_review import (
    HumanReviewError,
    ReviewDecision,
    apply_review_decision,
    review_reason,
)
from company_researcher.ingestion import ingest_company
from company_researcher.investigation_agent import (
    InvestigationAgentError,
    investigate_with_review,
)
from company_researcher.judge_calibration import (
    JudgeCalibrationError,
    load_entailment_dataset,
    run_calibration,
)
from company_researcher.llm_client import ChatClient, ChatError
from company_researcher.pdf_extraction import PdfExtractionError, TesseractPdfExtractor
from company_researcher.retrieval_evaluation import (
    RetrievalEvaluationError,
    load_evaluation_dataset,
    run_evaluation,
    run_hybrid_evaluation,
    run_vector_evaluation,
    with_derived_queries,
    with_discriminative_queries,
)

DEFAULT_EVALUATION_DATASET = "evaluation/gymshark_retrieval_questions.json"
DEFAULT_ADVERSARIAL_DATASET = "evaluation/adversarial_injection_cases.json"
DEFAULT_INVESTIGATION_QUESTION = (
    "What did the directors identify as Gymshark's going-concern position "
    "in the FY2023 accounts, and does the evidence support that?"
)
DEFAULT_INVESTIGATION_COMPANY_NUMBER = "08130873"
DEFAULT_ENTAILMENT_DATASET = "evaluation/citation_entailment_judgments.json"


class DocumentExtractionCommandError(Exception):
    """Raised when a requested filing document cannot be extracted."""


class DocumentEmbeddingCommandError(Exception):
    """Raised when a requested document extraction cannot be embedded."""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="company-researcher",
        description="Inspect public UK company data from Companies House.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Fetch a company profile and complete filing history.",
    )
    inspect_parser.add_argument(
        "company_number",
        help="Companies House company number, for example 00000006.",
    )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Fetch a company profile and filing history and persist them.",
    )
    ingest_parser.add_argument(
        "company_number",
        help="Companies House company number, for example 00000006.",
    )

    document_parser = subparsers.add_parser(
        "ingest-document",
        help="Download and persist one filing document.",
    )
    document_parser.add_argument(
        "company_number",
        help="Previously ingested Companies House company number.",
    )
    document_parser.add_argument(
        "transaction_id",
        help="Previously ingested Companies House filing transaction ID.",
    )

    extraction_parser = subparsers.add_parser(
        "extract-document",
        help="OCR and persist the pages of one downloaded filing document.",
    )
    extraction_parser.add_argument(
        "filing_document_id",
        type=int,
        help="Database ID of a previously downloaded filing document.",
    )

    embedding_parser = subparsers.add_parser(
        "embed-document",
        help="Embed and persist the pages of one succeeded document extraction.",
    )
    embedding_parser.add_argument(
        "document_extraction_id",
        type=int,
        help="Database ID of a previously succeeded document extraction.",
    )

    evaluation_parser = subparsers.add_parser(
        "evaluate-retrieval",
        help="Measure retrieval Recall@K and MRR against a labelled question set.",
    )
    evaluation_parser.add_argument(
        "dataset_path",
        nargs="?",
        default=DEFAULT_EVALUATION_DATASET,
        help=f"Path to an evaluation dataset (default: {DEFAULT_EVALUATION_DATASET}).",
    )
    evaluation_parser.add_argument(
        "--retrieval-method",
        choices=["lexical", "vector", "hybrid"],
        default="lexical",
        help=(
            "'lexical' (default) uses PostgreSQL full-text search; "
            "--query-source selects how its query text is built. 'vector' "
            "instead embeds each question's full text with the configured "
            "embeddings provider and ranks pages by cosine distance against "
            "previously persisted page embeddings (run embed-document "
            "first); --query-source is ignored and a real embeddings API "
            "call is made per question. 'hybrid' runs both lexical (using "
            "--query-source) and vector search and combines their rankings "
            "with Reciprocal Rank Fusion; like 'vector', a real embeddings "
            "API call is made per question."
        ),
    )
    evaluation_parser.add_argument(
        "--query-source",
        choices=["dataset", "derived", "derived-idf"],
        default="dataset",
        help=(
            "Used when --retrieval-method is 'lexical' or 'hybrid' ('vector' "
            "ignores it and always embeds the full question text). 'dataset' "
            "uses each question's hand-picked 'query' field (default). "
            "'derived' ignores it and derives a query from 'text' with "
            "derive_query(), a deterministic stopword-removal rule. "
            "'derived-idf' further ranks derive_query()'s content words by "
            "document frequency across all persisted document pages and "
            "keeps only the rarest few. None of these can have been tuned "
            "to the known-relevant pages."
        ),
    )

    investigation_parser = subparsers.add_parser(
        "investigate",
        help="Run the investigation agent for one question over persisted filings.",
    )
    investigation_parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_INVESTIGATION_QUESTION,
        help=(
            "Natural-language investigation question (default: a Gymshark "
            "FY2023 going-concern question)."
        ),
    )
    investigation_parser.add_argument(
        "--company-number",
        default=DEFAULT_INVESTIGATION_COMPANY_NUMBER,
        help=(
            "Companies House company number to scope retrieval to "
            f"(default: {DEFAULT_INVESTIGATION_COMPANY_NUMBER}, Gymshark Ltd)."
        ),
    )
    investigation_parser.add_argument(
        "--as-of-date",
        type=date.fromisoformat,
        default=None,
        help=(
            "Restrict retrieval to filings publicly registered with "
            "Companies House on or before this date (YYYY-MM-DD). Filters "
            "on each filing's own registration date, not its accounting "
            "period - a filing dated after this cutoff is excluded from "
            "evidence entirely, never merely deprioritized. Omit for no "
            "restriction (default)."
        ),
    )

    review_parser = subparsers.add_parser(
        "review",
        help="Record a human decision on a pending investigation review.",
    )
    review_parser.add_argument(
        "review_id",
        type=int,
        help="ID of a pending human review (see list-reviews).",
    )
    review_parser.add_argument(
        "--decision",
        required=True,
        choices=["approve", "edit", "reject", "request-more-research"],
        help=(
            "'approve' accepts the claim as-is. 'edit' replaces the claim "
            "text with --edited-claim. 'reject' marks the claim as not to "
            "be served. 'request-more-research' records the reviewer's "
            "note without re-running the investigation - re-run investigate "
            "with a refined question separately."
        ),
    )
    review_parser.add_argument(
        "--edited-claim",
        default=None,
        help="Replacement claim text, required when --decision edit.",
    )
    review_parser.add_argument(
        "--note", default=None, help="Optional free-text note from the reviewer."
    )
    review_parser.add_argument(
        "--reviewer", default=None, help="Optional reviewer name or identifier."
    )

    list_reviews_parser = subparsers.add_parser(
        "list-reviews",
        help="List persisted human reviews, optionally filtered by status.",
    )
    list_reviews_parser.add_argument(
        "--status",
        choices=[
            "pending",
            "approved",
            "edited",
            "rejected",
            "more_research_requested",
        ],
        default=None,
        help="Only list reviews with this status (default: all).",
    )

    calibration_parser = subparsers.add_parser(
        "calibrate-judge",
        help=(
            "Measure an LLM judge's agreement with human labels on a "
            "citation-entailment calibration dataset. Offline evaluation "
            "only - does not affect investigate's live citation validation."
        ),
    )
    calibration_parser.add_argument(
        "dataset_path",
        nargs="?",
        default=DEFAULT_ENTAILMENT_DATASET,
        help=f"Path to a calibration dataset (default: {DEFAULT_ENTAILMENT_DATASET}).",
    )

    comparison_parser = subparsers.add_parser(
        "compare-baseline",
        help=(
            "Compare a no-retrieval general-LLM baseline against the "
            "specialized investigation agent over a labelled question set."
        ),
    )
    comparison_parser.add_argument(
        "dataset_path",
        nargs="?",
        default=DEFAULT_EVALUATION_DATASET,
        help=f"Path to an evaluation dataset (default: {DEFAULT_EVALUATION_DATASET}).",
    )

    injection_parser = subparsers.add_parser(
        "test-injection",
        help=(
            "Run the investigation agent against a hand-built set of "
            "prompt-injection cases over synthetic filing pages, scoring "
            "each case deterministically. Offline adversarial evaluation "
            "only - does not change investigate's live guardrails."
        ),
    )
    injection_parser.add_argument(
        "dataset_path",
        nargs="?",
        default=DEFAULT_ADVERSARIAL_DATASET,
        help=f"Path to an adversarial dataset (default: {DEFAULT_ADVERSARIAL_DATASET}).",
    )

    return parser


async def inspect_company(company_number: str) -> str:
    """Fetch and serialize a company profile and filing history."""
    settings = Settings()
    async with CompaniesHouseClient.from_settings(settings) as client:
        profile = await client.get_company_profile(company_number)
        filing_history = await client.get_filing_history(company_number)

    payload = {
        "profile": profile.model_dump(mode="json"),
        "filing_history": filing_history.model_dump(mode="json"),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def run_inspection(company_number: str) -> str:
    """Run the async inspection from a synchronous console entry point."""
    return asyncio.run(inspect_company(company_number))


async def ingest_company_command(company_number: str) -> str:
    """Fetch a company's data and persist it, reporting what was stored."""
    settings = Settings()
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        async with (
            CompaniesHouseClient.from_settings(settings) as client,
            session_factory() as session,
        ):
            result = await ingest_company(session, client, company_number)
    finally:
        await engine.dispose()

    return (
        f"Ingested {result.company_name} ({result.company_number}): "
        f"{result.filing_count} filing(s) persisted."
    )


def run_ingestion(company_number: str) -> str:
    """Run the async ingestion from a synchronous console entry point."""
    return asyncio.run(ingest_company_command(company_number))


async def ingest_document_command(company_number: str, transaction_id: str) -> str:
    """Acquire and persist one previously ingested filing document."""
    settings = Settings()
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            filing = await session.scalar(
                select(Filing).where(
                    Filing.company_number == normalize_company_number(company_number),
                    Filing.transaction_id == transaction_id.strip(),
                )
            )
            if filing is None:
                raise DocumentIngestionError(
                    "Filing is not persisted; ingest the company first"
                )

            async with CompaniesHouseDocumentClient.from_settings(settings) as client:
                result = await ingest_filing_document(
                    session,
                    client,
                    LocalArtifactStore(settings.artifact_root),
                    filing,
                )
    finally:
        await engine.dispose()

    action = "Created" if result.created else "Refreshed"
    return (
        f"{action} document {result.source_document_id}: "
        f"sha256={result.sha256} storage_key={result.storage_key}"
    )


def run_document_ingestion(company_number: str, transaction_id: str) -> str:
    """Run one filing-document ingestion from the synchronous CLI."""
    return asyncio.run(ingest_document_command(company_number, transaction_id))


async def extract_document_command(filing_document_id: int) -> str:
    """OCR and persist one previously downloaded filing document."""
    settings = Settings()
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            filing_document = await session.get(FilingDocument, filing_document_id)
            if filing_document is None:
                raise DocumentExtractionCommandError(
                    f"Filing document {filing_document_id} is not persisted; "
                    "ingest it first"
                )

            result = await extract_filing_document(
                session,
                LocalArtifactStore(settings.artifact_root),
                TesseractPdfExtractor(),
                filing_document,
            )
    finally:
        await engine.dispose()

    action = {"created": "Created", "retried": "Retried", "reused": "Reused"}[
        result.outcome
    ]
    return (
        f"{action} extraction {result.document_extraction_id}: "
        f"{result.page_count} page(s), "
        f"{result.total_character_count} character(s)."
    )


def run_document_extraction(filing_document_id: int) -> str:
    """Run one filing-document extraction from the synchronous CLI."""
    return asyncio.run(extract_document_command(filing_document_id))


async def embed_document_command(document_extraction_id: int) -> str:
    """Embed and persist the pages of one succeeded document extraction."""
    settings = Settings()
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            document_extraction = await session.get(
                DocumentExtraction, document_extraction_id
            )
            if document_extraction is None:
                raise DocumentEmbeddingCommandError(
                    f"Document extraction {document_extraction_id} is not "
                    "persisted; extract it first"
                )
            if document_extraction.status != "succeeded":
                raise DocumentEmbeddingCommandError(
                    f"Document extraction {document_extraction_id} has not "
                    f"succeeded (status={document_extraction.status})"
                )

            async with EmbeddingsClient.from_settings(settings) as embeddings_client:
                result = await embed_document_extraction(
                    session,
                    embeddings_client,
                    document_extraction,
                    provider="openai",
                    model=settings.openai_embedding_model,
                    dimensions=EMBEDDING_DIMENSIONS,
                )
    finally:
        await engine.dispose()

    action = {"created": "Created", "retried": "Retried", "reused": "Reused"}[
        result.outcome
    ]
    return f"{action} embedding {result.document_embedding_id}: {result.page_count} page(s)."


def run_document_embedding(document_extraction_id: int) -> str:
    """Run one document-extraction embedding from the synchronous CLI."""
    return asyncio.run(embed_document_command(document_extraction_id))


async def evaluate_retrieval_command(
    dataset_path: str, query_source: str, retrieval_method: str
) -> str:
    """Measure retrieval Recall@K and MRR against a labelled question set."""
    dataset = load_evaluation_dataset(Path(dataset_path))
    settings = Settings()
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            if retrieval_method in ("lexical", "hybrid"):
                if query_source == "derived":
                    dataset = with_derived_queries(dataset)
                elif query_source == "derived-idf":
                    dataset = await with_discriminative_queries(session, dataset)

            if retrieval_method == "vector":
                async with EmbeddingsClient.from_settings(
                    settings
                ) as embeddings_client:
                    summary = await run_vector_evaluation(
                        session,
                        embeddings_client,
                        dataset,
                        provider="openai",
                        model=settings.openai_embedding_model,
                        dimensions=EMBEDDING_DIMENSIONS,
                    )
            elif retrieval_method == "hybrid":
                async with EmbeddingsClient.from_settings(
                    settings
                ) as embeddings_client:
                    summary = await run_hybrid_evaluation(
                        session,
                        embeddings_client,
                        dataset,
                        provider="openai",
                        model=settings.openai_embedding_model,
                        dimensions=EMBEDDING_DIMENSIONS,
                    )
            else:
                summary = await run_evaluation(session, dataset)
    finally:
        await engine.dispose()

    lines = []
    for metrics in summary.per_question:
        recall_text = ", ".join(
            f"R@{k}={value:.2f}" for k, value in metrics.recall_at_k.items()
        )
        lines.append(
            f"{metrics.question_id}: {recall_text}, RR={metrics.reciprocal_rank:.2f}"
        )

    mean_recall_text = ", ".join(
        f"R@{k}={value:.3f}" for k, value in summary.mean_recall_at_k.items()
    )
    lines.append(f"Mean: {mean_recall_text}, MRR={summary.mean_reciprocal_rank:.3f}")
    return "\n".join(lines)


def run_retrieval_evaluation(
    dataset_path: str, query_source: str, retrieval_method: str
) -> str:
    """Run one retrieval evaluation from the synchronous CLI."""
    return asyncio.run(
        evaluate_retrieval_command(dataset_path, query_source, retrieval_method)
    )


async def investigate_command(
    question: str, company_number: str, as_of_date: date | None
) -> str:
    """Run the investigation agent for one question and serialize its finding.

    A finding whose claim is an interpretation, or whose evidence is
    insufficient, is not presented as settled: `status` is "pending_review"
    and the finding is also persisted as a `human_reviews` row (see
    `investigate_with_review`) for a human analyst to resolve with the
    `review` command, rather than served as a final answer.
    """
    settings = Settings()
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            async with ChatClient.from_settings(settings) as chat_client:
                finding, review_id = await investigate_with_review(
                    session,
                    chat_client,
                    question,
                    company_number,
                    as_of_date=as_of_date,
                )
    finally:
        await engine.dispose()

    payload: dict[str, object] = {
        "question": question,
        "company_number": company_number,
        "status": "pending_review" if review_id is not None else "final",
        "claim": finding.claim,
        "claim_type": finding.claim_type,
        "evidence_sufficient": finding.evidence_sufficient,
        "citations": [citation.model_dump() for citation in finding.citations],
    }
    if as_of_date is not None:
        payload["as_of_date"] = as_of_date.isoformat()
    if review_id is not None:
        payload["review_id"] = review_id
        payload["review_reason"] = review_reason(
            claim_type=finding.claim_type,
            evidence_sufficient=finding.evidence_sufficient,
        )
    return json.dumps(payload, indent=2, ensure_ascii=False)


def run_investigation(
    question: str, company_number: str, as_of_date: date | None
) -> str:
    """Run one investigation from the synchronous CLI."""
    return asyncio.run(investigate_command(question, company_number, as_of_date))


_REVIEW_DECISIONS: dict[str, ReviewDecision] = {
    "approve": "approved",
    "edit": "edited",
    "reject": "rejected",
    "request-more-research": "more_research_requested",
}


async def review_command(
    review_id: int,
    decision: str,
    edited_claim: str | None,
    note: str | None,
    reviewer: str | None,
) -> str:
    """Record a human analyst's decision against one pending review."""
    settings = Settings()
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            result = await apply_review_decision(
                session,
                review_id,
                _REVIEW_DECISIONS[decision],
                edited_claim=edited_claim,
                note=note,
                reviewer=reviewer,
            )
    finally:
        await engine.dispose()

    payload = {
        "review_id": result.review_id,
        "status": result.status,
        "claim": result.claim,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def run_review(
    review_id: int,
    decision: str,
    edited_claim: str | None,
    note: str | None,
    reviewer: str | None,
) -> str:
    """Run one review decision from the synchronous CLI."""
    return asyncio.run(
        review_command(review_id, decision, edited_claim, note, reviewer)
    )


async def list_reviews_command(status: str | None) -> str:
    """List persisted human reviews, optionally filtered by status."""
    settings = Settings()
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            statement = select(HumanReview).order_by(HumanReview.created_at)
            if status is not None:
                statement = statement.where(HumanReview.status == status)
            reviews = (await session.scalars(statement)).all()
    finally:
        await engine.dispose()

    if not reviews:
        return "No human reviews found."

    lines = []
    for review in reviews:
        lines.append(
            f"{review.id}: [{review.status}] company={review.company_number} "
            f"claim_type={review.claim_type} reason={review.review_reason}"
        )
        lines.append(f"    question: {review.question}")
        lines.append(f"    claim: {review.claim}")
    return "\n".join(lines)


def run_list_reviews(status: str | None) -> str:
    """List human reviews from the synchronous CLI."""
    return asyncio.run(list_reviews_command(status))


async def calibrate_judge_command(dataset_path: str) -> str:
    """Measure the entailment judge's agreement with human labels on a calibration dataset."""
    dataset = load_entailment_dataset(Path(dataset_path))
    settings = Settings()
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            async with ChatClient.from_settings(settings) as chat_client:
                summary = await run_calibration(session, chat_client, dataset)
    finally:
        await engine.dispose()

    lines = []
    for result in summary.per_example:
        mark = "agree" if result.agrees else "DISAGREE"
        lines.append(
            f"{result.example_id}: human={result.human_verdict} "
            f"judge={result.judge_verdict} [{mark}]"
        )
        lines.append(f"    judge reason: {result.judge_reason}")

    lines.append(
        f"Accuracy={summary.accuracy:.3f} "
        f"Precision(unsupported)={summary.precision_unsupported:.3f} "
        f"Recall(unsupported)={summary.recall_unsupported:.3f} "
        f"F1(unsupported)={summary.f1_unsupported:.3f}"
    )
    return "\n".join(lines)


def run_judge_calibration(dataset_path: str) -> str:
    """Run one judge-calibration pass from the synchronous CLI."""
    return asyncio.run(calibrate_judge_command(dataset_path))


def _format_comparison_report(comparisons: list[QuestionComparison]) -> str:
    """Render a per-question baseline-vs-specialized report for manual review.

    Only latency, token usage, and citation realism are reported as
    computed numbers here - factual accuracy and completeness are
    deliberately not scored automatically (see README.md's baseline-
    comparison section for why); a reader compares each printed claim
    against the dataset's own hand-verified ground truth by eye.
    """
    lines = []
    for comparison in comparisons:
        lines.append(f"{comparison.question_id}: {comparison.question_text}")

        baseline = comparison.baseline_finding
        real_count = sum(1 for r in comparison.baseline_citation_realism if r.exists)
        usage_text = (
            f", tokens={comparison.baseline_usage.total_tokens}"
            if comparison.baseline_usage is not None
            else ""
        )
        lines.append(
            f"  baseline    [{comparison.baseline_latency_seconds:.2f}s{usage_text}] "
            f"evidence_sufficient={baseline.evidence_sufficient} "
            f"citations={len(baseline.citations)} ({real_count} real)"
        )
        lines.append(f"    claim: {baseline.claim}")

        specialized_usage_text = (
            f", tokens={comparison.specialized_usage.total_tokens}"
            if comparison.specialized_usage is not None
            else ""
        )
        if comparison.specialized_error is not None:
            lines.append(
                f"  specialized [{comparison.specialized_latency_seconds:.2f}s"
                f"{specialized_usage_text}] ERROR: {comparison.specialized_error}"
            )
        elif comparison.specialized_finding is not None:
            specialized = comparison.specialized_finding
            lines.append(
                f"  specialized [{comparison.specialized_latency_seconds:.2f}s"
                f"{specialized_usage_text}] "
                f"evidence_sufficient={specialized.evidence_sufficient} "
                f"citations={len(specialized.citations)}"
            )
            lines.append(f"    claim: {specialized.claim}")
        lines.append("")

    return "\n".join(lines).rstrip("\n")


async def compare_baseline_command(dataset_path: str) -> str:
    """Compare the no-retrieval baseline against the specialized agent over a dataset."""
    dataset = load_evaluation_dataset(Path(dataset_path))
    settings = Settings()
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            async with ChatClient.from_settings(settings) as chat_client:
                comparisons = await run_comparison(session, chat_client, dataset)
    finally:
        await engine.dispose()

    return _format_comparison_report(comparisons)


def run_baseline_comparison(dataset_path: str) -> str:
    """Run one baseline comparison from the synchronous CLI."""
    return asyncio.run(compare_baseline_command(dataset_path))


async def test_injection_command(dataset_path: str) -> str:
    """Run every adversarial case in a dataset against a real chat client."""
    dataset = load_injection_dataset(Path(dataset_path))
    settings = Settings()
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            async with ChatClient.from_settings(settings) as chat_client:
                results = await run_injection_dataset(session, chat_client, dataset)
    finally:
        await engine.dispose()

    lines = []
    for result in results:
        mark = "PASS" if result.passed else "FAIL"
        lines.append(
            f"{result.case_id} [{result.case_type}] [{mark}]: {result.description}"
        )
        lines.append(f"    {result.detail}")
        if result.claim is not None:
            lines.append(f"    claim: {result.claim}")
            lines.append(
                f"    claim_type={result.claim_type} "
                f"evidence_sufficient={result.evidence_sufficient}"
            )

    passed = sum(1 for result in results if result.passed)
    lines.append(f"{passed}/{len(results)} case(s) passed.")
    return "\n".join(lines)


def run_injection_test(dataset_path: str) -> str:
    """Run one adversarial injection dataset from the synchronous CLI."""
    return asyncio.run(test_injection_command(dataset_path))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Company Researcher command-line interface."""
    args = build_parser().parse_args(argv)

    try:
        if args.command == "evaluate-retrieval":
            output = run_retrieval_evaluation(
                args.dataset_path, args.query_source, args.retrieval_method
            )
        elif args.command == "extract-document":
            output = run_document_extraction(args.filing_document_id)
        elif args.command == "embed-document":
            output = run_document_embedding(args.document_extraction_id)
        elif args.command == "ingest-document":
            output = run_document_ingestion(args.company_number, args.transaction_id)
        elif args.command == "ingest":
            output = run_ingestion(args.company_number)
        elif args.command == "investigate":
            output = run_investigation(
                args.question, args.company_number, args.as_of_date
            )
        elif args.command == "review":
            output = run_review(
                args.review_id,
                args.decision,
                args.edited_claim,
                args.note,
                args.reviewer,
            )
        elif args.command == "list-reviews":
            output = run_list_reviews(args.status)
        elif args.command == "calibrate-judge":
            output = run_judge_calibration(args.dataset_path)
        elif args.command == "compare-baseline":
            output = run_baseline_comparison(args.dataset_path)
        elif args.command == "test-injection":
            output = run_injection_test(args.dataset_path)
        else:
            output = run_inspection(args.company_number)
    except CompaniesHouseConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except CompaniesHouseAuthenticationError as error:
        print(f"Authentication error: {error}", file=sys.stderr)
        return 3
    except CompaniesHouseNotFoundError as error:
        print(f"Not found: {error}", file=sys.stderr)
        return 4
    except CompaniesHouseRateLimitError as error:
        print(f"Rate limit error: {error}", file=sys.stderr)
        return 5
    except (CompaniesHouseConnectionError, CompaniesHouseResponseError) as error:
        print(f"Companies House error: {error}", file=sys.stderr)
        return 1
    except DocumentIngestionError as error:
        print(f"Document ingestion error: {error}", file=sys.stderr)
        return 1
    except (
        DocumentExtractionCommandError,
        ArtifactStoreError,
        PdfExtractionError,
    ) as error:
        print(f"Document extraction error: {error}", file=sys.stderr)
        return 1
    except (DocumentEmbeddingCommandError, EmbeddingsError) as error:
        print(f"Document embedding error: {error}", file=sys.stderr)
        return 1
    except RetrievalEvaluationError as error:
        print(f"Retrieval evaluation error: {error}", file=sys.stderr)
        return 1
    except (InvestigationAgentError, ChatError) as error:
        print(f"Investigation error: {error}", file=sys.stderr)
        return 1
    except HumanReviewError as error:
        print(f"Human review error: {error}", file=sys.stderr)
        return 1
    except JudgeCalibrationError as error:
        print(f"Judge calibration error: {error}", file=sys.stderr)
        return 1
    except AdversarialInjectionError as error:
        print(f"Adversarial injection error: {error}", file=sys.stderr)
        return 1

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
