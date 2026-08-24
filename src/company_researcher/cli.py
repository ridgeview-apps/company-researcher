import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import select

from company_researcher.artifact_store import ArtifactStoreError, LocalArtifactStore
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
)
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.document_ingestion import (
    DocumentIngestionError,
    ingest_filing_document,
)
from company_researcher.embedding_persistence import embed_document_extraction
from company_researcher.embeddings_client import EmbeddingsClient, EmbeddingsError
from company_researcher.extraction_persistence import extract_filing_document
from company_researcher.ingestion import ingest_company
from company_researcher.investigation_agent import InvestigationAgentError, investigate
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
DEFAULT_INVESTIGATION_QUESTION = (
    "What did the directors identify as Gymshark's going-concern position "
    "in the FY2023 accounts, and does the evidence support that?"
)


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


async def investigate_command(question: str) -> str:
    """Run the investigation agent for one question and serialize its finding."""
    settings = Settings()
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            async with ChatClient.from_settings(settings) as chat_client:
                finding = await investigate(session, chat_client, question)
    finally:
        await engine.dispose()

    payload = {
        "question": question,
        "claim": finding.claim,
        "evidence_sufficient": finding.evidence_sufficient,
        "citations": [citation.model_dump() for citation in finding.citations],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def run_investigation(question: str) -> str:
    """Run one investigation from the synchronous CLI."""
    return asyncio.run(investigate_command(question))


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
            output = run_investigation(args.question)
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

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
