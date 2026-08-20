import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from sqlalchemy import select

from company_researcher.artifact_store import LocalArtifactStore
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
from company_researcher.db.models import Filing
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.document_ingestion import (
    DocumentIngestionError,
    ingest_filing_document,
)
from company_researcher.ingestion import ingest_company


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Company Researcher command-line interface."""
    args = build_parser().parse_args(argv)

    try:
        if args.command == "ingest-document":
            output = run_document_ingestion(args.company_number, args.transaction_id)
        elif args.command == "ingest":
            output = run_ingestion(args.company_number)
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

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
