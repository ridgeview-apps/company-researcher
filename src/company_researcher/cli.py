import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from company_researcher.companies_house import CompaniesHouseClient
from company_researcher.companies_house.exceptions import (
    CompaniesHouseAuthenticationError,
    CompaniesHouseConfigurationError,
    CompaniesHouseConnectionError,
    CompaniesHouseNotFoundError,
    CompaniesHouseRateLimitError,
    CompaniesHouseResponseError,
)
from company_researcher.config import Settings
from company_researcher.db.session import create_database_engine, create_session_factory
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Company Researcher command-line interface."""
    args = build_parser().parse_args(argv)

    try:
        if args.command == "ingest":
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

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
