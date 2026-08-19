from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.companies_house import CompaniesHouseClient
from company_researcher.companies_house.models import CompanyProfile, FilingHistory
from company_researcher.db.models import Company, Filing


@dataclass(frozen=True)
class IngestionResult:
    """Summary of one company ingestion, for reporting to a caller."""

    company_number: str
    company_name: str
    filing_count: int


async def ingest_company(
    session: AsyncSession,
    client: CompaniesHouseClient,
    company_number: str,
) -> IngestionResult:
    """Fetch a company's profile and filing history and persist them.

    Safe to call repeatedly for the same company: existing rows are updated
    in place rather than duplicated.
    """
    profile = await client.get_company_profile(company_number)
    filing_history = await client.get_filing_history(profile.company_number)
    retrieved_at = datetime.now(UTC)

    await _upsert_company(session, profile, retrieved_at)
    await _upsert_filings(session, profile.company_number, filing_history, retrieved_at)
    await session.commit()

    return IngestionResult(
        company_number=profile.company_number,
        company_name=profile.company_name,
        filing_count=len(filing_history.items),
    )


async def _upsert_company(
    session: AsyncSession,
    profile: CompanyProfile,
    retrieved_at: datetime,
) -> None:
    values: dict[str, Any] = {
        "company_number": profile.company_number,
        "company_name": profile.company_name,
        "type": profile.type,
        "company_status": profile.company_status,
        "date_of_creation": profile.date_of_creation,
        "date_of_cessation": profile.date_of_cessation,
        "sic_codes": profile.sic_codes,
        "registered_office_address": profile.registered_office_address,
        "raw_profile": profile.model_dump(mode="json"),
        "retrieved_at": retrieved_at,
    }
    insert_statement = postgresql_insert(Company).values(**values)
    update_columns = {
        column_name: insert_statement.excluded[column_name]
        for column_name in values
        if column_name != "company_number"
    }
    statement = insert_statement.on_conflict_do_update(
        index_elements=[Company.company_number],
        set_=update_columns,
    )
    await session.execute(statement)


async def _upsert_filings(
    session: AsyncSession,
    company_number: str,
    filing_history: FilingHistory,
    retrieved_at: datetime,
) -> None:
    if not filing_history.items:
        return

    rows: list[dict[str, Any]] = [
        {
            "company_number": company_number,
            "transaction_id": item.transaction_id,
            "category": item.category,
            "type": item.type,
            "description": item.description,
            "date": item.date,
            "action_date": item.action_date,
            "barcode": item.barcode,
            "pages": item.pages,
            "paper_filed": item.paper_filed,
            "raw_filing": item.model_dump(mode="json"),
            "retrieved_at": retrieved_at,
        }
        for item in filing_history.items
    ]
    insert_statement = postgresql_insert(Filing).values(rows)
    update_columns = {
        column_name: insert_statement.excluded[column_name]
        for column_name in rows[0]
        if column_name not in ("company_number", "transaction_id")
    }
    statement = insert_statement.on_conflict_do_update(
        index_elements=[Filing.company_number, Filing.transaction_id],
        set_=update_columns,
    )
    await session.execute(statement)
