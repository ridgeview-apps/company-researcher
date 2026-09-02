from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.db.models import DocumentExtraction, Filing, FilingDocument


async def document_extraction_ids_for_fiscal_year(
    session: AsyncSession, fiscal_year: str, *, company_number: str | None = None
) -> list[int]:
    """Find document extractions belonging to a filing whose accounting period ends in `fiscal_year`.

    Uses each filing's Companies House `made_up_date` (its accounting
    reference date, e.g. "2023-07-31"), already persisted in `raw_filing`
    from ingestion -- a structured, authoritative fact, not an inference
    from OCR page text. This matters because page text is unreliable for
    this: a filing's pages can literally contain a *different* year than
    its accounting period (e.g. Gymshark's amended FY2022 accounts were
    signed in November 2023, so several of its pages contain the literal
    string "2023" despite reporting the year ended 31 July 2022).

    `company_number`, when given, restricts the match to that company's own
    filings. Without it, a filing from an unrelated company sharing the same
    accounting-period year would otherwise be returned -- a real, observed
    failure: a single-company corpus made this safe to omit at first, but
    once a second company's filings share the same database, an empty
    result (relied on elsewhere to mean "no filing this year") can no
    longer be trusted unless the search was scoped to one company to begin
    with, the same reasoning that scoped `search_pages` by company.
    """
    made_up_date = Filing.raw_filing["description_values"]["made_up_date"].astext
    statement = (
        select(DocumentExtraction.id)
        .join(
            FilingDocument, FilingDocument.id == DocumentExtraction.filing_document_id
        )
        .join(Filing, Filing.id == FilingDocument.filing_id)
        .where(made_up_date.like(f"{fiscal_year}-%"))
    )
    if company_number is not None:
        statement = statement.where(Filing.company_number == company_number)
    result = await session.execute(statement)
    return [row[0] for row in result]
