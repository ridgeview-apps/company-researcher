from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.db.models import DocumentExtraction, Filing, FilingDocument


async def document_extraction_ids_for_fiscal_year(
    session: AsyncSession, fiscal_year: str, *, company_number: str | None = None
) -> list[int]:
    """Find document extractions belonging to a filing whose accounting period ends in `fiscal_year`."""
    # Uses made_up_date (the accounting reference date, e.g. "2023-07-31"),
    # a structured fact from ingestion - not an inference from OCR page
    # text, which is unreliable here: a filing's pages can literally contain
    # a different year than its accounting period (Gymshark's amended
    # FY2022 accounts were signed in November 2023, so several pages
    # contain the literal string "2023" despite reporting FY2022).
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
        # Without this, a filing from an unrelated company sharing the same
        # accounting-period year would also match - safe to omit on a
        # single-company corpus, but once a second company shares the
        # database, an empty result (relied on elsewhere to mean "no filing
        # this year") can't be trusted unless scoped to one company, same
        # reasoning as search_pages's company scoping.
        statement = statement.where(Filing.company_number == company_number)
    result = await session.execute(statement)
    return [row[0] for row in result]
