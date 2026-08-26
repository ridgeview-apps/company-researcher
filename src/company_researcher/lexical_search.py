from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy import Text, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.db.models import (
    DocumentExtraction,
    DocumentPage,
    Filing,
    FilingDocument,
)

_TEXT_SEARCH_CONFIGURATION = "english"


@dataclass(frozen=True)
class PageMatch:
    """One document page matched by a lexical search query, ranked by relevance."""

    document_extraction_id: int
    page_number: int
    rank: float


async def search_pages(
    session: AsyncSession,
    query: str,
    *,
    limit: int,
    document_extraction_ids: Sequence[int] | None = None,
    company_number: str | None = None,
    as_of_date: date | None = None,
) -> list[PageMatch]:
    """Rank document pages by PostgreSQL full-text search relevance to `query`.

    Query terms are OR-combined rather than AND-combined: `plainto_tsquery`
    alone requires every term to appear on the same page, which almost never
    holds for a multi-word natural-language question against a single page.

    `document_extraction_ids`, when given, restricts candidates to those
    extractions before ranking -- e.g. scoping to filings for one fiscal
    year. `company_number` and `as_of_date`, when given, restrict candidates
    to pages belonging to that company's filings and/or to filings whose
    Companies House `date` (the date the filing was registered and became
    part of the public record -- distinct from `made_up_date`, a filing's
    accounting period end, used elsewhere for fiscal-year scoping) is on or
    before `as_of_date`, joining DocumentPage -> DocumentExtraction ->
    FilingDocument -> Filing to reach `Filing.company_number`/`Filing.date`.
    `as_of_date` never falls back to "no restriction" when it excludes every
    candidate: unlike the fiscal-year restriction (whose emptiness is
    ambiguous), a cutoff that finds nothing is a meaningful, correct answer
    here -- silently widening the search would defeat the reason this
    restriction exists. All three restrictions default to no restriction, so
    a caller that omits them is unaffected.

    Ties in `rank` are broken by `document_extraction_id` then
    `page_number` so ranking is fully deterministic regardless of query
    plan -- `ts_rank` alone produces exact ties fairly often (e.g. several
    pages matching the same OR-combined terms an equal number of times),
    and without an explicit secondary sort key PostgreSQL is free to
    return tied rows in whatever order its query plan happens to produce,
    which silently changed (and changed a real evaluation score) the
    first time a second `company_number`-restricting join was added here.
    """
    stemmed_terms = func.plainto_tsquery(_TEXT_SEARCH_CONFIGURATION, query).cast(Text)
    tsquery = func.to_tsquery(
        _TEXT_SEARCH_CONFIGURATION, func.replace(stemmed_terms, " & ", " | ")
    )
    tsvector = func.to_tsvector(_TEXT_SEARCH_CONFIGURATION, DocumentPage.text)
    rank = func.ts_rank(tsvector, tsquery).label("rank")
    statement = select(
        DocumentPage.document_extraction_id, DocumentPage.page_number, rank
    ).where(tsvector.op("@@")(tsquery))
    if document_extraction_ids is not None:
        statement = statement.where(
            DocumentPage.document_extraction_id.in_(document_extraction_ids)
        )
    if company_number is not None or as_of_date is not None:
        statement = (
            statement.join(
                DocumentExtraction,
                DocumentExtraction.id == DocumentPage.document_extraction_id,
            )
            .join(
                FilingDocument,
                FilingDocument.id == DocumentExtraction.filing_document_id,
            )
            .join(Filing, Filing.id == FilingDocument.filing_id)
        )
        if company_number is not None:
            statement = statement.where(Filing.company_number == company_number)
        if as_of_date is not None:
            statement = statement.where(Filing.date <= as_of_date)
    statement = statement.order_by(
        rank.desc(), DocumentPage.document_extraction_id, DocumentPage.page_number
    ).limit(limit)
    result = await session.execute(statement)
    return [
        PageMatch(
            document_extraction_id=row.document_extraction_id,
            page_number=row.page_number,
            rank=row.rank,
        )
        for row in result
    ]
