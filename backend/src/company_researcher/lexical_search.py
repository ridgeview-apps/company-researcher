from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy import Text, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

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


def _or_combined_tsquery(query: str) -> ColumnElement[str]:
    """Build an OR-combined, stemmed tsquery expression from `query`.

    Shared by `search_pages` and `text_matches_query` so both use the exact
    same stemming and OR-combination rules rather than two independently
    tuned copies of the same logic.
    """
    stemmed_terms = func.plainto_tsquery(_TEXT_SEARCH_CONFIGURATION, query).cast(Text)
    return func.to_tsquery(
        _TEXT_SEARCH_CONFIGURATION, func.replace(stemmed_terms, " & ", " | ")
    )


async def text_matches_query(session: AsyncSession, text: str, query: str) -> bool:
    """Check whether `text` contains at least one stemmed term from `query`.

    Reuses `search_pages`'s own OR-combined, stemmed tsquery construction, so
    word-form variation (e.g. a query built from "resignations" matching
    text containing "resigned") is handled the same way retrieval already
    handles it, rather than by a separately hand-rolled matching rule.
    """
    tsvector = func.to_tsvector(_TEXT_SEARCH_CONFIGURATION, text)
    result = await session.scalar(
        select(tsvector.op("@@")(_or_combined_tsquery(query)))
    )
    return bool(result)


async def search_pages(
    session: AsyncSession,
    query: str,
    *,
    limit: int,
    document_extraction_ids: Sequence[int] | None = None,
    company_number: str | None = None,
    as_of_date: date | None = None,
) -> list[PageMatch]:
    """Rank document pages by PostgreSQL full-text search relevance to `query`."""
    # OR-combined, not AND: plainto_tsquery alone requires every term on the
    # same page, which almost never holds for a multi-word question.
    tsquery = _or_combined_tsquery(query)
    tsvector = func.to_tsvector(_TEXT_SEARCH_CONFIGURATION, DocumentPage.text)
    rank = func.ts_rank(tsvector, tsquery).label("rank")
    statement = select(
        DocumentPage.document_extraction_id, DocumentPage.page_number, rank
    ).where(tsvector.op("@@")(tsquery))
    if document_extraction_ids is not None:
        # e.g. scoping candidates to filings for one fiscal year.
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
            # Filing.date is when it was registered/became public - distinct
            # from made_up_date (accounting period end, used for fiscal-year
            # scoping elsewhere). Never falls back to "no restriction" when
            # it excludes everything: unlike the fiscal-year case, a cutoff
            # that finds nothing here is a meaningful, correct answer, not
            # an ambiguity to silently widen past.
            statement = statement.where(Filing.date <= as_of_date)
    # Ties broken by document_extraction_id then page_number so ranking is
    # deterministic regardless of query plan: ts_rank alone ties often, and
    # without an explicit secondary key Postgres can return tied rows in
    # whatever order it likes - this silently changed a real evaluation
    # score the first time a second restricting join was added here.
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
