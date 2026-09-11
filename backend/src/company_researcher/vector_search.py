from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.db.models import (
    DocumentEmbedding,
    DocumentExtraction,
    DocumentPage,
    Filing,
    FilingDocument,
    PageEmbedding,
)


@dataclass(frozen=True)
class PageMatch:
    """One document page matched by vector search, ranked by cosine distance."""

    document_extraction_id: int
    page_number: int
    distance: float


async def search_pages_by_embedding(
    session: AsyncSession,
    query_embedding: list[float],
    *,
    provider: str,
    model: str,
    dimensions: int,
    limit: int,
    company_number: str | None = None,
) -> list[PageMatch]:
    """Rank document pages by cosine distance to `query_embedding`."""
    distance = PageEmbedding.embedding.cosine_distance(query_embedding).label(
        "distance"
    )
    statement = (
        select(DocumentPage.document_extraction_id, DocumentPage.page_number, distance)
        .select_from(PageEmbedding)
        .join(
            DocumentEmbedding,
            PageEmbedding.document_embedding_id == DocumentEmbedding.id,
        )
        .join(DocumentPage, PageEmbedding.document_page_id == DocumentPage.id)
        .where(
            # Only this provider/model/dimensions: vectors from different
            # models aren't comparable, so mixing them would be meaningless.
            DocumentEmbedding.provider == provider,
            DocumentEmbedding.model == model,
            DocumentEmbedding.dimensions == dimensions,
        )
    )
    if company_number is not None:
        # Mirrors search_pages's company-scoping join in lexical_search.py.
        # Defaults to no restriction (a latent gap while evaluation data was
        # single-company, not a currently-observed cross-company leak).
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
            .where(Filing.company_number == company_number)
        )
    statement = statement.order_by(distance.asc()).limit(limit)
    result = await session.execute(statement)
    return [
        PageMatch(
            document_extraction_id=row.document_extraction_id,
            page_number=row.page_number,
            distance=row.distance,
        )
        for row in result
    ]
