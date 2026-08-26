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
    """Rank document pages by cosine distance to `query_embedding`.

    Only searches embeddings persisted under the given provider/model/
    dimensions configuration: vectors from different models are not
    comparable, so mixing them would produce a meaningless ranking.

    `company_number`, when given, restricts candidates to pages belonging
    to that company's filings, joining DocumentPage -> DocumentExtraction ->
    FilingDocument -> Filing to reach `Filing.company_number` - mirroring
    `search_pages`'s own company-scoping join in `lexical_search.py`.
    Defaults to no restriction, so an existing caller that omits it is
    unaffected; this was a latent gap (this project's evaluation datasets
    were single-company until Nothing Technology's pages were also
    embedded) rather than a currently-observed cross-company leak.
    """
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
            DocumentEmbedding.provider == provider,
            DocumentEmbedding.model == model,
            DocumentEmbedding.dimensions == dimensions,
        )
    )
    if company_number is not None:
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
