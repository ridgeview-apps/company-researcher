from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.db.models import DocumentEmbedding, DocumentPage, PageEmbedding


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
) -> list[PageMatch]:
    """Rank document pages by cosine distance to `query_embedding`.

    Only searches embeddings persisted under the given provider/model/
    dimensions configuration: vectors from different models are not
    comparable, so mixing them would produce a meaningless ranking.
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
        .order_by(distance.asc())
        .limit(limit)
    )
    result = await session.execute(statement)
    return [
        PageMatch(
            document_extraction_id=row.document_extraction_id,
            page_number=row.page_number,
            distance=row.distance,
        )
        for row in result
    ]
