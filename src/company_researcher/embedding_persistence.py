from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.db.models import (
    DocumentEmbedding,
    DocumentExtraction,
    DocumentPage,
    PageEmbedding,
)
from company_researcher.embeddings_client import EmbeddingsError, EmbeddingsProvider


@dataclass(frozen=True)
class EmbeddingPersistenceResult:
    """Summary of one persisted document embedding run."""

    document_embedding_id: int
    page_count: int
    created: bool


async def embed_document_extraction(
    session: AsyncSession,
    embeddings_client: EmbeddingsProvider,
    document_extraction: DocumentExtraction,
    *,
    provider: str,
    model: str,
    dimensions: int,
) -> EmbeddingPersistenceResult:
    """Embed and persist every page of one succeeded document extraction."""
    document_embedding = await session.scalar(
        select(DocumentEmbedding).where(
            DocumentEmbedding.document_extraction_id == document_extraction.id,
            DocumentEmbedding.provider == provider,
            DocumentEmbedding.model == model,
            DocumentEmbedding.dimensions == dimensions,
        )
    )

    if document_embedding is not None and document_embedding.status == "succeeded":
        return EmbeddingPersistenceResult(
            document_embedding_id=document_embedding.id,
            page_count=document_embedding.page_count or 0,
            created=False,
        )

    pages = list(
        (
            await session.scalars(
                select(DocumentPage)
                .where(DocumentPage.document_extraction_id == document_extraction.id)
                .order_by(DocumentPage.page_number)
            )
        ).all()
    )

    created = document_embedding is None
    started_at = datetime.now(UTC)
    if document_embedding is None:
        document_embedding = DocumentEmbedding(
            document_extraction_id=document_extraction.id,
            status="running",
            provider=provider,
            model=model,
            dimensions=dimensions,
            started_at=started_at,
        )
        session.add(document_embedding)
    else:
        await session.execute(
            delete(PageEmbedding).where(
                PageEmbedding.document_embedding_id == document_embedding.id
            )
        )
        document_embedding.status = "running"
        document_embedding.page_count = None
        document_embedding.started_at = started_at
        document_embedding.completed_at = None
        document_embedding.error_message = None
    await session.commit()

    try:
        vectors = await embeddings_client.embed([page.text for page in pages])
    except EmbeddingsError as error:
        document_embedding.status = "failed"
        document_embedding.completed_at = datetime.now(UTC)
        document_embedding.error_message = str(error)
        await session.commit()
        raise

    page_embeddings = [
        PageEmbedding(
            document_embedding_id=document_embedding.id,
            document_page_id=page.id,
            embedding=vector,
        )
        for page, vector in zip(pages, vectors, strict=True)
    ]
    session.add_all(page_embeddings)
    document_embedding.status = "succeeded"
    document_embedding.page_count = len(page_embeddings)
    document_embedding.completed_at = datetime.now(UTC)
    document_embedding.error_message = None
    await session.commit()

    return EmbeddingPersistenceResult(
        document_embedding_id=document_embedding.id,
        page_count=document_embedding.page_count,
        created=created,
    )
