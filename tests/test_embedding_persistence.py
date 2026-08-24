from collections.abc import AsyncIterator, Sequence
from datetime import UTC, date, datetime

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.config import Settings
from company_researcher.db.models import (
    EMBEDDING_DIMENSIONS,
    Company,
    DocumentEmbedding,
    DocumentExtraction,
    DocumentPage,
    Filing,
    FilingDocument,
    PageEmbedding,
)
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.embedding_persistence import embed_document_extraction
from company_researcher.embeddings_client import EmbeddingsError

TEST_COMPANY_NUMBER = "TE000009"


def _fake_vector(leading_value: float) -> list[float]:
    """A vector matching the real pgvector column's fixed width."""
    return [leading_value] + [0.0] * (EMBEDDING_DIMENSIONS - 1)


class FakeEmbeddingsProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.call_count = 0
        self.fail = fail
        self.requested_texts: list[Sequence[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.call_count += 1
        self.requested_texts.append(texts)
        if self.fail:
            raise EmbeddingsError("Controlled embeddings failure")
        return [_fake_vector(float(index)) for index in range(len(texts))]


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_database_engine(Settings(_env_file=None))  # type: ignore[call-arg]
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as db_session:
            yield db_session
    finally:
        async with session_factory() as cleanup_session:
            await cleanup_session.execute(
                delete(Filing).where(Filing.company_number == TEST_COMPANY_NUMBER)
            )
            await cleanup_session.execute(
                delete(Company).where(Company.company_number == TEST_COMPANY_NUMBER)
            )
            await cleanup_session.commit()
        await engine.dispose()


async def _create_extraction(
    session: AsyncSession, texts: list[str]
) -> DocumentExtraction:
    now = datetime.now(UTC)
    company = Company(
        company_number=TEST_COMPANY_NUMBER,
        company_name="EMBEDDING PERSISTENCE TEST LIMITED",
        type="ltd",
        sic_codes=[],
        raw_profile={},
        retrieved_at=now,
    )
    filing = Filing(
        company_number=TEST_COMPANY_NUMBER,
        transaction_id="embedding-persistence-transaction",
        category="accounts",
        type="AA",
        description="accounts",
        date=date(2026, 1, 1),
        raw_filing={},
        retrieved_at=now,
    )
    session.add_all([company, filing])
    await session.flush()
    document = FilingDocument(
        filing_id=filing.id,
        source_document_id="embedding-persistence-document",
        media_type="application/pdf",
        content_length=1234,
        sha256="d" * 64,
        storage_key="sha256/test.pdf",
        source_created_at=now,
        raw_metadata={},
        first_retrieved_at=now,
        last_retrieved_at=now,
    )
    session.add(document)
    await session.flush()
    extraction = DocumentExtraction(
        filing_document_id=document.id,
        status="succeeded",
        extractor="tesseract",
        extractor_version="5.5.3",
        renderer="pypdfium2",
        renderer_version="5.13.0",
        language="eng",
        render_dpi=300,
        page_segmentation_mode=3,
        started_at=now,
        completed_at=now,
        page_count=len(texts),
        total_character_count=sum(len(text) for text in texts),
    )
    session.add(extraction)
    await session.flush()
    session.add_all(
        [
            DocumentPage(
                document_extraction_id=extraction.id,
                page_number=page_number,
                text=text,
                character_count=len(text),
            )
            for page_number, text in enumerate(texts, start=1)
        ]
    )
    await session.commit()
    return extraction


@pytest.mark.asyncio
async def test_embed_document_extraction_persists_vectors_in_page_order(
    session: AsyncSession,
) -> None:
    extraction = await _create_extraction(session, ["First page", "Second page"])
    provider = FakeEmbeddingsProvider()

    result = await embed_document_extraction(
        session,
        provider,
        extraction,
        provider="fake",
        model="fake-model",
        dimensions=EMBEDDING_DIMENSIONS,
    )
    document_embedding = await session.get(
        DocumentEmbedding, result.document_embedding_id
    )
    page_embeddings = list(
        await session.scalars(
            select(PageEmbedding)
            .join(DocumentPage, PageEmbedding.document_page_id == DocumentPage.id)
            .where(PageEmbedding.document_embedding_id == result.document_embedding_id)
            .order_by(DocumentPage.page_number)
        )
    )

    assert result.created is True
    assert result.page_count == 2
    assert document_embedding is not None
    assert document_embedding.status == "succeeded"
    assert [list(page.embedding) for page in page_embeddings] == [
        _fake_vector(0.0),
        _fake_vector(1.0),
    ]


@pytest.mark.asyncio
async def test_embed_document_extraction_skips_succeeded_configuration(
    session: AsyncSession,
) -> None:
    extraction = await _create_extraction(session, ["First page", "Second page"])
    provider = FakeEmbeddingsProvider()

    first = await embed_document_extraction(
        session,
        provider,
        extraction,
        provider="fake",
        model="fake-model",
        dimensions=EMBEDDING_DIMENSIONS,
    )
    second = await embed_document_extraction(
        session,
        provider,
        extraction,
        provider="fake",
        model="fake-model",
        dimensions=EMBEDDING_DIMENSIONS,
    )
    embedding_count = await session.scalar(
        select(func.count())
        .select_from(PageEmbedding)
        .where(PageEmbedding.document_embedding_id == first.document_embedding_id)
    )

    assert first.created is True
    assert second.created is False
    assert provider.call_count == 1
    assert embedding_count == 2


@pytest.mark.asyncio
async def test_embed_document_extraction_records_controlled_failure(
    session: AsyncSession,
) -> None:
    extraction = await _create_extraction(session, ["First page"])

    with pytest.raises(EmbeddingsError, match="Controlled embeddings failure"):
        await embed_document_extraction(
            session,
            FakeEmbeddingsProvider(fail=True),
            extraction,
            provider="fake",
            model="fake-model",
            dimensions=EMBEDDING_DIMENSIONS,
        )

    document_embedding = await session.scalar(
        select(DocumentEmbedding).where(
            DocumentEmbedding.document_extraction_id == extraction.id
        )
    )
    assert document_embedding is not None
    assert document_embedding.status == "failed"
    assert document_embedding.error_message == "Controlled embeddings failure"
    assert document_embedding.completed_at is not None
