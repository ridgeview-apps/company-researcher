from collections.abc import AsyncIterator
from datetime import UTC, date, datetime

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
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
from company_researcher.vector_search import search_pages_by_embedding

TEST_COMPANY_NUMBER = "TE000011"
OTHER_COMPANY_NUMBER = "TE000012"


def _axis_vector(index: int, value: float = 1.0) -> list[float]:
    """A vector with `value` at `index` and zero elsewhere, for clean cosine geometry."""
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[index] = value
    return vector


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_database_engine(Settings(_env_file=None))  # type: ignore[call-arg]
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as db_session:
            yield db_session
    finally:
        async with session_factory() as cleanup_session:
            for company_number in (TEST_COMPANY_NUMBER, OTHER_COMPANY_NUMBER):
                await cleanup_session.execute(
                    delete(Filing).where(Filing.company_number == company_number)
                )
                await cleanup_session.execute(
                    delete(Company).where(Company.company_number == company_number)
                )
            await cleanup_session.commit()
        await engine.dispose()


@pytest_asyncio.fixture
async def company(session: AsyncSession) -> Company:
    company = Company(
        company_number=TEST_COMPANY_NUMBER,
        company_name="VECTOR SEARCH TEST LIMITED",
        type="ltd",
        sic_codes=[],
        raw_profile={},
        retrieved_at=datetime.now(UTC),
    )
    session.add(company)
    await session.commit()
    return company


async def _create_page_embeddings(
    session: AsyncSession,
    vectors: list[list[float]],
    *,
    provider: str = "fake",
    model: str = "fake-model",
    dimensions: int = EMBEDDING_DIMENSIONS,
    company_number: str = TEST_COMPANY_NUMBER,
) -> DocumentExtraction:
    now = datetime.now(UTC)
    company_statement = select(Company).where(Company.company_number == company_number)
    company = (await session.execute(company_statement)).scalar_one_or_none()
    if company is None:
        company = Company(
            company_number=company_number,
            company_name="VECTOR SEARCH TEST LIMITED",
            type="ltd",
            sic_codes=[],
            raw_profile={},
            retrieved_at=now,
        )
        session.add(company)
    document_key = f"{model}-{company_number}"
    filing = Filing(
        company_number=company_number,
        transaction_id=f"vector-search-transaction-{document_key}",
        category="accounts",
        type="AA",
        description="accounts",
        date=date(2026, 1, 1),
        raw_filing={},
        retrieved_at=now,
    )
    session.add(filing)
    await session.flush()
    document = FilingDocument(
        filing_id=filing.id,
        source_document_id=f"vector-search-document-{document_key}",
        media_type="application/pdf",
        content_length=1234,
        sha256=f"{abs(hash(document_key)):064x}"[:64],
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
    )
    session.add(extraction)
    await session.flush()
    pages = [
        DocumentPage(
            document_extraction_id=extraction.id,
            page_number=page_number,
            text=f"page {page_number}",
            character_count=6,
        )
        for page_number in range(1, len(vectors) + 1)
    ]
    session.add_all(pages)
    await session.flush()
    document_embedding = DocumentEmbedding(
        document_extraction_id=extraction.id,
        status="succeeded",
        provider=provider,
        model=model,
        dimensions=dimensions,
        page_count=len(vectors),
        started_at=now,
        completed_at=now,
    )
    session.add(document_embedding)
    await session.flush()
    session.add_all(
        [
            PageEmbedding(
                document_embedding_id=document_embedding.id,
                document_page_id=page.id,
                embedding=vector,
            )
            for page, vector in zip(pages, vectors, strict=True)
        ]
    )
    await session.commit()
    return extraction


@pytest.mark.asyncio
async def test_search_pages_by_embedding_ranks_by_cosine_distance(
    session: AsyncSession, company: Company
) -> None:
    await _create_page_embeddings(
        session,
        [
            _axis_vector(0, -1.0),  # page 1: opposite direction, farthest
            _axis_vector(0, 1.0),  # page 2: identical direction, closest
            _axis_vector(1, 1.0),  # page 3: orthogonal, in the middle
        ],
    )

    matches = await search_pages_by_embedding(
        session,
        _axis_vector(0, 1.0),
        provider="fake",
        model="fake-model",
        dimensions=EMBEDDING_DIMENSIONS,
        limit=10,
    )

    assert [match.page_number for match in matches] == [2, 3, 1]
    assert matches[0].distance < matches[1].distance < matches[2].distance


@pytest.mark.asyncio
async def test_search_pages_by_embedding_filters_by_configuration(
    session: AsyncSession, company: Company
) -> None:
    await _create_page_embeddings(session, [_axis_vector(0, 1.0)], model="model-a")
    await _create_page_embeddings(session, [_axis_vector(0, 1.0)], model="model-b")

    matches = await search_pages_by_embedding(
        session,
        _axis_vector(0, 1.0),
        provider="fake",
        model="model-a",
        dimensions=EMBEDDING_DIMENSIONS,
        limit=10,
    )

    assert len(matches) == 1


@pytest.mark.asyncio
async def test_search_pages_by_embedding_restricts_to_given_company_number(
    session: AsyncSession, company: Company
) -> None:
    allowed = await _create_page_embeddings(
        session,
        [_axis_vector(0, 1.0)],
        company_number=TEST_COMPANY_NUMBER,
    )
    excluded = await _create_page_embeddings(
        session,
        [_axis_vector(0, 1.0)],
        company_number=OTHER_COMPANY_NUMBER,
    )

    matches = await search_pages_by_embedding(
        session,
        _axis_vector(0, 1.0),
        provider="fake",
        model="fake-model",
        dimensions=EMBEDDING_DIMENSIONS,
        limit=10,
        company_number=TEST_COMPANY_NUMBER,
    )
    matched_extraction_ids = {match.document_extraction_id for match in matches}

    assert allowed.id in matched_extraction_ids
    assert excluded.id not in matched_extraction_ids


@pytest.mark.asyncio
async def test_search_pages_by_embedding_respects_limit(
    session: AsyncSession, company: Company
) -> None:
    await _create_page_embeddings(
        session,
        [_axis_vector(page_number) for page_number in range(5)],
    )

    matches = await search_pages_by_embedding(
        session,
        _axis_vector(0, 1.0),
        provider="fake",
        model="fake-model",
        dimensions=EMBEDDING_DIMENSIONS,
        limit=2,
    )

    assert len(matches) == 2
