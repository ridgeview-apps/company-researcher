import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from pathlib import Path

import httpx2
import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.artifact_store import LocalArtifactStore
from company_researcher.companies_house import CompaniesHouseDocumentClient
from company_researcher.config import Settings
from company_researcher.db.models import Company, Filing, FilingDocument
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.document_ingestion import (
    DocumentIngestionError,
    ingest_filing_document,
)

TEST_COMPANY_NUMBER = "TE000003"
TEST_DOCUMENT_ID = "document-ingestion-123"


def _list_directory(path: Path) -> list[Path]:
    return list(path.iterdir())


def _create_client(
    content: bytes,
    *,
    declared_length: int | None = None,
    include_pdf: bool = True,
) -> CompaniesHouseDocumentClient:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/content"):
            return httpx2.Response(
                200,
                headers={"Content-Type": "application/pdf"},
                content=content,
                request=request,
            )
        return httpx2.Response(
            200,
            json={
                "created_at": "2026-05-10T22:33:40Z",
                "etag": "document-etag",
                "pages": 10,
                "resources": (
                    {
                        "application/pdf": {
                            "content_length": (
                                declared_length
                                if declared_length is not None
                                else len(content)
                            )
                        }
                    }
                    if include_pdf
                    else {}
                ),
            },
            request=request,
        )

    return CompaniesHouseDocumentClient(
        api_key="test-api-key",
        base_url="https://document.example.test",
        transport=httpx2.MockTransport(handler),
    )


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_database_engine(Settings(_env_file=None))  # type: ignore[call-arg]
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as db_session:
            yield db_session
    finally:
        async with session_factory() as cleanup_session:
            filing_ids = list(
                await cleanup_session.scalars(
                    select(Filing.id).where(
                        Filing.company_number == TEST_COMPANY_NUMBER
                    )
                )
            )
            if filing_ids:
                await cleanup_session.execute(
                    delete(FilingDocument).where(
                        FilingDocument.filing_id.in_(filing_ids)
                    )
                )
            await cleanup_session.execute(
                delete(Filing).where(Filing.company_number == TEST_COMPANY_NUMBER)
            )
            await cleanup_session.execute(
                delete(Company).where(Company.company_number == TEST_COMPANY_NUMBER)
            )
            await cleanup_session.commit()
        await engine.dispose()


async def _create_filing(session: AsyncSession) -> Filing:
    retrieved_at = datetime.now(UTC)
    company = Company(
        company_number=TEST_COMPANY_NUMBER,
        company_name="DOCUMENT INGESTION TEST LIMITED",
        type="ltd",
        sic_codes=[],
        raw_profile={},
        retrieved_at=retrieved_at,
    )
    filing = Filing(
        company_number=TEST_COMPANY_NUMBER,
        transaction_id="document-ingestion-transaction",
        category="accounts",
        type="AA",
        description="accounts",
        date=date(2026, 1, 1),
        source_document_id=TEST_DOCUMENT_ID,
        document_metadata_url=(
            "https://document-api.company-information.service.gov.uk/"
            f"document/{TEST_DOCUMENT_ID}"
        ),
        raw_filing={},
        retrieved_at=retrieved_at,
    )
    session.add_all([company, filing])
    await session.commit()
    return filing


@pytest.mark.asyncio
async def test_ingest_filing_document_persists_artifact_and_provenance(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    filing = await _create_filing(session)
    store = LocalArtifactStore(tmp_path)
    async with _create_client(b"%PDF-1.7 first version") as client:
        result = await ingest_filing_document(session, client, store, filing)

    persisted = await session.get(FilingDocument, result.filing_document_id)

    assert result.created is True
    assert persisted is not None
    assert persisted.filing_id == filing.id
    assert persisted.source_document_id == TEST_DOCUMENT_ID
    assert persisted.sha256 == result.sha256
    assert (tmp_path / result.storage_key).read_bytes() == b"%PDF-1.7 first version"


@pytest.mark.asyncio
async def test_ingest_filing_document_refreshes_same_content_version(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    filing = await _create_filing(session)
    store = LocalArtifactStore(tmp_path)
    async with _create_client(b"%PDF-1.7 same version") as client:
        first = await ingest_filing_document(session, client, store, filing)
        second = await ingest_filing_document(session, client, store, filing)

    count = await session.scalar(
        select(func.count())
        .select_from(FilingDocument)
        .where(FilingDocument.filing_id == filing.id)
    )

    assert first.created is True
    assert second.created is False
    assert second.filing_document_id == first.filing_document_id
    assert count == 1


@pytest.mark.asyncio
async def test_ingest_filing_document_preserves_changed_content_version(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    filing = await _create_filing(session)
    store = LocalArtifactStore(tmp_path)
    async with _create_client(b"%PDF-1.7 first version") as client:
        first = await ingest_filing_document(session, client, store, filing)
    async with _create_client(b"%PDF-1.7 changed version") as client:
        second = await ingest_filing_document(session, client, store, filing)

    count = await session.scalar(
        select(func.count())
        .select_from(FilingDocument)
        .where(FilingDocument.filing_id == filing.id)
    )

    assert first.created is True
    assert second.created is True
    assert second.filing_document_id != first.filing_document_id
    assert count == 2


@pytest.mark.asyncio
async def test_ingest_filing_document_rejects_missing_document_reference(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    filing = await _create_filing(session)
    filing.source_document_id = None
    async with _create_client(b"%PDF-1.7 content") as client:
        with pytest.raises(DocumentIngestionError, match="does not reference"):
            await ingest_filing_document(
                session,
                client,
                LocalArtifactStore(tmp_path),
                filing,
            )


@pytest.mark.asyncio
async def test_ingest_filing_document_requires_pdf_resource(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    filing = await _create_filing(session)
    async with _create_client(b"content", include_pdf=False) as client:
        with pytest.raises(DocumentIngestionError, match="not available as PDF"):
            await ingest_filing_document(
                session,
                client,
                LocalArtifactStore(tmp_path),
                filing,
            )


@pytest.mark.asyncio
async def test_ingest_filing_document_rejects_content_length_mismatch(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    filing = await _create_filing(session)
    async with _create_client(b"content", declared_length=999) as client:
        with pytest.raises(DocumentIngestionError, match="length"):
            await ingest_filing_document(
                session,
                client,
                LocalArtifactStore(tmp_path),
                filing,
            )

    entries = await asyncio.to_thread(_list_directory, tmp_path)
    assert entries == []
