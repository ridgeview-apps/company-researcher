from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from company_researcher import cli
from company_researcher.companies_house.exceptions import (
    CompaniesHouseAuthenticationError,
    CompaniesHouseConfigurationError,
    CompaniesHouseConnectionError,
    CompaniesHouseNotFoundError,
    CompaniesHouseRateLimitError,
    CompaniesHouseResponseError,
)
from company_researcher.db.models import FilingDocument
from company_researcher.document_ingestion import DocumentIngestionError
from company_researcher.extraction_persistence import ExtractionPersistenceResult
from company_researcher.pdf_extraction import PdfExtractionError


def test_main_prints_inspection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "run_inspection", lambda company_number: '{"ok": true}')

    exit_code = cli.main(["inspect", "00000006"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == '{"ok": true}\n'
    assert captured.err == ""


def test_main_prints_ingestion(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "run_ingestion", lambda company_number: "Ingested.")

    exit_code = cli.main(["ingest", "00000006"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "Ingested.\n"
    assert captured.err == ""


def test_main_prints_document_ingestion(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "run_document_ingestion",
        lambda company_number, transaction_id: "Created document.",
    )

    exit_code = cli.main(["ingest-document", "08130873", "filing-transaction-id"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "Created document.\n"
    assert captured.err == ""


def test_main_prints_document_extraction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "run_document_extraction",
        lambda filing_document_id: "Created extraction.",
    )

    exit_code = cli.main(["extract-document", "42"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "Created extraction.\n"
    assert captured.err == ""


def test_main_reports_document_extraction_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_extraction(filing_document_id: int) -> str:
        raise PdfExtractionError("safe OCR failure")

    monkeypatch.setattr(cli, "run_document_extraction", fail_extraction)

    exit_code = cli.main(["extract-document", "42"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "Document extraction error: safe OCR failure\n"


@pytest.mark.asyncio
async def test_extract_document_command_orchestrates_persisted_extraction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    filing_document = object()
    session = MagicMock()
    session.get = AsyncMock(return_value=filing_document)
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=False)
    session_factory = MagicMock(return_value=session_context)
    engine = MagicMock()
    engine.dispose = AsyncMock()
    extractor = object()
    extraction_result = ExtractionPersistenceResult(
        document_extraction_id=7,
        page_count=50,
        total_character_count=115_763,
        created=True,
    )
    persist = AsyncMock(return_value=extraction_result)

    monkeypatch.setattr(
        cli,
        "Settings",
        lambda: MagicMock(artifact_root=tmp_path),
    )
    monkeypatch.setattr(cli, "create_database_engine", lambda settings: engine)
    monkeypatch.setattr(
        cli,
        "create_session_factory",
        lambda configured_engine: session_factory,
    )
    monkeypatch.setattr(cli, "TesseractPdfExtractor", lambda: extractor)
    monkeypatch.setattr(cli, "extract_filing_document", persist)

    output = await cli.extract_document_command(42)

    assert output == "Created extraction 7: 50 page(s), 115763 character(s)."
    session.get.assert_awaited_once_with(FilingDocument, 42)
    persist.assert_awaited_once()
    assert persist.await_args is not None
    persisted_session, artifact_store, persisted_extractor, persisted_document = (
        persist.await_args.args
    )
    assert persisted_session is session
    assert artifact_store._root == tmp_path
    assert persisted_extractor is extractor
    assert persisted_document is filing_document
    engine.dispose.assert_awaited_once_with()


@pytest.mark.parametrize(
    ("error_factory", "expected_exit_code", "expected_message"),
    [
        (
            CompaniesHouseConfigurationError,
            2,
            "Configuration error",
        ),
        (
            CompaniesHouseAuthenticationError,
            3,
            "Authentication error",
        ),
        (
            CompaniesHouseNotFoundError,
            4,
            "Not found",
        ),
        (
            CompaniesHouseRateLimitError,
            5,
            "Rate limit error",
        ),
        (
            CompaniesHouseConnectionError,
            1,
            "Companies House error",
        ),
        (
            CompaniesHouseResponseError,
            1,
            "Companies House error",
        ),
        (
            DocumentIngestionError,
            1,
            "Document ingestion error",
        ),
    ],
)
def test_main_maps_expected_errors_to_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error_factory: Callable[[str], Exception],
    expected_exit_code: int,
    expected_message: str,
) -> None:
    def raise_error(company_number: str) -> str:
        raise error_factory("safe test message")

    monkeypatch.setattr(cli, "run_inspection", raise_error)

    exit_code = cli.main(["inspect", "00000006"])

    captured = capsys.readouterr()
    assert exit_code == expected_exit_code
    assert captured.out == ""
    assert expected_message in captured.err
    assert "safe test message" in captured.err
