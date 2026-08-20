from collections.abc import Callable

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
from company_researcher.document_ingestion import DocumentIngestionError


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
