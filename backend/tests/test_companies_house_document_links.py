import pytest

from company_researcher.companies_house.document_links import (
    parse_document_metadata_url,
)
from company_researcher.companies_house.exceptions import CompaniesHouseResponseError


def test_parse_document_metadata_url_returns_document_id() -> None:
    document_id = parse_document_metadata_url(
        "https://document-api.company-information.service.gov.uk/"
        "document/xZTDnsspCOpP_kzIgzKnwRzrBUwubHvRAzk3K4cbWg4"
    )

    assert document_id == "xZTDnsspCOpP_kzIgzKnwRzrBUwubHvRAzk3K4cbWg4"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.test/document/document-123",
        "http://document-api.company-information.service.gov.uk/document/document-123",
        "https://document-api.company-information.service.gov.uk/other/document-123",
        "https://document-api.company-information.service.gov.uk/document/",
        "https://document-api.company-information.service.gov.uk/document/a/b",
        "https://document-api.company-information.service.gov.uk/document/id?format=pdf",
        "https://document-api.company-information.service.gov.uk:invalid/document/id",
        "https://document-api.company-information.service.gov.uk/document/id%2Fother",
    ],
)
def test_parse_document_metadata_url_rejects_unexpected_url(url: str) -> None:
    with pytest.raises(CompaniesHouseResponseError):
        parse_document_metadata_url(url)
