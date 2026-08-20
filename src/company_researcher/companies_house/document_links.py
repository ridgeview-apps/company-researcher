import re
from urllib.parse import urlsplit

from company_researcher.companies_house.exceptions import CompaniesHouseResponseError

DOCUMENT_API_HOST = "document-api.company-information.service.gov.uk"
DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_document_metadata_url(url: str) -> str:
    """Validate a Companies House document metadata URL and return its ID."""
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise CompaniesHouseResponseError(
            "Companies House returned an invalid document metadata URL"
        ) from error

    path_parts = parsed.path.strip("/").split("/")
    if (
        parsed.scheme != "https"
        or parsed.hostname != DOCUMENT_API_HOST
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or len(path_parts) != 2
        or path_parts[0] != "document"
        or not DOCUMENT_ID_PATTERN.fullmatch(path_parts[1])
    ):
        raise CompaniesHouseResponseError(
            "Companies House returned an invalid document metadata URL"
        )

    return path_parts[1]
