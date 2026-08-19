class CompaniesHouseError(Exception):
    """Base exception for Companies House integration failures."""


class CompaniesHouseConfigurationError(CompaniesHouseError):
    """Raised when Companies House configuration is incomplete."""


class CompaniesHouseConnectionError(CompaniesHouseError):
    """Raised when Companies House cannot be reached."""


class CompaniesHouseAuthenticationError(CompaniesHouseError):
    """Raised when Companies House rejects the API key."""


class CompaniesHouseNotFoundError(CompaniesHouseError):
    """Raised when a requested Companies House resource does not exist."""


class CompaniesHouseRateLimitError(CompaniesHouseError):
    """Raised when the Companies House request quota is exhausted."""


class CompaniesHouseResponseError(CompaniesHouseError):
    """Raised for an unexpected status or invalid response payload."""
