from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a local .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Company Researcher"
    app_environment: Literal["local", "test", "production"] = "local"
    artifact_root: Path = Path("data/artifacts")

    database_url: str = (
        "postgresql+asyncpg://company_researcher:company_researcher@"
        "localhost:5432/company_researcher"
    )

    companies_house_api_key: SecretStr | None = None
    companies_house_base_url: AnyHttpUrl = AnyHttpUrl(
        "https://api.company-information.service.gov.uk"
    )
    companies_house_document_base_url: AnyHttpUrl = AnyHttpUrl(
        "https://document-api.company-information.service.gov.uk"
    )
