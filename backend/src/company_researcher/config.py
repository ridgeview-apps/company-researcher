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
    cors_allowed_origins: list[str] = ["http://localhost:5173"]

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

    openai_api_key: SecretStr | None = None
    openai_base_url: AnyHttpUrl = AnyHttpUrl("https://api.openai.com/v1")
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    langsmith_tracing_enabled: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "company-researcher"
    langsmith_endpoint: AnyHttpUrl = AnyHttpUrl("https://api.smith.langchain.com")
