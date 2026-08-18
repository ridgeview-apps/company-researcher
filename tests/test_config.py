import os
from unittest.mock import patch

from company_researcher.config import Settings


def test_settings_have_local_defaults() -> None:
    # Temporarily isolate defaults from variables injected by shells or IDEs.
    with patch.dict(os.environ, clear=True):
        settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.app_name == "Company Researcher"
    assert settings.app_environment == "local"
    assert settings.companies_house_api_key is None
    assert settings.database_url.startswith("postgresql+asyncpg://")
