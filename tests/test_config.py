from company_researcher.config import Settings


def test_settings_have_local_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "Company Researcher"
    assert settings.app_environment == "local"
    assert settings.companies_house_api_key is None
    assert settings.database_url.startswith("postgresql+asyncpg://")
