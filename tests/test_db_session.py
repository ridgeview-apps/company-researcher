import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.config import Settings
from company_researcher.db.session import (
    create_database_engine,
    create_session_factory,
)


@pytest.mark.asyncio
async def test_create_database_engine_uses_asyncpg() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    engine = create_database_engine(settings)

    try:
        assert engine.url.drivername == "postgresql+asyncpg"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sessions_retain_values_after_commit() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)

    try:
        async with session_factory() as session:
            assert isinstance(session, AsyncSession)
            assert session.sync_session.expire_on_commit is False
    finally:
        await engine.dispose()
