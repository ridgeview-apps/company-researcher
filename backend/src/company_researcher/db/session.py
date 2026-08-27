from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from company_researcher.config import Settings


def create_database_engine(settings: Settings) -> AsyncEngine:
    """Create the application's async SQLAlchemy engine."""
    return create_async_engine(settings.database_url, pool_pre_ping=True)


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create sessions that retain loaded values after a commit."""
    return async_sessionmaker(engine, expire_on_commit=False)
