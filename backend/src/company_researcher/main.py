from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from company_researcher.api.health import router as health_router
from company_researcher.config import Settings
from company_researcher.db.session import create_database_engine


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure a FastAPI application instance."""
    app_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_database_engine(app_settings)
        app.state.settings = app_settings
        app.state.database_engine = engine
        try:
            yield
        finally:
            await engine.dispose()

    application = FastAPI(
        title=app_settings.app_name,
        lifespan=lifespan,
    )
    application.include_router(health_router)
    return application


app = create_app()
