import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from company_researcher.config import Settings
from company_researcher.main import create_app


def test_health_returns_ok() -> None:
    # Keep this unit test independent of developer and CI configuration.
    with patch.dict(os.environ, clear=True):
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
