FROM ghcr.io/astral-sh/uv:0.11.6 AS uv

FROM python:3.13-slim-bookworm

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --locked --no-dev

COPY alembic.ini ./
COPY migrations ./migrations

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "company_researcher.main:app", "--host", "0.0.0.0", "--port", "8000"]
