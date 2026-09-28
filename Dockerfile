FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# tiktoken fetches its BPE file from the internet on first use; without this, every container
# would need egress to a third-party host just to start, and fail behind an egress allowlist.
# Fetched once here, where the build has network, and read from the image at runtime.
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken
RUN .venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
COPY app ./app
COPY web ./web
COPY alembic.ini ./
RUN uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm
WORKDIR /app
RUN useradd -m -u 1000 synaptix && mkdir -p data/storage && chown -R synaptix data
COPY --from=builder --chown=synaptix /app /app
USER synaptix
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 TIKTOKEN_CACHE_DIR=/app/.tiktoken
EXPOSE 8000
# Our middleware writes a structured access line per request; uvicorn's would duplicate it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
