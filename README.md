# Synaptix

Production RAG over PDFs. Documents are parsed into typed, bbox-anchored blocks — text, tables,
figures, and scanned pages — chunked along their heading structure, and indexed for hybrid
search, with every chunk traceable to the exact blocks and bounding boxes it came from.

## Architecture

```
PDF ──► parse ──► typed blocks ──► chunk ──► embed ──► Postgres 17 + pgvector
```

| Layer | Choice | Why |
|---|---|---|
| Store | Postgres 17 + pgvector (HNSW) + tsvector (GIN) | vectors, FTS, metadata and transactions in one place |

## Run

Requirements: Docker, [`uv`](https://docs.astral.sh/uv/).

```sh
cp .env.example .env
uv sync                          # project-local Python 3.12 + deps in ./.venv
make infra                       # Postgres :5433 (a non-default port: no collisions)
make migrate
```

## Develop

```sh
make test                 # unit tests
make lint                 # ruff
```

Configuration is entirely via `SYNAPTIX_*` environment variables (see
`app/core/settings.py`). The original prototype lives in `legacy/` for reference only.
