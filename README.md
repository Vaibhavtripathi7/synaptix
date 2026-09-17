# Synaptix

Production RAG over PDFs. Documents are parsed into typed, bbox-anchored blocks — text, tables,
figures, and **scanned or handwritten pages**, which a vision model transcribes to Markdown and
LaTeX — chunked along their heading structure, and indexed for hybrid search, with every chunk
traceable to the exact blocks and bounding boxes it came from.

## Architecture

```
client  ──►  FastAPI (app/api)  ──►  Postgres 17 + pgvector  ◄──  arq worker (app/workers)
              │ documents                         ▲                   └─ ingest: parse → OCR/enrich → chunk → embed
              └── /healthz /readyz
```

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI, async SQLAlchemy 2.0 | typed schemas, one DB session per request |
| Jobs | arq on Redis | async-native, tiny footprint, timeouts built in |
| Store | Postgres 17 + pgvector (HNSW) + tsvector (GIN) | vectors, FTS, metadata and transactions in one place |
| Parse | PyMuPDF + `pymupdf-layout` | block classes (heading/text/table/figure) with bboxes, tables as Markdown |
| LLM | Gemini via `google-genai` (Vertex or API key) | vision for figures/OCR, structured JSON output |
| Embed | `gemini-embedding-001` @1024-d, or local `bge-m3` | Matryoshka truncation lets both share one column |
| Rerank | listwise Gemini, or local `bge-reranker-v2-m3` | precision on top of RRF without a hosted reranking service |

## Run

Requirements: Docker, [`uv`](https://docs.astral.sh/uv/).

```sh
cp .env.example .env            # add SYNAPTIX_GOOGLE_API_KEY (or Vertex project)
uv sync                          # project-local Python 3.12 + deps in ./.venv
make infra                       # Postgres :5433 and Redis :6380 (non-default ports, no collisions)
make migrate
make api                         # http://localhost:8000/docs (API)
make worker                      # second terminal
```

### API

```sh
curl -F file=@paper.pdf localhost:8000/api/v1/documents   # 202 → queued … ready
curl localhost:8000/api/v1/documents/<id>                    # status, page count, title
```

`GET /documents/{id}/chunks/{chunk_id}` returns the page elements (with bboxes) a chunk was built
from, so a client can highlight the exact blocks on the page.

### Resilience

`app/llm/client.py` retries transport errors in place and fails over to the next model in
`SYNAPTIX_LLM_FALLBACK_MODELS` on quota 429s or 503 overloads. Enrichment degrades per element
(a figure that could not be captioned keeps its caption text) instead of failing the document.

## Develop

```sh
make test                 # unit tests + DB-backed retrieval tests (skipped if Postgres is down)
make lint                 # ruff
uv sync --extra local     # adds sentence-transformers for local embeddings / reranking
```

Configuration is entirely via `SYNAPTIX_*` environment variables (see
`app/core/settings.py`). The original prototype lives in `legacy/` for reference only.
