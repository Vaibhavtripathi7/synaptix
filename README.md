# Synaptix

Production RAG over PDFs. Documents are parsed into typed, bbox-anchored blocks — text, tables,
figures, and **scanned or handwritten pages**, which a vision model transcribes to Markdown and
LaTeX — chunked along their heading structure, and indexed for hybrid search, with every chunk
traceable to the exact blocks and bounding boxes it came from.

## Architecture

```
PDF ──► parse (layout model) ──► enrich (vision: figures, tables, scans) ──► chunk ──► embed
                                                                                     │
                                                         Postgres 17 + pgvector ◄────┘
```

| Layer | Choice | Why |
|---|---|---|
| Store | Postgres 17 + pgvector (HNSW) + tsvector (GIN) | vectors, FTS, metadata and transactions in one place |
| Parse | PyMuPDF + `pymupdf-layout` | block classes (heading/text/table/figure) with bboxes, tables as Markdown |
| LLM | Gemini via `google-genai` (Vertex or API key) | vision for figures/OCR, structured JSON output |
| Embed | `gemini-embedding-001` @1024-d, or local `bge-m3` | Matryoshka truncation lets both share one column |

## Run

Requirements: Docker, [`uv`](https://docs.astral.sh/uv/).

```sh
cp .env.example .env            # add SYNAPTIX_GOOGLE_API_KEY (or Vertex project)
uv sync                          # project-local Python 3.12 + deps in ./.venv
make infra                       # Postgres :5433 (a non-default port: no collisions)
make migrate
```

### Resilience

`app/llm/client.py` retries transport errors in place and fails over to the next model in
`SYNAPTIX_LLM_FALLBACK_MODELS` on quota 429s or 503 overloads. Enrichment degrades per element
(a figure that could not be captioned keeps its caption text) instead of failing the document.

## Develop

```sh
make test                 # unit tests
make lint                 # ruff
uv sync --extra local     # adds sentence-transformers for local embeddings
```

Configuration is entirely via `SYNAPTIX_*` environment variables (see
`app/core/settings.py`). The original prototype lives in `legacy/` for reference only.
