# Synaptix

Production RAG over PDFs. Documents are parsed into typed, bbox-anchored blocks — text, tables,
figures, and **scanned or handwritten pages**, which a vision model transcribes to Markdown and
LaTeX — chunked along their heading structure, and indexed for hybrid search. Every answer cites
the passages it came from, and clicking a citation highlights that exact block on the rendered
page. On top of retrieval sits an analysis DAG (competitor profile, SWOT and likely strategy,
combined into a key analysis) built from targeted retrieval rather than a truncated document
dump, so every point it makes is evidence-linked.

## Architecture

```
browser (web/)  ──►  FastAPI (app/api)  ──►  Postgres 17 + pgvector  ◄──  arq worker (app/workers)
   │ pdf.js page      │ documents · query · analyses      ▲                   ├─ ingest: parse → OCR/enrich → chunk → embed
   │ + highlights      │ SSE answers · graph · summary     │                   └─ analyze: profile · SWOT · strategy → key analysis
   └──────────────────┴── bearer-key auth · request ids · access log · /healthz /readyz
```

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI, async SQLAlchemy 2.0 | streaming answers, typed schemas, one DB session per request |
| Jobs | arq on Redis | async-native, tiny footprint, timeouts built in |
| Store | Postgres 17 + pgvector (HNSW) + tsvector (GIN) | vectors, FTS, metadata and transactions in one place |
| Parse | PyMuPDF + `pymupdf-layout` | block classes (heading/text/table/figure) with bboxes, tables as Markdown |
| LLM | Gemini via `google-genai` (Vertex or API key) | vision for figures/OCR, structured JSON output, streaming |
| Embed | `gemini-embedding-001` @1024-d, or local `bge-m3` | Matryoshka truncation lets both share one column |
| Rerank | listwise Gemini, or local `bge-reranker-v2-m3` | precision on top of RRF without a hosted reranking service |
| UI | static HTML/JS, pdf.js, vis-network | no build step, no framework; the page is the proof |

## Run

Requirements: Docker, [`uv`](https://docs.astral.sh/uv/).

```sh
cp .env.example .env            # add SYNAPTIX_GOOGLE_API_KEY (or Vertex project)
uv sync                          # project-local Python 3.12 + deps in ./.venv
make infra                       # Postgres :5433 and Redis :6380 (non-default ports, no collisions)
make migrate
make api                         # http://localhost:8000  (UI)   http://localhost:8000/docs (API)
make worker                      # second terminal
```

### API in four calls

```sh
curl -F file=@paper.pdf localhost:8000/api/v1/documents   # 202 → queued … ready
curl localhost:8000/api/v1/query -H 'content-type: application/json' \
  -d '{"question": "What does it conclude about genetic algorithms?", "document_ids": ["<id>"]}'
  # {"answer": "... [1][3]", "passages": [{"n": 1, "page_start": 13, "heading_path": [...], "chunk_ids": [...]}]}
  # "stream": true → Server-Sent Events: passages, token…, done
curl -X POST localhost:8000/api/v1/analyses -H 'content-type: application/json' \
  -d '{"kind": "competitor", "document_ids": ["<id>"]}'                  # or "comparison" with two ids
curl -X POST localhost:8000/api/v1/documents/<id>/graph   # entity graph, cached
```

`GET /documents/{id}/chunks/{chunk_id}` returns the page elements (with bboxes) behind a citation;
that is what the UI highlights. Set `SYNAPTIX_API_KEYS=key1,key2` to additionally gate the whole
instance behind `Authorization: Bearer` (the UI asks for it once and remembers it).

### Retrieval, in one paragraph

A question is paraphrased into two extra queries (grounded in the corpus titles), all three are
embedded, and each runs a dense (pgvector cosine, HNSW) and a sparse (full-text, GIN) search.
The six rankings are fused with reciprocal rank fusion, the top 40 are graded listwise by the
LLM (or a local cross-encoder), and the best 8 become the context: a chunk is swapped for its
whole section when that section is short or several hits share it. The answer cites `[n]`
markers that map back to page ranges, heading paths and element kinds.

### Analysis agents

Each agent is a set of retrieval *facets* (e.g. "revenue, sales channels and financial
results") whose hits are unioned into one budgeted context, plus a structured prompt; the output
is a Pydantic model where every item lists the passage numbers it rests on. A competitor analysis
runs profile, SWOT and strategy, then a key analysis that reads all three. A comparison
retrieves the same facets from two documents. A stage that finds nothing returns an empty list
instead of inventing content. Runs are rows in `analysis_runs` storing the passages each stage
used, so results stay auditable.

### Resilience

`app/llm/client.py` retries transport errors in place and fails over to the next model in
`SYNAPTIX_LLM_FALLBACK_MODELS` on quota 429s or 503 overloads. Enrichment degrades per element
(a figure that could not be captioned keeps its caption text) instead of failing the document. Each request carries an
`x-request-id` that appears on every log line and is echoed to the client.

## Develop

```sh
make test                 # unit tests + DB-backed retrieval tests (skipped if Postgres is down)
make lint                 # ruff
uv sync --extra local     # adds sentence-transformers for local embeddings / reranking
```

Configuration is entirely via `SYNAPTIX_*` environment variables (see
`app/core/settings.py`). The original prototype lives in `legacy/` for reference only.
