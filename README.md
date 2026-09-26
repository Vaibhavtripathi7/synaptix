# Synaptix

A study and research workbench built on production RAG. Put your lecture notes, slides and
papers in one corpus and ask questions across them — including **scanned and handwritten pages**,
which are transcribed to Markdown and LaTeX by a vision model and indexed like everything else.
Every answer cites the page it came from, and clicking a citation highlights that exact block on
the rendered page. Study guides and paper comparisons are generated from targeted retrieval, not
from a truncated document dump, so every point they make is evidence-linked.

Most "chat with your PDF" tools read the text layer and return nothing useful for a scanned
notebook. This one answered a phase-relationship question from handwritten AC-circuit notes,
in rendered LaTeX, citing pages 1–3.

## Architecture

```
browser (web/)  ──►  FastAPI (app/api)  ──►  Postgres 17 + pgvector  ◄──  arq worker (app/workers)
   │ pdf.js page      │ documents · query · analyses      ▲                   ├─ ingest: parse → OCR/enrich → chunk → embed
   │ + highlights      │ SSE answers · graph · summary     │                   └─ analyze: outline → practice questions
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

Fully containerised: `docker compose --profile app up --build`.

### API in four calls

Every call carries a workspace token (your library) and, unless the server provides one, your own
Gemini key:

```sh
export H1="x-workspace: $(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export H2="x-gemini-key: $GEMINI_API_KEY"
curl -H "$H1" -H "$H2" -F file=@paper.pdf localhost:8000/api/v1/documents   # 202 → queued … ready
curl localhost:8000/api/v1/query -H "$H1" -H "$H2" -H 'content-type: application/json' \
  -d '{"question": "What does it conclude about genetic algorithms?", "document_ids": ["<id>"]}'
  # {"answer": "... [1][3]", "passages": [{"n": 1, "page_start": 13, "heading_path": [...], "chunk_ids": [...]}]}
  # "stream": true → Server-Sent Events: passages, token…, done
curl -X POST localhost:8000/api/v1/analyses -H "$H1" -H "$H2" -H 'content-type: application/json' \
  -d '{"kind": "study_guide", "document_ids": ["<id>"]}'                  # or "comparison" with two ids
curl -X POST -H "$H1" -H "$H2" localhost:8000/api/v1/documents/<id>/graph   # entity graph, cached
```

`GET /documents/{id}/chunks/{chunk_id}` returns the page elements (with bboxes) behind a citation;
that is what the UI highlights. Set `SYNAPTIX_API_KEYS=key1,key2` to additionally gate the whole
instance behind `Authorization: Bearer` (the UI asks for it once and remembers it).

### Hosting it for other people

Synaptix is built to be hosted for strangers at zero model cost to the operator:

- **Anonymous workspaces.** Each browser generates a 256-bit token on first visit. It *is* the
  library: every document, chunk, figure and analysis is scoped to it, through one ownership check
  that every route shares. Only its SHA-256 is stored, so a database copy grants no access. A
  document in another workspace is reported as missing, not forbidden. Upload dedupe is per
  workspace — a global content hash would hand one tenant's upload to another.
- **Bring your own key.** The user's Gemini key lives in their browser and travels as a header.
  For ingestion it rides the job payload as a `SecretStr`, whose repr is masked (arq logs job
  arguments). Redis runs without persistence and the worker keeps no job results, so the key is
  gone when the job ends; it is never written to Postgres. Each key gets its own client and
  concurrency limit, so one user's burst cannot starve another. Leave `SYNAPTIX_GOOGLE_API_KEY`
  unset on a public instance and requests without a key are refused before anything is queued.
- **Abuse bounds.** Anonymous means anyone can create a workspace, so upload size, page count and
  documents per workspace are capped (`SYNAPTIX_MAX_UPLOAD_MB`, `_MAX_PAGES`,
  `_MAX_DOCUMENTS_PER_WORKSPACE`).
- **Filtered vector search that doesn't starve.** HNSW returns the nearest vectors *globally* and
  applies the tenant filter afterwards, so a workspace that is a sliver of the table can get fewer
  results than it has. Searches run with pgvector's iterative index scan (`strict_order`). The test
  suite forces the index plan and shows the collapse — 4 of 10 results for a tenant holding 1,000
  matching chunks — then shows search returning all 10.
- **No locks held across model calls.** A read transaction left open during a multi-second LLM call
  keeps its table locks; a migration queued behind them then blocks every other query. Every
  retrieval path closes its transaction before calling a model, a test asserts it, and migrations
  run with `lock_timeout` so a blocked deploy fails fast instead of freezing the app.

Each guarantee has a test in `tests/test_workspaces.py`, and each test was checked by breaking the
guarantee it covers and confirming the test fails.

### Retrieval, in one paragraph

A question is paraphrased into two extra queries (grounded in the corpus titles), all three are
embedded, and each runs a dense (pgvector cosine, HNSW) and a sparse (full-text, GIN) search.
The six rankings are fused with reciprocal rank fusion, the top 40 are graded listwise by the
LLM (or a local cross-encoder), and the best 8 become the context: a chunk is swapped for its
whole section when that section is short or several hits share it. The answer cites `[n]`
markers that map back to page ranges, heading paths and element kinds.

### Study guides and comparisons

Each agent is a set of retrieval *facets* (e.g. "equations, formulas and mathematical
relationships") whose hits are unioned into one budgeted context, plus a structured prompt; the
output is a Pydantic model where every item lists the passage numbers it rests on. A study guide
runs in two stages — `outline → practice`, so the questions are set from concepts actually
extracted rather than from the raw text. A stage that finds nothing returns an empty list instead
of inventing content. Runs are rows in `analysis_runs` storing the passages each stage used, so
results stay auditable.

### Cost

Vision OCR is the dominant cost of ingesting scanned material, so scanned pages are transcribed
in batches of `SYNAPTIX_OCR_BATCH_PAGES` (default 6) images per request, each labelled so the
model's indexed replies cannot misalign. On a 33-page handwritten deck that is 7 requests instead
of 33 — the difference between "fits in the free tier" and "does not". Pages that already have a
text layer never reach a vision model at all.

### Resilience

`app/llm/client.py` retries transport errors in place, waits out per-minute rate limits using the
API's own `retryDelay`, and fails over to the next model in `SYNAPTIX_LLM_FALLBACK_MODELS` on
daily-quota 429s or 503 overloads. Enrichment degrades per element (a figure that could not be
captioned keeps its caption text) instead of failing the document. Each request carries an
`x-request-id` that appears on every log line and is echoed to the client.

## Operating it

**Observability.** Prometheus metrics on an internal port (API `:9100`, worker `:9101`), never
the public one. Request rate, latency and status by route *template* (bounded labels: a raw
path would create a time series per document id); model calls by model, operation and outcome
(`ok`, `failover`, `rate_limited`, `error`) with latency; tokens by the model that actually
served each call, after failover; job outcomes and durations; per-stage ingestion time; queue
depth (the worker autoscaling signal); answer-cache hits, misses and coalesced requests. Every
answer reports the tokens it consumed, ingested documents and analysis runs record theirs, and
a request's id follows the jobs it queues, so one id traces a request through the worker's logs.
Set `SYNAPTIX_LLM_PRICES_PER_MTOK` to also report cost; there is no default, because a stale
price would be a confident wrong number.

**Protection.** Per-workspace token buckets (atomic Lua in Redis, on Redis's clock, so replicas
enforce one limit) return `429 Retry-After`. Each replica caps model-backed requests in flight
and refuses the excess with an immediate `503 Retry-After` instead of queueing. Identical
concurrent questions collapse into one model call, and repeated ones are answered from a cache
keyed on the documents' current versions, so it can never serve an answer from before a document
changed. The cache fails open and lives in its own evicting Redis, apart from the job queue,
which never evicts.

## Develop

```sh
make test                 # unit tests + DB-backed retrieval tests (skipped if Postgres is down)
make lint                 # ruff
uv run python -m eval.run            # retrieval metrics on eval/golden.jsonl (hit@k, MRR, terms)
uv run python -m eval.run --answers  # also generate answers and grade faithfulness/correctness
uv sync --extra local     # adds sentence-transformers for local embeddings / reranking
```

CI (`.github/workflows/ci.yml`) runs lint, migrations and tests against a pgvector service, and
builds the image. Configuration is entirely via `SYNAPTIX_*` environment variables (see
`app/core/settings.py`). The original prototype lives in `legacy/` for reference only.
