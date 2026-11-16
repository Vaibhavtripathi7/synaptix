# Security

Synaptix is meant to run as a public service: no accounts, anyone can create a workspace, upload
PDFs that strangers wrote, and send text that ends up in prompts. This file lists what that
exposes, what defends against each threat, where each defence lives in the code, and what is
still not covered.

## Assets and trust boundaries

| Asset | Why it matters |
|---|---|
| A user's Gemini key | It bills the user's Google account; leaking it spends their money |
| A workspace's documents and answers | Private to whoever holds the workspace token |
| Server capacity: CPU, disk, database, model concurrency | Free to consume by design, so it must be rationed |
| The host | The PDF parser is a large C library fed untrusted files |

Untrusted inputs are HTTP requests (headers, bodies, uploads), the contents of every PDF, and
**everything a model produces from PDF content**, which is treated as untrusted too.

## Threats and defences

### Workspace isolation

- **Token.** A workspace is identified by a client-generated token (32-256 URL-safe
  characters). The server stores and compares only its SHA-256, so a database dump holds no
  usable tokens. Tokens with the wrong shape get 401 before any query runs
  (`app/api/deps.py`, `app/core/validation.py`).
- **Queries.** Every query filters on the workspace, including retrieval, figures, analyses
  and the answer cache, whose keys include the workspace. Deduplication is per workspace, so
  one workspace cannot learn by uploading a file whether another workspace holds the same file.
- **Tests.** `tests/test_workspaces.py` checks cross-workspace reads, including through the
  cache.

### Users' Gemini keys (BYOK)

- **Shape check.** The key is checked against a strict pattern at the edge (400 otherwise). This
  also rules out header smuggling through it.
- **Masking.** The key is wrapped in `SecretStr` straight away, so every repr masks it,
  including arq's job-argument log lines. Jobs that carry it keep no result (`keep_result=0`),
  so the key is not left in Redis after the job finishes.
- **No persistence.** The key is never written to the database or to logs. The browser keeps
  it in `localStorage` for that user only.
- **No server key.** A public deployment sets no key of its own
  (`deploy/env.production.example`). Requests without a key are refused before anything is
  queued, so the service cannot be used to spend the operator's quota.

### Abuse and resource exhaustion

- **Rate limits.** Every route is charged to a token bucket per workspace *and* per client IP.
  One atomic Lua script checks both, and takes from neither unless both allow the request
  (`app/core/limits.py`). The IP bucket stops a client from minting workspaces to multiply its
  allowance. Classes are `model`, `upload` and `read`, with uploads the tightest.
- **Client IP.** The client IP comes only from the X-Forwarded-For entry written by a
  configured number of trusted proxies (`trusted_proxy_hops`), counted from the right. A client
  that forges the header just adds entries to the left, and those are ignored.
  `tests/test_workspaces.py` checks this.
- **Admission control.** Each replica admits a bounded number of in-flight model requests and
  answers 503 with Retry-After above that, rather than queueing without limit.
- **Quotas.** An upload may be at most 100 MB and 500 pages. A workspace may hold at most 50
  documents and 500 MB, and the server as a whole caps total stored bytes, returning 507 before
  the disk fills. Quotas are checked before the file is stored.
- **Cleanup.** Failed uploads, including their source files, are purged by a cron job after 24
  h, so rejected files cannot accumulate.
- **Request bounds.** Every request body field is bounded, for example questions at 3-2000
  characters and document id lists at 50, so no request can make an unbounded SQL `IN` list or
  prompt. Output tokens are capped per model call (`llm_max_output_tokens`).

### Malicious PDFs

- **Sandbox.** Each PDF is parsed in a child process (`app/ingest/sandbox.py`). The kernel
  enforces an address-space limit (1.5 GB) and a CPU-time limit, and the parent kills the child
  at a wall-clock deadline or when the job is cancelled. A parser crash, hang or memory bomb
  costs one job, not the worker.
- **Tight memory limit.** Capping glibc arenas and OpenMP threads in the child cut reserved
  address space from ~2.6 GB to ~0.9 GB, with the same memory in use and the same speed. That
  is what lets the address-space limit sit close to real usage.
- **JSON, not pickle.** Results come back as JSON. A child compromised by a crafted PDF can
  only return data, not code to run in the worker.
- **Safe errors.** The child's traceback is never shown to users. Rejections map to a small set
  of fixed reasons.

### Prompt injection

A PDF, or a table or figure inside one, can contain text such as "ignore your instructions and...".
Defences come in layers, because no single one is reliable:

1. **Spotlighting** (`app/llm/untrusted.py`). Every untrusted span is fenced with markers that
   carry a random per-call nonce, and the system prompt says that fenced text is data, not
   instructions. This covers passages, titles, headings, table text, figure captions, and
   earlier-stage model outputs fed into later agents. Marker look-alikes inside the text are
   neutralised, so a document cannot close a fence early. It cannot forge one either, because
   it cannot know the nonce in advance.
2. **No tools.** The models have no tools, network access or write access. The worst an
   injected instruction can do is change the text of an answer.
3. **Blast radius.** A document only ever reaches its own workspace's prompts, so a poisoned
   PDF can only mislead the person who uploaded it.
4. **Output validation.** Citation markers that point to passages the model was not given are
   removed. Analysis evidence is filtered to passage numbers the model actually saw. Structured
   outputs are parsed against schemas.
5. **Safe rendering.** Model output is rendered as Markdown through DOMPurify. Links are forced
   to `rel="noopener noreferrer nofollow"`, and a strict Content-Security-Policy (no inline
   script, `connect-src 'self'`, `frame-ancestors 'none'`) blocks both script injection and
   exfiltration through the page, for example via an injected image URL to an attacker's host.

`tests/test_e2e.py` plants injections in chunks and sections, and asserts the following:
the fence reaches the model, the planted markers are neutralised, forged citations are
stripped, and invented evidence is dropped. Each defence was mutation-checked: disabling it
makes its test fail.

### Web and HTTP

- **Filenames.** Uploaded filenames are reduced to their last path component, NFC-normalised,
  and stripped of control and bidi-override characters as well as quote and header
  metacharacters. They are served with an RFC 6266 `Content-Disposition` header (ASCII fallback
  plus `filename*`), so a filename cannot inject headers or spoof its extension. Figure names are
  whitelisted by pattern, which rules out path traversal.
- **Request ids.** Incoming request ids are restricted to log-safe characters, which prevents
  forged log fields.
- **Security headers.** Caddy terminates TLS and sets HSTS, CSP, `nosniff`, `no-referrer` and
  frame denial. Metrics are served on internal ports only, and `/metrics` answers 404 publicly.

### Host and supply chain

- **Container hardening** (`docker-compose.prod.yml`, verified on a running stack):
  - Every container that runs our code, plus Caddy, has a read-only root filesystem, an
    in-memory `/tmp`, all Linux capabilities dropped (`CapEff: 0`), no privilege escalation
    (`NoNewPrivs: 1`), a process limit and a memory limit.
  - Caddy gets back exactly one capability, binding ports 80 and 443.
  - App processes run as a non-root user.
  - Only Caddy publishes ports.
- **Minimal image.** The image contains only the runtime dependencies and the tokenizer data,
  baked in at build time, so containers need no egress at startup. `.dockerignore` is a
  whitelist.
- **Dependency audit.** CI runs `pip-audit` on the exact locked production dependency set on
  every push, and weekly, because new advisories appear for code that has not changed.
- **Secrets.** Secrets live in `.env`, which is git-ignored, never baked into the image, and
  never logged.

## Residual risks

- **Prompt injection is reduced, not solved.** A determined document can still make an answer
  wrong or misleading for its own uploader. Answers cite their passages so readers can check
  them.
- **The workspace token is a bearer secret held in `localStorage`.** Anyone who has it, or who
  runs script in the page, has the workspace. The CSP makes injected script hard to get running,
  but there is no second factor.
- **Parser zero-days.** The sandbox contains resource abuse and crashes. A memory-corruption
  exploit in MuPDF would still run as the worker's user, inside a capability-less, read-only
  container. A seccomp profile or gVisor would narrow this further.
- **Distributed abuse.** Per-IP limits do not stop a botnet. That needs an edge service
  (a CDN/WAF, or proof-of-work on workspace creation).
- **Users' own keys are only as safe as their browser.** A compromised browser leaks the key,
  whatever the server does.

## Reporting

Please report vulnerabilities privately through GitHub's security advisories for this
repository rather than in a public issue.
