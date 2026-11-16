# Capacity model

How much load Synaptix handles, where it breaks, and what changes at each order of magnitude.
Every number here was measured on this codebase; `bench/run.py` reproduces the throughput table.

## Two kinds of request

A request either reaches the model or it does not, and the two differ by four orders of magnitude.

| | answered from our own state | reaches the model |
|---|---|---|
| examples | library listing, document pages, figures, **cached answers** | an uncached question |
| cost | one Redis round trip, one indexed query | 3 model calls, **~17k input tokens**, 10-20 s |
| scales with | API replicas | the model provider's capacity, and the caller's quota |

The whole design follows from that split: keep the first kind cheap and stateless, and make as
much traffic as possible the first kind.

## Per-replica throughput (measured)

One API process is one replica (one process per container, scaled by adding containers). Server
pinned to 4 cores, load generator to the other 4, 64 connections, 15 s after a warm-up run.

| path | 1 process | 4 processes | p99 at 4 |
|---|---:|---:|---:|
| framework floor (`/healthz`) | 4,401 req/s | 8,631 req/s | 14.5 ms |
| authenticated database read | 283 req/s | 634 req/s | 135 ms |
| cached answer | 282 req/s | 610 req/s | 165 ms |

An authenticated read pays for the workspace check, an atomic rate-limit script in Redis, an
indexed query, and response validation. A cached answer is served from Redis alone - the
workspace's library version, then the answer - which is why it now outruns a database read. Four processes do not reach 4x on this machine because Postgres and Redis share its
eight cores with the server and the load generator; on separate hosts, API replicas scale
close to linearly, since they share no state.

### What moved these numbers

Profiling first, then one change at a time, re-measured after each:

| change | effect at 4 processes |
|---|---|
| list documents as plain rows instead of ORM entities | read throughput **396 -> 634 req/s** |
| drop `pool_pre_ping` (a round trip on every checkout); recycle connections by age instead | p99 **913 -> 135 ms** together with the above |
| pool sized to the concurrency a replica actually runs (5+10 -> 20+10) | removed queueing for connections |
| defer the `graph` column | listing no longer drags up to hundreds of KB per document out of Postgres |
| answer cache hits served from Redis alone (per-workspace library version) | cached answers **434 -> 610 req/s**, p99 196 -> 165 ms; from 0.7x to 1.1x the throughput of a database read |

Runs vary by about 10% on this machine (the framework floor measured 8,631 and 7,316 req/s on
different days), so compare rows within one run: `bench/results.md` holds the latest.

Measured and **rejected**:

- *Returning cached bytes without re-validating them.* 21x faster for that step (63 us -> 3 us),
  but that is ~1% of a 5 ms request. Not worth a second code path.
- *docker-proxy on the published Postgres port.* Suspected; measured at under 2%.
- *Re-enabling asyncpg's prepared-statement cache.* Worth ~12% of read throughput, but it breaks
  on migrations applied while workers run, and is incompatible with pgbouncer in transaction
  mode. Kept off.

## The model path

One *brief* question, measured end to end:

| stage | input tokens | share |
|---|---:|---:|
| query expansion | 108 | 1% |
| listwise reranking of 40 passages | 14,317 | **84%** |
| answer generation | 4,643 | 15% |

Throughput on this path is set by latency, not CPU. Each replica admits at most
`SYNAPTIX_MAX_INFLIGHT_MODEL_REQUESTS` model-backed requests (default 64) and refuses the rest
with `503 Retry-After`. By Little's law, 64 in flight at ~15 s each is roughly **4 uncached
questions per second per replica**. Raising the cap is cheap in memory (the work is I/O-bound),
so the real ceiling is the provider.

Two properties keep that ceiling from being ours:

- **Bring-your-own-key spreads the provider bottleneck across users.** Each user's key has its
  own quota and gets its own client and concurrency limit, so the model capacity available grows
  with the number of users instead of being one shared allowance.
- **Per-document work is computed once.** Study guides, summaries and entity graphs are stored
  after the first request and served as reads. Only free-form questions cost model time per
  request - and repeats of those are cached, with identical concurrent questions coalesced into
  a single model call.

## Scaling by order of magnitude

Each stage lists the trigger - the metric that says it is time - rather than a date.

**Up to ~500 req/s (now).** One Postgres, a queue Redis, a cache Redis, N API replicas behind a
load balancer, workers scaled on `synaptix_queue_depth`.
*Triggers for the next stage:* connections - each replica holds up to 30 (pool 20 + overflow
10), so a fourth replica exceeds Postgres's default 100 - and **a second host**. On one machine
source PDFs and figures can live on a local volume; across machines they cannot, because a
worker cannot read a file another host received. `SYNAPTIX_STORAGE_BACKEND=s3` moves them to any
S3-compatible store (S3, R2, GCS interop), after which no state lives on an API or worker host
at all. The client is a small async S3 implementation over httpx with its own SigV4 signer,
checked against AWS's published signatures and against a server that validates them; the
viewer's ranged reads are passed through, so a 50 MB PDF is still fetched a page at a time.

**~5,000 req/s.** pgbouncer in transaction mode in front of Postgres (already compatible: the
statement cache is off). A read replica for retrieval queries. Size the cache Redis from the
observed hit ratio (`synaptix_answer_cache_total`).
*Trigger:* Postgres CPU on the primary, or cache evictions rising while the hit ratio falls.

**~50,000 req/s.** Postgres is already off the cache-hit path: a hit reads the workspace's
library version and the answer from Redis, and a document becoming ready or being deleted
replaces that version, so there is no database read to validate freshness. What remains is
Redis itself: Redis Cluster for the cache and rate limits (both keyed by workspace, so they
shard naturally). Serve the static UI
from a CDN; move figures to short-lived signed URLs so they can be cached at the edge (today
they are fetched with a header, which a CDN cannot cache).
*Trigger:* Redis CPU or memory on the cache instance.

**Toward 1,000,000 req/s.** For reads and cached answers this is a matter of fleet size:
at ~280 cached answers per core, about 3,600 API cores, and three Redis operations per cached
answer (rate limit, library version, answer) at ~100k operations per shard, about 30 shards,
spread across regions. Uncached questions do not scale this way under *any* design: at ~19k tokens each,
a million per second would be ~19 billion tokens per second, beyond what model providers serve.
At that scale the product has to be mostly cache hits and precomputed per-document artifacts,
which is exactly what the model path above is built around.

## Protection already in place

- Per-workspace token buckets (`model`, `upload`, `read`) run as one atomic Lua script in Redis,
  on Redis's clock, so N replicas enforce one limit: `429 Retry-After`.
- Per-replica admission control on model-backed requests: an immediate `503 Retry-After`, never
  an unbounded queue.
- API processes never sleep on a provider rate limit inside a request; they fail over to the
  next model. Only workers wait.
- The answer cache fails open (a cache outage costs latency, not availability) and is excluded
  from readiness, so a cache blip cannot pull every replica out of the load balancer. Library
  versions expire after ten minutes, which bounds how long a failed invalidation (the cache
  unreachable at the moment a document changed) can serve answers from before the change.
- The queue Redis never evicts; the cache Redis always may. They are separate instances for that
  reason.
- No database transaction is held across a model call, and migrations run with `lock_timeout`,
  so a schema change cannot queue the whole application behind a slow request.
