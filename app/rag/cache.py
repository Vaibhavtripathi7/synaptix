"""Answer cache with per-replica request coalescing.

A model call dominates both latency (seconds) and cost (tens of thousands of tokens), and
questions repeat — a class asks its lecture notes the same things. An answer is keyed on exactly
what determines it: the workspace's *library version*, the documents asked about, the normalised
question, the style, and the model and retrieval configuration.

The library version is what keeps a hit off Postgres. It is a random token per workspace, kept in
this Redis; any change to what the workspace can search — a document becoming ready, a document
deleted — deletes it, and the next request mints a new one on the database path. So a hit needs
no database read, and a stale answer is unreachable the moment its library changes: it is stored
under a version nobody will ask for again. Random rather than a counter because this Redis evicts:
a counter evicted back to zero would collide with old entries, a random token never repeats.
Ownership needs no database either: an entry only exists because a miss validated those document
ids in that workspace, and any change to them replaces the version.

Identical questions arriving together would all miss and all call the model. Within a replica
they collapse into one computation; across N replicas at most N run. That bounds the herd without
a distributed lock on the hot path.

The cache fails open: if its Redis is unreachable every lookup is a miss and answers are simply
computed, so a cache outage costs latency, never availability. One failure is not free: an
invalidation that cannot reach Redis leaves the old version in place, so versions expire after
`_VERSION_TTL_S` — that bounds how long a missed invalidation can serve answers from before a
change. For the same reason the cache is not part of readiness: a cache blip must not pull every
replica out of the load balancer at once.

This must be a different Redis from the job queue: a cache is configured to evict under memory
pressure (allkeys-lru), and an evicting Redis would silently drop queued jobs."""

import asyncio
import hashlib
import json
import secrets
import uuid

import structlog
from redis.exceptions import RedisError

from app.core.metrics import ANSWER_CACHE
from app.core.settings import Settings

log = structlog.get_logger()

_FORMAT = 1  # bump when the cached payload's shape changes


_VERSION_TTL_S = 600


def cache_key(
    workspace: str,
    version: str,
    document_ids: list[uuid.UUID] | None,
    question: str,
    style: str,
    settings: Settings,
) -> str:
    material = {
        "format": _FORMAT,
        "workspace": workspace,
        "version": version,
        "docs": sorted({str(i) for i in document_ids}) if document_ids else "*",
        "question": " ".join(question.casefold().split()).rstrip(" ?.!"),
        "style": style,
        "config": [
            settings.llm_model,
            settings.embedding_model,
            settings.reranker,
            settings.retrieval_k,
            settings.rerank_top_n,
            settings.context_tokens,
        ],
    }
    return "ans:" + hashlib.sha256(json.dumps(material).encode()).hexdigest()


class AnswerCache:
    def __init__(self, redis, ttl_s: int) -> None:
        self._redis, self._ttl = redis, ttl_s
        self._leaders: dict[str, asyncio.Future] = {}

    async def version(self, workspace: str) -> str | None:
        """The workspace's current library version, or None: unknown, evicted, or invalidated.
        None sends the request down the database path, which then mints a fresh version."""
        if not self._ttl:
            return None
        try:
            raw = await self._redis.get(f"lib:{workspace}")
        except (RedisError, OSError) as exc:
            log.warning("cache.unavailable", error=str(exc))
            return None
        return raw.decode() if raw else None

    async def mint_version(self, workspace: str) -> str:
        """Sets a new version unless a concurrent request already did, and returns the winner.
        Reaching here means no usable version existed, so the request is a miss."""
        token = secrets.token_hex(8)
        if not self._ttl:
            return token
        ANSWER_CACHE.labels("miss").inc()
        try:
            await self._redis.set(f"lib:{workspace}", token, nx=True, ex=_VERSION_TTL_S)
            return (await self._redis.get(f"lib:{workspace}") or token.encode()).decode()
        except (RedisError, OSError) as exc:
            log.warning("cache.unavailable", error=str(exc))
            return token

    async def get(self, key: str) -> dict | None:
        if not self._ttl:
            return None
        try:
            raw = await self._redis.get(key)
        except (RedisError, OSError) as exc:
            ANSWER_CACHE.labels("error").inc()
            log.warning("cache.unavailable", error=str(exc))
            return None
        ANSWER_CACHE.labels("hit" if raw else "miss").inc()
        return json.loads(raw) if raw else None

    async def follow(self, key: str) -> asyncio.Future | None:
        """The computation of `key` to wait on, if another request on this replica leads it;
        otherwise registers the caller as leader and returns None.

        Registering first and then reading closes the gap between the caller's lookup and this
        call. A leader that finished in between has stood down, but it stores before standing
        down, so its answer is in Redis: the caller gets it as an already settled computation."""
        if (leader := self._leaders.get(key)) is not None:
            ANSWER_CACHE.labels("coalesced").inc()
            return leader
        self._leaders[key] = leader = asyncio.get_running_loop().create_future()
        if not self._ttl:
            return None
        try:
            raw = await self._redis.get(key)
        except (RedisError, OSError) as exc:
            ANSWER_CACHE.labels("error").inc()
            log.warning("cache.unavailable", error=str(exc))
            return None
        except BaseException:
            self.abandon(key)  # cancelled while leading: the caller never gets to stand down
            raise
        if raw is None:
            return None
        ANSWER_CACHE.labels("coalesced").inc()
        self._settle(key, json.loads(raw))
        return leader

    async def put(self, key: str, value: dict) -> None:
        """Stores the leader's answer and hands it to everyone who waited on it."""
        if self._ttl:
            try:
                await self._redis.set(key, json.dumps(value), ex=self._ttl)
            except (RedisError, OSError) as exc:
                log.warning("cache.unavailable", error=str(exc))
        self._settle(key, value)

    async def invalidate(self, workspace: str) -> None:
        await invalidate_library(self._redis, workspace)

    def abandon(self, key: str) -> None:
        """The leader failed: waiters stop waiting and compute for themselves."""
        self._settle(key, None)

    def _settle(self, key: str, value: dict | None) -> None:
        """Ends the leadership of `key`, handing `value` to everyone waiting on it."""
        if (leader := self._leaders.pop(key, None)) and not leader.done():
            leader.set_result(value)


async def invalidate_library(redis, workspace: str) -> None:
    """Called after a change to what `workspace` can search has been committed: a document ready,
    a document deleted. Must run after the commit — before it, a request could re-mint a version
    over the old state. A failure is logged, and bounded by the version's expiry."""
    try:
        await redis.delete(f"lib:{workspace}")
    except (RedisError, OSError) as exc:
        ANSWER_CACHE.labels("invalidation_failed").inc()
        log.warning("cache.invalidation_failed", workspace=workspace, error=str(exc))
