"""Answer cache with per-replica request coalescing.

A model call dominates both latency (seconds) and cost (tens of thousands of tokens), and
questions repeat — a class asks its lecture notes the same things. An answer is keyed on exactly
what determines it: the documents searched *and their versions* (`updated_at`), the normalised
question, the style, and the model and retrieval configuration. A document that is re-ingested
or deleted changes the key, so a stale answer is never served; it simply ages out.

Identical questions arriving together would all miss and all call the model. Within a replica
they collapse into one computation; across N replicas at most N run. That bounds the herd without
a distributed lock on the hot path.

The cache fails open: if its Redis is unreachable every lookup is a miss and answers are simply
computed, so a cache outage costs latency, never availability. For the same reason the cache is
not part of readiness: a cache blip must not pull every replica out of the load balancer at once.

This must be a different Redis from the job queue: a cache is configured to evict under memory
pressure (allkeys-lru), and an evicting Redis would silently drop queued jobs."""

import asyncio
import hashlib
import json
import uuid
from datetime import datetime

import structlog
from redis.exceptions import RedisError

from app.core.metrics import ANSWER_CACHE
from app.core.settings import Settings

log = structlog.get_logger()

_FORMAT = 1  # bump when the cached payload's shape changes


def cache_key(
    workspace: str,
    versions: dict[uuid.UUID, datetime],
    question: str,
    style: str,
    settings: Settings,
) -> str:
    material = {
        "format": _FORMAT,
        "workspace": workspace,
        "docs": sorted(f"{i}@{v.isoformat()}" for i, v in versions.items()),
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

    def follow(self, key: str) -> asyncio.Future | None:
        """The in-flight computation of `key` on this replica, if another request leads it;
        otherwise registers the caller as leader and returns None."""
        if (leader := self._leaders.get(key)) is not None:
            ANSWER_CACHE.labels("coalesced").inc()
            return leader
        self._leaders[key] = asyncio.get_running_loop().create_future()
        return None

    async def put(self, key: str, value: dict) -> None:
        """Stores the leader's answer and hands it to everyone who waited on it."""
        if self._ttl:
            try:
                await self._redis.set(key, json.dumps(value), ex=self._ttl)
            except (RedisError, OSError) as exc:
                log.warning("cache.unavailable", error=str(exc))
        if (leader := self._leaders.pop(key, None)) and not leader.done():
            leader.set_result(value)

    def abandon(self, key: str) -> None:
        """The leader failed: waiters stop waiting and compute for themselves."""
        if (leader := self._leaders.pop(key, None)) and not leader.done():
            leader.set_result(None)
