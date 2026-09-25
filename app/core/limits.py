"""Protection for a shared, horizontally scaled API: per-workspace rate limits and per-replica
admission control. They fail differently on purpose — a rate limit is the caller's problem
(429, try later), a full replica is the platform's (503, retry elsewhere)."""

import math

from fastapi import HTTPException

from app.core.settings import get_settings

# Token bucket, evaluated atomically in Redis so N replicas enforce one limit, not N. Time comes
# from Redis's clock: replica clocks drift, and a bucket refilled by a fast clock leaks capacity.
_BUCKET = """
local rate, burst = tonumber(ARGV[1]), tonumber(ARGV[2])
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1e6
local state = redis.call('HMGET', KEYS[1], 'tokens', 'at')
local tokens = tonumber(state[1]) or burst
local at = tonumber(state[2]) or now
tokens = math.min(burst, tokens + (now - at) * rate)
local allowed, retry = 0, (1 - tokens) / rate
if tokens >= 1 then
  tokens, allowed, retry = tokens - 1, 1, 0
end
redis.call('HSET', KEYS[1], 'tokens', tokens, 'at', now)
redis.call('EXPIRE', KEYS[1], math.ceil(burst / rate) + 1)
return {allowed, tostring(retry)}
"""

BUCKETS: dict[str, tuple[float, int]] = get_settings().rate_limits


class RateLimiter:
    def __init__(self, redis) -> None:
        self._script = redis.register_script(_BUCKET)

    async def check(self, workspace: str, bucket: str) -> None:
        """Charges one request to `bucket` for `workspace`; raises 429 when it is empty."""
        rate, burst = BUCKETS[bucket]
        allowed, retry = await self._script(keys=[f"rl:{bucket}:{workspace}"], args=[rate, burst])
        if not int(allowed):
            raise HTTPException(
                429,
                f"too many {bucket} requests; retry later",
                headers={"Retry-After": str(max(1, math.ceil(float(retry))))},
            )


class Admission:
    """Caps model-backed requests in flight on this replica. A request over the cap is refused
    at once rather than queued: queueing converts overload into unbounded latency and memory,
    while a fast 503 lets the load balancer send the retry to a replica with room."""

    def __init__(self, limit: int) -> None:
        self.limit, self.in_flight = limit, 0

    def enter(self) -> "Ticket":
        if self.in_flight >= self.limit:
            raise HTTPException(503, "server busy, retry shortly", headers={"Retry-After": "1"})
        self.in_flight += 1
        return Ticket(self)


class Ticket:
    """One admitted request. Release is idempotent: a streamed response releases in its
    generator *and* in a post-response task, since a client may vanish before streaming starts."""

    def __init__(self, admission: Admission) -> None:
        self._admission, self._held = admission, True

    def release(self) -> None:
        if self._held:
            self._held = False
            self._admission.in_flight -= 1
