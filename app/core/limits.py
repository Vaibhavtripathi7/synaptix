"""Protection for a shared, horizontally scaled API: per-workspace rate limits and per-replica
admission control. They fail differently on purpose - a rate limit is the caller's problem
(429, try later), a full replica is the platform's (503, retry elsewhere)."""

import hashlib
import math

from fastapi import HTTPException

from app.core.settings import get_settings

# Token buckets, evaluated atomically in Redis so N replicas enforce one limit, not N. Several
# buckets are checked in one call - the workspace's and the client IP's - and tokens are taken
# from none unless all allow: a request refused by its IP limit must not also spend its
# workspace's allowance. Time comes from Redis's clock: replica clocks drift, and a bucket refilled
# by a fast clock leaks capacity.
_BUCKETS = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1e6
local levels, retry = {}, 0
for i, key in ipairs(KEYS) do
  local rate, burst = tonumber(ARGV[2 * i - 1]), tonumber(ARGV[2 * i])
  local state = redis.call('HMGET', key, 'tokens', 'at')
  local tokens = tonumber(state[1]) or burst
  local at = tonumber(state[2]) or now
  tokens = math.min(burst, tokens + (now - at) * rate)
  levels[i] = tokens
  if tokens < 1 then retry = math.max(retry, (1 - tokens) / rate) end
end
for i, key in ipairs(KEYS) do
  local rate, burst = tonumber(ARGV[2 * i - 1]), tonumber(ARGV[2 * i])
  local tokens = levels[i]
  if retry == 0 then tokens = tokens - 1 end
  redis.call('HSET', key, 'tokens', tokens, 'at', now)
  redis.call('EXPIRE', key, math.ceil(burst / rate) + 1)
end
return {retry == 0 and 1 or 0, tostring(retry)}
"""

BUCKETS: dict[str, tuple[float, int]] = get_settings().rate_limits
IP_BUCKETS: dict[str, tuple[float, int]] = get_settings().ip_rate_limits


class RateLimiter:
    def __init__(self, redis) -> None:
        self._script = redis.register_script(_BUCKETS)

    async def check(self, workspace: str, bucket: str, client_ip: str | None = None) -> None:
        """Charges one request to the workspace's bucket and, when known, the client IP's.
        Workspaces are free to create, so the IP bucket is what bounds one client in the end."""
        keys, args = [f"rl:{bucket}:ws:{workspace}"], list(BUCKETS[bucket])
        if client_ip:
            keys.append(f"rl:{bucket}:ip:{hashlib.sha256(client_ip.encode()).hexdigest()[:24]}")
            args += list(IP_BUCKETS[bucket])
        allowed, retry = await self._script(keys=keys, args=args)
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
