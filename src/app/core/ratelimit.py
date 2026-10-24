"""Per-caller rate limiting (C-197).

Token bucket per user/API key: RATE_LIMIT_PER_MINUTE requests, refilled continuously.
In-memory buckets are per process; with several instances set REDIS_URL so the limit
is shared (fixed one-minute windows in Redis).
"""

import time
from typing import Annotated

from fastapi import Depends, HTTPException, status

from app.core.auth import CurrentUser
from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class InMemoryRateLimiter:
    def __init__(self, per_minute: int, max_keys: int = 50_000) -> None:
        self._rate = per_minute / 60.0
        self._capacity = float(per_minute)
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_ts)
        self._max_keys = max_keys

    async def acquire(self, key: str) -> float | None:
        """None if allowed, else seconds to wait."""
        now = time.monotonic()
        tokens, last = self._buckets.get(key, (self._capacity, now))
        tokens = min(self._capacity, tokens + (now - last) * self._rate)
        if tokens < 1:
            self._buckets[key] = (tokens, now)
            return (1 - tokens) / self._rate
        if len(self._buckets) >= self._max_keys and key not in self._buckets:
            self._buckets.clear()  # crude bound on memory; limits restart for everyone
        self._buckets[key] = (tokens - 1, now)
        return None


class RedisRateLimiter:
    """Fixed-window counter shared by all instances (needs REDIS_URL)."""

    def __init__(self, redis_url: str, per_minute: int) -> None:
        from redis.asyncio import Redis

        self._redis = Redis.from_url(redis_url)
        self._limit = per_minute

    async def acquire(self, key: str) -> float | None:
        window = int(time.time() // 60)
        redis_key = f"ratelimit:{key}:{window}"
        count = await self._redis.incr(redis_key)
        if count == 1:
            await self._redis.expire(redis_key, 70)
        if count > self._limit:
            return 60 - time.time() % 60
        return None


_limiter = None


def _get_limiter(settings: Settings):
    global _limiter
    if _limiter is None:
        redis_url = settings.redis_url.get_secret_value()
        if redis_url:
            _limiter = RedisRateLimiter(redis_url, settings.rate_limit_per_minute)
        else:
            _limiter = InMemoryRateLimiter(settings.rate_limit_per_minute)
    return _limiter


async def enforce_rate_limit(
    user: CurrentUser, settings: Annotated[Settings, Depends(get_settings)]
) -> None:
    if settings.rate_limit_per_minute <= 0:
        return
    wait = await _get_limiter(settings).acquire(user.rate_limit_key)
    if wait is not None:
        logger.warning("rate_limited", user_id=user.user_id)
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many requests",
            headers={"Retry-After": str(max(1, int(wait) + 1))},
        )
