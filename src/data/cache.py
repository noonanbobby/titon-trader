"""Redis-based data caching layer for Project Titan.

Provides a thin async wrapper around Redis for caching API responses,
rate-limit tracking, and pub/sub message distribution.  Every external
API client in ``src/data/`` should route through this layer to avoid
redundant network calls and to respect third-party rate limits.

Usage::

    from src.data.cache import RedisCache

    cache = RedisCache(redis_url="redis://redis:6379/0")
    await cache.connect()

    # Cache an API response for 5 minutes
    await cache.set_json("polygon:AAPL:bars", bar_data, ttl=300)
    data = await cache.get_json("polygon:AAPL:bars")

    # Rate-limit tracking
    allowed = await cache.rate_limit_check("finnhub", max_calls=30, window=60)

    await cache.close()
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import redis.asyncio as aioredis
from pydantic import BaseModel, Field

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TTL_SECONDS: int = 300
MAX_TTL_SECONDS: int = 86_400  # 24 hours
RATE_LIMIT_KEY_PREFIX: str = "ratelimit"
CACHE_KEY_PREFIX: str = "cache"
PUBSUB_CHANNEL_MARKET_DATA: str = "titan:market_data"
PUBSUB_CHANNEL_SIGNALS: str = "titan:signals"
PUBSUB_CHANNEL_ORDERS: str = "titan:orders"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CacheStats(BaseModel):
    """Snapshot of cache hit/miss statistics."""

    hits: int = Field(default=0, description="Total cache hits")
    misses: int = Field(default=0, description="Total cache misses")
    sets: int = Field(default=0, description="Total cache writes")
    evictions: int = Field(default=0, description="Keys expired or evicted")

    @property
    def hit_rate(self) -> float:
        """Return the cache hit rate as a fraction 0.0–1.0."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Cache implementation
# ---------------------------------------------------------------------------


class RedisCache:
    """Async Redis cache with JSON serialization, TTLs, and rate limiting.

    Parameters
    ----------
    redis_url:
        Redis connection URL, e.g. ``redis://redis:6379/0``.
    default_ttl:
        Default TTL in seconds for cached values.
    key_prefix:
        Optional global prefix prepended to every key.
    """

    def __init__(
        self,
        redis_url: str = "redis://redis:6379/0",
        default_ttl: int = DEFAULT_TTL_SECONDS,
        key_prefix: str = CACHE_KEY_PREFIX,
    ) -> None:
        self._url = redis_url
        self._default_ttl = default_ttl
        self._prefix = key_prefix
        self._client: aioredis.Redis | None = None
        self._stats = CacheStats()
        self._log: structlog.BoundLogger = get_logger("data.cache")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the Redis connection and verify with a PING."""
        self._client = aioredis.from_url(
            self._url,
            decode_responses=True,
            socket_connect_timeout=5.0,
            socket_timeout=5.0,
        )
        await self._client.ping()
        self._log.info("redis_cache_connected", url=self._url)

    async def close(self) -> None:
        """Gracefully close the Redis connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._log.info("redis_cache_closed")

    @property
    def connected(self) -> bool:
        """Return True if the Redis client is initialized."""
        return self._client is not None

    @property
    def stats(self) -> CacheStats:
        """Return current cache statistics."""
        return self._stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, name: str) -> str:
        """Build a fully-qualified Redis key."""
        return f"{self._prefix}:{name}"

    def _ensure_connected(self) -> aioredis.Redis:
        """Return the client or raise if not connected."""
        if self._client is None:
            raise RuntimeError("RedisCache is not connected. Call connect() first.")
        return self._client

    # ------------------------------------------------------------------
    # Basic get / set
    # ------------------------------------------------------------------

    async def get(self, key: str) -> str | None:
        """Retrieve a raw string value by key.

        Returns None on cache miss or Redis failure (graceful degradation).
        """
        try:
            client = self._ensure_connected()
            val = await client.get(self._key(key))
            if val is None:
                self._stats.misses += 1
            else:
                self._stats.hits += 1
            return val
        except Exception:
            self._log.warning("redis_get_failed", key=key)
            self._stats.misses += 1
            return None

    async def set(
        self,
        key: str,
        value: str,
        ttl: int | None = None,
    ) -> None:
        """Store a raw string value with an optional TTL.

        Silently fails on Redis errors (graceful degradation).

        Parameters
        ----------
        key:
            Cache key (prefix is added automatically).
        value:
            String value to store.
        ttl:
            Time-to-live in seconds.  Defaults to ``default_ttl``.
        """
        try:
            client = self._ensure_connected()
            effective_ttl = min(ttl or self._default_ttl, MAX_TTL_SECONDS)
            await client.set(self._key(key), value, ex=effective_ttl)
            self._stats.sets += 1
        except Exception:
            self._log.warning("redis_set_failed", key=key)

    async def delete(self, key: str) -> bool:
        """Delete a key.  Returns True if the key existed."""
        try:
            client = self._ensure_connected()
            result = await client.delete(self._key(key))
            return result > 0
        except Exception:
            self._log.warning("redis_delete_failed", key=key)
            return False

    async def exists(self, key: str) -> bool:
        """Return True if key exists in cache.  Returns False on Redis failure."""
        try:
            client = self._ensure_connected()
            return bool(await client.exists(self._key(key)))
        except Exception:
            self._log.warning("redis_exists_failed", key=key)
            return False

    # ------------------------------------------------------------------
    # JSON get / set
    # ------------------------------------------------------------------

    async def get_json(self, key: str) -> Any | None:
        """Retrieve a JSON-serialized value.

        Returns the deserialized Python object, or None on cache miss.
        """
        raw = await self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            self._log.warning("cache_json_decode_error", key=key)
            return None

    async def set_json(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Store a value as JSON with an optional TTL."""
        raw = json.dumps(value, default=str)
        await self.set(key, raw, ttl=ttl)

    # ------------------------------------------------------------------
    # Rate-limit helpers
    # ------------------------------------------------------------------

    async def rate_limit_check(
        self,
        service: str,
        max_calls: int,
        window: int,
    ) -> bool:
        """Check and increment a sliding-window rate-limit counter.

        On Redis failure, returns True (allow the call) to avoid blocking
        the entire trading pipeline when Redis is down.

        Parameters
        ----------
        service:
            Identifier for the API being rate-limited (e.g. ``"finnhub"``).
        max_calls:
            Maximum calls allowed within the window.
        window:
            Window size in seconds.

        Returns
        -------
        bool
            True if the call is allowed, False if the limit is exceeded.
        """
        try:
            client = self._ensure_connected()
            rk = f"{RATE_LIMIT_KEY_PREFIX}:{service}"
            now = time.time()
            cutoff = now - window

            pipe = client.pipeline(transaction=True)
            pipe.zremrangebyscore(rk, "-inf", cutoff)
            pipe.zcard(rk)
            pipe.zadd(rk, {str(now): now})
            pipe.expire(rk, window + 1)
            results = await pipe.execute()

            current_count: int = results[1]
            allowed = current_count < max_calls
            if not allowed:
                self._log.debug(
                    "rate_limit_exceeded",
                    service=service,
                    count=current_count,
                    max=max_calls,
                )
            return allowed
        except Exception:
            self._log.warning("redis_rate_limit_check_failed", service=service)
            return True

    async def rate_limit_wait(
        self,
        service: str,
        max_calls: int,
        window: int,
    ) -> None:
        """Block until the rate limit allows another call.

        Uses a simple polling loop with exponential back-off.
        """
        import asyncio

        delay = 0.1
        while not await self.rate_limit_check(service, max_calls, window):
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 5.0)

    # ------------------------------------------------------------------
    # Pub/Sub helpers
    # ------------------------------------------------------------------

    async def publish(self, channel: str, message: dict[str, Any]) -> int:
        """Publish a JSON message to a Redis Pub/Sub channel.

        Returns the number of subscribers that received the message.
        Returns 0 on Redis failure.
        """
        try:
            client = self._ensure_connected()
            payload = json.dumps(message, default=str)
            return await client.publish(channel, payload)
        except Exception:
            self._log.warning("redis_publish_failed", channel=channel)
            return 0

    async def subscribe(
        self,
        channel: str,
    ) -> aioredis.client.PubSub:
        """Subscribe to a Redis Pub/Sub channel.

        Returns a PubSub object.  Caller is responsible for iterating
        messages and calling ``unsubscribe()`` / ``aclose()`` when done.
        """
        client = self._ensure_connected()
        pubsub = client.pubsub()
        await pubsub.subscribe(channel)
        self._log.info("pubsub_subscribed", channel=channel)
        return pubsub

    # ------------------------------------------------------------------
    # Bulk / maintenance
    # ------------------------------------------------------------------

    async def flush_prefix(self, prefix: str) -> int:
        """Delete all keys matching a prefix pattern.

        Parameters
        ----------
        prefix:
            Key prefix to match (e.g. ``"polygon"`` deletes all
            ``cache:polygon:*`` keys).

        Returns
        -------
        int
            Number of keys deleted.  Returns 0 on Redis failure.
        """
        try:
            client = self._ensure_connected()
            pattern = f"{self._prefix}:{prefix}:*"
            count = 0
            async for key in client.scan_iter(match=pattern, count=200):
                await client.delete(key)
                count += 1
            if count:
                self._log.info("cache_prefix_flushed", prefix=prefix, count=count)
            return count
        except Exception:
            self._log.warning("redis_flush_prefix_failed", prefix=prefix)
            return 0

    async def get_ttl(self, key: str) -> int:
        """Return the remaining TTL in seconds for a key, or -1 if no TTL."""
        try:
            client = self._ensure_connected()
            return await client.ttl(self._key(key))
        except Exception:
            self._log.warning("redis_get_ttl_failed", key=key)
            return -1
