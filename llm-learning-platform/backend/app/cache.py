"""
Redis Cache Manager

Provides async Redis connection management and a caching decorator
for FastAPI endpoints.
"""

import json
import hashlib
import logging
from typing import Optional, Callable, Any
from functools import wraps

import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)

# Global Redis connection pool
_redis: Optional[aioredis.Redis] = None


async def init_redis() -> Optional[aioredis.Redis]:
    """Initialize Redis connection. Returns None if unavailable."""
    global _redis
    settings = get_settings()
    try:
        _redis = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=3,
        )
        await _redis.ping()
        logger.info("Redis connected at %s", settings.redis_url)
        return _redis
    except Exception as e:
        logger.warning("Redis unavailable (%s) — caching disabled.", e)
        _redis = None
        return None


async def close_redis():
    """Close the Redis connection pool."""
    global _redis
    if _redis:
        await _redis.close()
        _redis = None


async def get_redis() -> Optional[aioredis.Redis]:
    """FastAPI dependency that returns the Redis client (or None)."""
    return _redis


def _cache_key(prefix: str, *args: Any, **kwargs: Any) -> str:
    """Generate a deterministic cache key from prefix + arguments."""
    raw = json.dumps({"a": list(args), "k": kwargs}, sort_keys=True, default=str)
    digest = hashlib.md5(raw.encode()).hexdigest()
    return f"cache:{prefix}:{digest}"


def cached(prefix: str, ttl: int = 300):
    """
    Caching decorator for async functions.

    Args:
        prefix: Namespace prefix for the cache key
        ttl: Time-to-live in seconds (default 5 minutes)
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if _redis is None:
                return await func(*args, **kwargs)

            key = _cache_key(prefix, *args, **kwargs)
            try:
                hit = await _redis.get(key)
                if hit is not None:
                    return json.loads(hit)
            except Exception:
                pass  # Cache miss or error — fall through

            result = await func(*args, **kwargs)
            try:
                await _redis.setex(key, ttl, json.dumps(result, default=str))
            except Exception:
                pass  # Don't fail the request if caching fails

            return result

        return wrapper

    return decorator
