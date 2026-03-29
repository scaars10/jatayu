"""Simple cache primitives for search results."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections.abc import Mapping
from typing import Protocol, TypeVar, runtime_checkable


T = TypeVar("T")


@runtime_checkable
class CacheBackend(Protocol[T]):
    """Minimal cache backend contract for search results."""

    def get(self, key: str) -> T | None:
        """Return a cached value if present and not expired."""

    def set(self, key: str, value: T, ttl_seconds: int) -> None:
        """Cache a value for a bounded amount of time."""

    def delete(self, key: str) -> None:
        """Remove a cache entry if present."""

    def clear(self) -> None:
        """Remove all cached entries."""


class InMemoryTTLCache(CacheBackend[T]):
    """Thread-safe in-memory TTL cache suitable for a single process."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[float, T]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> T | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            expires_at, value = entry
            if expires_at <= now:
                self._entries.pop(key, None)
                return None

            return copy.deepcopy(value)

    def set(self, key: str, value: T, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return

        expires_at = time.monotonic() + ttl_seconds
        with self._lock:
            self._entries[key] = (expires_at, copy.deepcopy(value))

    def delete(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def make_cache_key(namespace: str, payload: Mapping[str, object]) -> str:
    """Create a deterministic cache key from normalized parameters."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"
