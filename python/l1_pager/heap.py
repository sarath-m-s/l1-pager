"""
L1-Pager: Heap interface and implementations.

The Heap is the "virtual memory" backing store. It must guarantee:
  - O(1) average get/put
  - LRU eviction when max_heap_entries is exceeded
  - Thread-safe async access

Phase 1 ships InMemoryHeap. RedisHeap is wired up in Phase 2.
"""
from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional

from .schema import EvictionConfig, PageHeapEntry

# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class AbstractHeap(ABC):
    """
    The common contract both Python heap implementations must satisfy.

    All methods are async to ensure the InMemoryHeap and RedisHeap are
    drop-in replaceable without changing call sites.
    """

    @abstractmethod
    async def put(self, entry: PageHeapEntry) -> None:
        """Store an entry. Evicts oldest entry if capacity is exceeded."""

    @abstractmethod
    async def get(self, page_id: str) -> Optional[PageHeapEntry]:
        """
        Retrieve and touch (LRU update) an entry.
        Returns None if the page_id is unknown or TTL has expired.
        """

    @abstractmethod
    async def delete(self, page_id: str) -> None:
        """Explicitly remove an entry (called after re-injection)."""

    @abstractmethod
    async def size(self) -> int:
        """Current number of entries in the heap."""


# ---------------------------------------------------------------------------
# InMemoryHeap — LRU OrderedDict, no external deps
# ---------------------------------------------------------------------------


class InMemoryHeap(AbstractHeap):
    """
    Thread-safe in-process heap backed by an OrderedDict for O(1) LRU.

    Suitable for single-process agents or testing. Data does not survive
    process restart — use RedisHeap for persistence.
    """

    def __init__(self, config: EvictionConfig) -> None:
        self._config = config
        # OrderedDict maintains insertion order; we move-to-end on access
        self._store: OrderedDict[str, PageHeapEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    async def put(self, entry: PageHeapEntry) -> None:
        async with self._lock:
            # Overwrite if page_id already exists (idempotent sweep)
            if entry.page_id in self._store:
                del self._store[entry.page_id]

            self._store[entry.page_id] = entry

            # Evict LRU entries if over capacity
            while len(self._store) > self._config.max_heap_entries:
                self._store.popitem(last=False)

    async def get(self, page_id: str) -> Optional[PageHeapEntry]:
        async with self._lock:
            entry = self._store.get(page_id)
            if entry is None:
                return None

            # TTL check
            age = time.time() - entry.evicted_at_ts
            if age > self._config.entry_ttl_seconds:
                del self._store[page_id]
                return None

            entry.touch()
            # Move to end (most-recently-used position)
            self._store.move_to_end(page_id)
            return entry

    async def delete(self, page_id: str) -> None:
        async with self._lock:
            self._store.pop(page_id, None)

    async def size(self) -> int:
        async with self._lock:
            return len(self._store)

    # Convenience for tests / debugging
    async def all_ids(self) -> list[str]:
        async with self._lock:
            return list(self._store.keys())


# ---------------------------------------------------------------------------
# RedisHeap — Phase 2 stub
# ---------------------------------------------------------------------------


class RedisHeap(AbstractHeap):
    """
    Redis-backed heap.  Key format: ``l1pager:heap:{page_id}`` → JSON string.

    TTL is enforced by Redis EXPIRE so entries evict automatically even if
    the Python process dies.  Accessing an entry resets its TTL (LRU signal).

    ``size()`` uses SCAN instead of KEYS to be safe on large keyspaces.

    Requires ``redis[asyncio]`` (redis-py ≥ 4.2).  Install with:
        pip install "redis[asyncio]>=4.2"
    """

    _KEY_PREFIX = "l1pager:heap:"

    def __init__(
        self,
        config: EvictionConfig,
        redis_url: str = "redis://localhost:6379",
    ) -> None:
        self._config = config
        self._redis_url = redis_url
        self._client: Any = None  # redis.asyncio.Redis, lazily created

    async def _conn(self) -> Any:
        """Lazy connection — no I/O at __init__ time."""
        if self._client is None:
            try:
                import redis.asyncio as aioredis
            except ImportError as exc:
                raise ImportError(
                    "RedisHeap requires redis-py with asyncio support. "
                    'Install with: pip install "redis[asyncio]>=4.2"'
                ) from exc
            self._client = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
            )
        return self._client

    def _key(self, page_id: str) -> str:
        return f"{self._KEY_PREFIX}{page_id}"

    async def put(self, entry: PageHeapEntry) -> None:
        import json as _json

        client = await self._conn()
        key = self._key(entry.page_id)
        ttl = int(self._config.entry_ttl_seconds)
        await client.set(key, _json.dumps(entry.to_dict()), ex=ttl)

    async def get(self, page_id: str) -> Optional[PageHeapEntry]:
        import json as _json

        client = await self._conn()
        key = self._key(page_id)
        raw = await client.get(key)
        if raw is None:
            return None

        entry = PageHeapEntry.from_dict(_json.loads(raw))
        entry.touch()

        # Persist updated access metadata + reset TTL (LRU signal)
        ttl = int(self._config.entry_ttl_seconds)
        await client.set(key, _json.dumps(entry.to_dict()), ex=ttl)
        return entry

    async def delete(self, page_id: str) -> None:
        client = await self._conn()
        await client.delete(self._key(page_id))

    async def size(self) -> int:
        """Use SCAN so we don't block the Redis event loop on large keyspaces."""
        client = await self._conn()
        count = 0
        cursor = 0
        pattern = f"{self._KEY_PREFIX}*"
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=100)
            count += len(keys)
            if cursor == 0:
                break
        return count

    async def close(self) -> None:
        """Release the Redis connection.  Call on graph shutdown."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_heap(config: EvictionConfig, backend: str = "memory", **kwargs) -> AbstractHeap:
    """
    Instantiate the correct heap backend.

    Args:
        backend: "memory" (default) | "redis"
        **kwargs: passed through to the backend constructor
                  (e.g. redis_url="redis://...")
    """
    if backend == "memory":
        return InMemoryHeap(config)
    if backend == "redis":
        return RedisHeap(config, **kwargs)
    raise ValueError(f"Unknown heap backend: {backend!r}. Choose 'memory' or 'redis'.")
