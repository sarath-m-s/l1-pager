"""
Phase 3: RedisHeap tests.

Tests are automatically skipped if Redis is not running at localhost:6379.
Run a local Redis with:  docker run -p 6379:6379 redis:7-alpine

The suite covers:
  - put / get / delete round-trip
  - TTL expiry (using a 1-second TTL)
  - LRU metadata update on access
  - Heap size via SCAN
  - close() releases the connection
  - JSON wire format compatibility with InMemoryHeap (cross-language invariant)
"""
from __future__ import annotations

import asyncio
import json
import time
from uuid import uuid4

import pytest

from l1_pager import EvictionConfig, PageHeapEntry, build_heap
from l1_pager.heap import RedisHeap

REDIS_URL = "redis://localhost:6379"
KEY_PREFIX = "l1pager:heap:"


# ---------------------------------------------------------------------------
# Skip marker — checked once per module, not per test
# ---------------------------------------------------------------------------


def _redis_available() -> bool:
    async def _ping():
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(REDIS_URL)
            await client.ping()
            await client.aclose()
            return True
        except Exception:
            return False

    try:
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(_ping())
    except Exception:
        return False
    finally:
        loop.close()


REDIS_AVAILABLE = _redis_available()
skip_no_redis = pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available at localhost:6379")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_entry(page_id: str | None = None, token_count: int = 600) -> PageHeapEntry:
    return PageHeapEntry(
        page_id=page_id or str(uuid4())[:16],
        original_message_id=str(uuid4()),
        original_type="ToolMessage",
        raw_content=json.dumps({"transaction_id": "TXN-REDIS-001", "amount": 42.0}),
        tool_name="lookup",
        tool_call_id=str(uuid4()),
        token_count=token_count,
        evicted_at_turn=1,
    )


async def _cleanup(page_ids: list[str]) -> None:
    """Remove test keys from Redis."""
    import redis.asyncio as aioredis
    client = aioredis.from_url(REDIS_URL, decode_responses=True)
    keys = [f"{KEY_PREFIX}{pid}" for pid in page_ids]
    if keys:
        await client.delete(*keys)
    await client.aclose()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRedisHeap:
    @skip_no_redis
    @pytest.mark.asyncio
    async def test_put_and_get_roundtrip(self):
        config = EvictionConfig(entry_ttl_seconds=60)
        heap = RedisHeap(config, redis_url=REDIS_URL)
        entry = _make_entry()

        try:
            await heap.put(entry)
            retrieved = await heap.get(entry.page_id)

            assert retrieved is not None
            assert retrieved.page_id == entry.page_id
            assert retrieved.raw_content == entry.raw_content
            assert retrieved.token_count == entry.token_count
            assert retrieved.tool_name == entry.tool_name
        finally:
            await _cleanup([entry.page_id])
            await heap.close()

    @skip_no_redis
    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self):
        config = EvictionConfig(entry_ttl_seconds=60)
        heap = RedisHeap(config, redis_url=REDIS_URL)

        result = await heap.get("nonexistent00000")
        assert result is None
        await heap.close()

    @skip_no_redis
    @pytest.mark.asyncio
    async def test_delete_removes_entry(self):
        config = EvictionConfig(entry_ttl_seconds=60)
        heap = RedisHeap(config, redis_url=REDIS_URL)
        entry = _make_entry()

        try:
            await heap.put(entry)
            await heap.delete(entry.page_id)
            result = await heap.get(entry.page_id)
            assert result is None
        finally:
            await _cleanup([entry.page_id])
            await heap.close()

    @skip_no_redis
    @pytest.mark.asyncio
    async def test_access_increments_access_count(self):
        config = EvictionConfig(entry_ttl_seconds=60)
        heap = RedisHeap(config, redis_url=REDIS_URL)
        entry = _make_entry()

        try:
            await heap.put(entry)
            r1 = await heap.get(entry.page_id)
            r2 = await heap.get(entry.page_id)

            assert r1 is not None and r1.access_count == 1
            assert r2 is not None and r2.access_count == 2
        finally:
            await _cleanup([entry.page_id])
            await heap.close()

    @skip_no_redis
    @pytest.mark.asyncio
    async def test_put_is_idempotent(self):
        """Writing the same page_id twice must not create duplicate keys."""
        config = EvictionConfig(entry_ttl_seconds=60)
        heap = RedisHeap(config, redis_url=REDIS_URL)
        entry = _make_entry()
        updated = PageHeapEntry(
            page_id=entry.page_id,
            original_message_id=entry.original_message_id,
            original_type=entry.original_type,
            raw_content='{"updated": true}',
            tool_name=entry.tool_name,
            tool_call_id=entry.tool_call_id,
            token_count=entry.token_count,
            evicted_at_turn=entry.evicted_at_turn,
        )

        try:
            await heap.put(entry)
            await heap.put(updated)
            retrieved = await heap.get(entry.page_id)
            assert retrieved is not None
            assert json.loads(retrieved.raw_content).get("updated") is True
        finally:
            await _cleanup([entry.page_id])
            await heap.close()

    @skip_no_redis
    @pytest.mark.asyncio
    async def test_size_counts_heap_entries(self):
        config = EvictionConfig(entry_ttl_seconds=60)
        heap = RedisHeap(config, redis_url=REDIS_URL)
        entries = [_make_entry() for _ in range(5)]

        try:
            for e in entries:
                await heap.put(e)
            size = await heap.size()
            assert size >= 5  # may include other test keys
        finally:
            await _cleanup([e.page_id for e in entries])
            await heap.close()

    @skip_no_redis
    @pytest.mark.asyncio
    async def test_ttl_expiry(self):
        """Entries with very short TTL must not be returned after expiry."""
        config = EvictionConfig(entry_ttl_seconds=1)
        heap = RedisHeap(config, redis_url=REDIS_URL)
        entry = _make_entry()

        try:
            await heap.put(entry)
            # Verify it exists immediately
            assert await heap.get(entry.page_id) is not None
            # Wait for TTL to expire
            await asyncio.sleep(1.5)
            assert await heap.get(entry.page_id) is None
        finally:
            await _cleanup([entry.page_id])
            await heap.close()

    @skip_no_redis
    @pytest.mark.asyncio
    async def test_wire_format_matches_in_memory_heap(self):
        """
        Cross-language invariant: the JSON stored by RedisHeap must deserialise
        correctly via PageHeapEntry.from_dict().  This is the wire format shared
        with the TypeScript heap.
        """
        import redis.asyncio as aioredis

        config = EvictionConfig(entry_ttl_seconds=60)
        heap = RedisHeap(config, redis_url=REDIS_URL)
        entry = _make_entry()

        try:
            await heap.put(entry)

            # Read raw JSON directly from Redis (bypass our API)
            client = aioredis.from_url(REDIS_URL, decode_responses=True)
            raw = await client.get(f"{KEY_PREFIX}{entry.page_id}")
            await client.aclose()

            assert raw is not None
            wire = json.loads(raw)

            # Must round-trip through from_dict
            restored = PageHeapEntry.from_dict(wire)
            assert restored.page_id == entry.page_id
            assert restored.raw_content == entry.raw_content
            assert restored.original_type == entry.original_type

            # Wire format must use snake_case keys (cross-language invariant)
            assert "page_id" in wire
            assert "raw_content" in wire
            assert "original_message_id" in wire
        finally:
            await _cleanup([entry.page_id])
            await heap.close()

    @skip_no_redis
    @pytest.mark.asyncio
    async def test_build_heap_redis_backend(self):
        """build_heap('redis') must produce a working RedisHeap."""
        config = EvictionConfig(entry_ttl_seconds=60)
        heap = build_heap(config, backend="redis", redis_url=REDIS_URL)
        assert isinstance(heap, RedisHeap)

        entry = _make_entry()
        try:
            await heap.put(entry)
            r = await heap.get(entry.page_id)
            assert r is not None
        finally:
            await _cleanup([entry.page_id])
            if hasattr(heap, "close"):
                await heap.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
