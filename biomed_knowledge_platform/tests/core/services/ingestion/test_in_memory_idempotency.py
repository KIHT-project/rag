from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from biomed_platform.core.services.ingestion.in_memory_idempotency import InMemoryIdempotencyStore


@pytest.mark.asyncio
async def test_idempotency_put_then_get_hit(monkeypatch) -> None:
    # Given an idempotency store and a stable clock
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = InMemoryIdempotencyStore(ttl_seconds=10)
    monkeypatch.setattr(store, "_now", lambda: base)

    # When storing a record and reading it back with the same body hash
    await store.put(key="k", body_hash="h", job_id="j", created_at=base)
    job_id = await store.get_job_id(key="k", body_hash="h")

    # Then we get a hit
    assert job_id == "j"


@pytest.mark.asyncio
async def test_idempotency_get_miss_on_body_hash_mismatch(monkeypatch) -> None:
    # Given a stored record
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = InMemoryIdempotencyStore(ttl_seconds=10)
    monkeypatch.setattr(store, "_now", lambda: base)
    await store.put(key="k", body_hash="h1", job_id="j", created_at=base)

    # When querying with a different body hash
    job_id = await store.get_job_id(key="k", body_hash="h2")

    # Then we get a miss
    assert job_id is None


@pytest.mark.asyncio
async def test_idempotency_ttl_cleanup_expires_records(monkeypatch) -> None:
    # Given a store with a short ttl
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = InMemoryIdempotencyStore(ttl_seconds=5)

    # When storing a record at base time
    monkeypatch.setattr(store, "_now", lambda: base)
    await store.put(key="k", body_hash="h", job_id="j", created_at=base)

    # When time advances beyond ttl and we query
    later = base + timedelta(seconds=6)
    monkeypatch.setattr(store, "_now", lambda: later)
    job_id = await store.get_job_id(key="k", body_hash="h")

    # Then record is expired
    assert job_id is None
