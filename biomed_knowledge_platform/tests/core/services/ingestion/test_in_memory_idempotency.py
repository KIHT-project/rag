from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from biomed_platform.core.services.ingestion import InMemoryIdempotencyStore
import biomed_platform.core.services.ingestion.in_memory_idempotency as idem_mod
from biomed_platform.core.services.ingestion.in_memory_idempotency import _IdemRecord


pytestmark = pytest.mark.asyncio

class _FakeDateTime:
    @staticmethod
    def now(tz: Any = None) -> datetime:
        assert tz is timezone.utc
        return datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


class TestInMemoryIdempotencyStore:
    def test_init_rejects_non_positive_ttl(self) -> None:
        with pytest.raises(ValueError):
            InMemoryIdempotencyStore(ttl_seconds=0)

        with pytest.raises(ValueError):
            InMemoryIdempotencyStore(ttl_seconds=-1)

    async def test_get_job_id_returns_none_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = InMemoryIdempotencyStore(ttl_seconds=60)
        monkeypatch.setattr(store, "_now", lambda: datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc))

        got = await store.get_job_id(key="k1", body_hash="h1")

        assert got is None

    async def test_put_then_get_job_id_returns_job_id_when_body_hash_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = InMemoryIdempotencyStore(ttl_seconds=60)

        base_now = datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
        monkeypatch.setattr(store, "_now", lambda: base_now)

        created_at = base_now
        await store.put(key="k1", body_hash="h1", job_id="j1", created_at=created_at)

        got = await store.get_job_id(key="k1", body_hash="h1")

        assert got == "j1"

    async def test_get_job_id_returns_none_when_body_hash_differs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = InMemoryIdempotencyStore(ttl_seconds=60)

        base_now = datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
        monkeypatch.setattr(store, "_now", lambda: base_now)

        await store.put(key="k1", body_hash="h1", job_id="j1", created_at=base_now)

        got = await store.get_job_id(key="k1", body_hash="h2")

        assert got is None

    async def test_get_job_id_cleans_up_expired_records(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = InMemoryIdempotencyStore(ttl_seconds=10)

        base_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(store, "_now", lambda: base_now)

        await store.put(key="k1", body_hash="h1", job_id="j1", created_at=base_now)

        monkeypatch.setattr(store, "_now", lambda: base_now + timedelta(seconds=11))

        got = await store.get_job_id(key="k1", body_hash="h1")

        assert got is None
        assert await store.peek_record(key="k1") is None

    async def test_put_cleans_up_expired_records_before_inserting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = InMemoryIdempotencyStore(ttl_seconds=10)

        base_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(store, "_now", lambda: base_now)

        await store.put(key="k_old", body_hash="h_old", job_id="j_old", created_at=base_now)

        monkeypatch.setattr(store, "_now", lambda: base_now + timedelta(seconds=11))

        await store.put(
            key="k_new",
            body_hash="h_new",
            job_id="j_new",
            created_at=base_now + timedelta(seconds=11),
        )

        assert await store.peek_record(key="k_old") is None
        assert await store.peek_record(key="k_new") == _IdemRecord(
            body_hash="h_new",
            job_id="j_new",
            created_at=base_now + timedelta(seconds=11),
        )

    async def test_put_normalizes_naive_created_at_to_utc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = InMemoryIdempotencyStore(ttl_seconds=60)

        base_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(store, "_now", lambda: base_now)

        naive_created_at = datetime(2026, 1, 1, 0, 0, 0)
        await store.put(key="k1", body_hash="h1", job_id="j1", created_at=naive_created_at)

        rec = await store.peek_record(key="k1")
        assert rec is not None
        assert rec.created_at.tzinfo == timezone.utc
        assert rec.created_at == naive_created_at.replace(tzinfo=timezone.utc)

    async def test_expiration_handles_naive_record_created_at(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = InMemoryIdempotencyStore(ttl_seconds=10)

        base_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        store._by_key["k1"] = _IdemRecord(
            body_hash="h1",
            job_id="j1",
            created_at=datetime(2026, 1, 1, 0, 0, 0),
        )

        monkeypatch.setattr(store, "_now", lambda: base_now + timedelta(seconds=11))

        got = await store.get_job_id(key="k1", body_hash="h1")

        assert got is None
        assert await store.peek_record(key="k1") is None

    async def test_peek_record_returns_without_cleanup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = InMemoryIdempotencyStore(ttl_seconds=10)

        base_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(store, "_now", lambda: base_now)

        await store.put(key="k1", body_hash="h1", job_id="j1", created_at=base_now)

        monkeypatch.setattr(store, "_now", lambda: base_now + timedelta(seconds=11))

        rec = await store.peek_record(key="k1")
        assert rec is not None
        assert rec.job_id == "j1"

        got = await store.get_job_id(key="k1", body_hash="h1")
        assert got is None
        assert await store.peek_record(key="k1") is None

    def test_now_calls_datetime_now_with_utc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(idem_mod, "datetime", _FakeDateTime)

        store = InMemoryIdempotencyStore(ttl_seconds=60)

        got = store._now()

        assert got == datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
