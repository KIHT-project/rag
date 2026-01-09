# tests/core/services/ingestion/test_in_memory_payload_store.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from biomed_platform.core.services.ingestion.in_memory_payload_store import (
    InMemoryIngestPayloadStore,
)


pytestmark = pytest.mark.asyncio


def _item(doi: str) -> Any:
    # Only attributes used by the store are list containment and copying.
    # Keep it lightweight and stable.
    return SimpleNamespace(doi_original=doi)


class TestInMemoryIngestPayloadStoreInit:
    def test_given_non_positive_ttl_when_init_then_raises_value_error(self) -> None:
        # Given, When, Then
        with pytest.raises(ValueError, match="ttl_seconds must be positive"):
            InMemoryIngestPayloadStore(ttl_seconds=0)

    def test_given_non_positive_max_jobs_when_init_then_raises_value_error(self) -> None:
        # Given, When, Then
        with pytest.raises(ValueError, match="max_jobs must be positive"):
            InMemoryIngestPayloadStore(max_jobs=0)


class TestInMemoryIngestPayloadStorePutGetDelete:
    async def test_given_put_then_get_returns_copy_not_same_list_object(self) -> None:
        # Given
        store = InMemoryIngestPayloadStore(ttl_seconds=60, max_jobs=10)
        items = [_item("10.1/A"), _item("10.2/B")]

        # When
        await store.put(job_id="j1", items=items)
        got1 = await store.get(job_id="j1")
        got2 = await store.get(job_id="j1")

        # Then
        assert [x.doi_original for x in got1] == ["10.1/A", "10.2/B"]
        assert [x.doi_original for x in got2] == ["10.1/A", "10.2/B"]
        assert got1 is not got2
        assert got1 is not items
        assert got2 is not items

    async def test_given_put_then_mutating_original_input_list_does_not_affect_stored_payload(self) -> None:
        # Given
        store = InMemoryIngestPayloadStore(ttl_seconds=60, max_jobs=10)
        items = [_item("10.1/A"), _item("10.2/B")]

        # When
        await store.put(job_id="j1", items=items)
        items.append(_item("10.3/C"))
        got = await store.get(job_id="j1")

        # Then
        assert [x.doi_original for x in got] == ["10.1/A", "10.2/B"]

    async def test_given_delete_existing_when_get_then_raises_key_error(self) -> None:
        # Given
        store = InMemoryIngestPayloadStore(ttl_seconds=60, max_jobs=10)
        await store.put(job_id="j1", items=[_item("10.1/A")])

        # When
        await store.delete(job_id="j1")

        # Then
        with pytest.raises(KeyError):
            await store.get(job_id="j1")

    async def test_given_delete_missing_when_called_then_no_error(self) -> None:
        # Given
        store = InMemoryIngestPayloadStore(ttl_seconds=60, max_jobs=10)

        # When, Then
        await store.delete(job_id="missing")


class TestInMemoryIngestPayloadStoreTtlCleanup:
    async def test_given_expired_payload_when_get_then_is_cleaned_and_raises_key_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        store = InMemoryIngestPayloadStore(ttl_seconds=10, max_jobs=10)

        t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t_expired = t0 + timedelta(seconds=11)

        times = iter([t0, t_expired])
        monkeypatch.setattr(store, "_now", lambda: next(times))

        await store.put(job_id="j1", items=[_item("10.1/A")])

        # When, Then
        with pytest.raises(KeyError):
            await store.get(job_id="j1")

    async def test_given_not_expired_payload_when_get_then_returns_items(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        store = InMemoryIngestPayloadStore(ttl_seconds=10, max_jobs=10)

        t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t_ok = t0 + timedelta(seconds=9)

        times = iter([t0, t_ok])
        monkeypatch.setattr(store, "_now", lambda: next(times))

        await store.put(job_id="j1", items=[_item("10.1/A")])

        # When
        got = await store.get(job_id="j1")

        # Then
        assert [x.doi_original for x in got] == ["10.1/A"]

    async def test_given_naive_created_at_in_store_when_cleanup_then_treated_as_utc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        store = InMemoryIngestPayloadStore(ttl_seconds=10, max_jobs=10)

        # Put once with real now, then inject a naive created_at to exercise the branch.
        t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(store, "_now", lambda: t0)
        await store.put(job_id="j1", items=[_item("10.1/A")])

        # Force naive created_at
        store._by_job["j1"] = store._by_job["j1"].__class__(  # type: ignore[attr-defined]
            items=store._by_job["j1"].items,  # type: ignore[attr-defined]
            created_at=datetime(2026, 1, 1, 0, 0, 0),  # naive
        )

        # Advance beyond TTL so cleanup should expire it even with naive timestamp
        t_expired = t0 + timedelta(seconds=11)
        monkeypatch.setattr(store, "_now", lambda: t_expired)

        # When, Then
        with pytest.raises(KeyError):
            await store.get(job_id="j1")


class TestInMemoryIngestPayloadStoreMaxJobsEviction:
    async def test_given_store_exceeds_max_jobs_when_put_then_evicts_oldest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        store = InMemoryIngestPayloadStore(ttl_seconds=9999, max_jobs=2)

        t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)

        seq = iter([t0, t1, t2])

        def _now() -> datetime:
            return next(seq, t2)

        monkeypatch.setattr(store, "_now", _now)

        await store.put(job_id="j0", items=[_item("10.0/A")])
        await store.put(job_id="j1", items=[_item("10.1/A")])

        # When
        await store.put(job_id="j2", items=[_item("10.2/A")])

        # Then
        with pytest.raises(KeyError):
            await store.get(job_id="j0")

        got1 = await store.get(job_id="j1")
        got2 = await store.get(job_id="j2")
        assert [x.doi_original for x in got1] == ["10.1/A"]
        assert [x.doi_original for x in got2] == ["10.2/A"]

    async def test_given_cleanup_eviction_order_uses_created_at_when_timestamps_are_out_of_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        store = InMemoryIngestPayloadStore(ttl_seconds=9999, max_jobs=2)

        t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)

        seq = iter([t0, t1, t2])

        def _now() -> datetime:
            return next(seq, t2)

        monkeypatch.setattr(store, "_now", _now)

        await store.put(job_id="a", items=[_item("a")])
        await store.put(job_id="b", items=[_item("b")])

        rec_a = store._by_job["a"]
        rec_b = store._by_job["b"]
        store._by_job["a"] = rec_a.__class__(items=rec_a.items, created_at=t1)
        store._by_job["b"] = rec_b.__class__(items=rec_b.items, created_at=t0)

        # When
        await store.put(job_id="c", items=[_item("c")])

        # Then
        with pytest.raises(KeyError):
            await store.get(job_id="b")

        assert [x.doi_original for x in await store.get(job_id="a")] == ["a"]
        assert [x.doi_original for x in await store.get(job_id="c")] == ["c"]
