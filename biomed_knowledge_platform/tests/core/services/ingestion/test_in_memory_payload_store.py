from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from biomed_platform.core.domains.ingestion import IngestItem
from biomed_platform.core.services.ingestion.in_memory_payload_store import InMemoryIngestPayloadStore


def _item(doi: str) -> IngestItem:
    return IngestItem(
        doi_original=doi,
        doi_normalized=doi,
        disease="d",
        source_type="t",
        content_text="x",
    )


@pytest.mark.asyncio
async def test_payload_store_put_get_delete_round_trip(monkeypatch) -> None:
    # Given a payload store and a stable clock
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = InMemoryIngestPayloadStore(ttl_seconds=10, max_jobs=10)
    monkeypatch.setattr(store, "_now", lambda: base)

    # When storing and reading back
    await store.put(job_id="j", items=[_item("a"), _item("b")])
    items = await store.get(job_id="j")

    # Then the payload is returned as a new list
    assert [it.doi_original for it in items] == ["a", "b"]
    assert items is not None

    # When deleting
    await store.delete(job_id="j")

    # Then subsequent get raises
    with pytest.raises(KeyError):
        await store.get(job_id="j")


@pytest.mark.asyncio
async def test_payload_store_ttl_expires_entries(monkeypatch) -> None:
    # Given a payload store with short ttl
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = InMemoryIngestPayloadStore(ttl_seconds=5, max_jobs=10)

    # When storing at base time
    monkeypatch.setattr(store, "_now", lambda: base)
    await store.put(job_id="j", items=[_item("a")])

    # When time advances beyond ttl
    later = base + timedelta(seconds=6)
    monkeypatch.setattr(store, "_now", lambda: later)

    # Then get raises because cleanup happens on access
    with pytest.raises(KeyError):
        await store.get(job_id="j")
