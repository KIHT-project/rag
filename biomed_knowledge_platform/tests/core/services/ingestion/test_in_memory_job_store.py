from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from biomed_platform.core.domains.ingestion import IngestionJob, JobState
from biomed_platform.core.services.ingestion.in_memory_job_store import InMemoryIngestionJobStore


@pytest.mark.asyncio
async def test_job_store_create_get_update_delete_happy_path(monkeypatch) -> None:
    # Given a job store and a stable clock
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = InMemoryIngestionJobStore(ttl_seconds_after_completion=10, max_jobs=10)
    monkeypatch.setattr(store, "_now", lambda: base)

    job = IngestionJob(
        job_id="j",
        state=JobState.queued,
        created_at=base,
        updated_at=base,
        effective_embedding_model_id="m",
        items=[],
        counts=None,
        correlation_id=None,
    )

    # When creating and fetching a job
    await store.create(job)
    loaded = await store.get("j")

    # Then the job is returned
    assert loaded.job_id == "j"
    assert loaded.state == JobState.queued

    # When updating the job to succeeded
    updated = IngestionJob(
        job_id="j",
        state=JobState.succeeded,
        created_at=base,
        updated_at=base,
        effective_embedding_model_id="m",
        items=[],
        counts=None,
        correlation_id=None,
    )
    await store.update(updated)
    loaded2 = await store.get("j")

    # Then the job reflects the update
    assert loaded2.state == JobState.succeeded

    # When deleting the job
    await store.delete("j")

    # Then subsequent get fails
    with pytest.raises(KeyError):
        await store.get("j")


@pytest.mark.asyncio
async def test_job_store_ttl_cleanup_removes_completed_jobs(monkeypatch) -> None:
    # Given a job store with a short ttl and a stable clock
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = InMemoryIngestionJobStore(ttl_seconds_after_completion=5, max_jobs=10)

    job = IngestionJob(
        job_id="j",
        state=JobState.succeeded,
        created_at=base,
        updated_at=base,
        effective_embedding_model_id="m",
        items=[],
        counts=None,
        correlation_id=None,
    )

    # When creating a completed job at base time
    monkeypatch.setattr(store, "_now", lambda: base)
    await store.create(job)

    # When time advances beyond ttl and we call get
    later = base + timedelta(seconds=6)
    monkeypatch.setattr(store, "_now", lambda: later)

    # Then job is removed and get raises
    with pytest.raises(KeyError):
        await store.get("j")
