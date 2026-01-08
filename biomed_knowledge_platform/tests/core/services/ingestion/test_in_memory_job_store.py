from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from biomed_platform.core.services.ingestion.in_memory_job_store import InMemoryIngestionJobStore


pytestmark = pytest.mark.asyncio


def _make_job(*, job_id: str, state_value: str) -> Any:
    """
    Avoid depending on the exact IngestionJob constructor.
    We only need .job_id and .state.value for this store.
    """
    state = type("State", (), {"value": state_value})()
    return type("Job", (), {"job_id": job_id, "state": state})()


class TestInMemoryIngestionJobStore:
    def test_init_rejects_non_positive_ttl(self) -> None:
        with pytest.raises(ValueError):
            InMemoryIngestionJobStore(ttl_seconds_after_completion=0)

        with pytest.raises(ValueError):
            InMemoryIngestionJobStore(ttl_seconds_after_completion=-1)

    def test_init_rejects_non_positive_max_jobs(self) -> None:
        with pytest.raises(ValueError):
            InMemoryIngestionJobStore(max_jobs=0)

        with pytest.raises(ValueError):
            InMemoryIngestionJobStore(max_jobs=-10)

    async def test_create_and_get_roundtrip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = InMemoryIngestionJobStore(ttl_seconds_after_completion=60, max_jobs=100)

        base_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(store, "_now", lambda: base_now)

        job = _make_job(job_id="j1", state_value="queued")

        await store.create(job)
        got = await store.get("j1")

        assert got is job

    async def test_get_raises_key_error_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = InMemoryIngestionJobStore(ttl_seconds_after_completion=60, max_jobs=100)
        monkeypatch.setattr(store, "_now", lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))

        with pytest.raises(KeyError) as exc:
            await store.get("missing")

        assert exc.value.args == ("missing",)

    async def test_delete_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = InMemoryIngestionJobStore(ttl_seconds_after_completion=60, max_jobs=100)
        monkeypatch.setattr(store, "_now", lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))

        await store.delete("j1")
        await store.delete("j1")

        with pytest.raises(KeyError):
            await store.get("j1")

    async def test_update_overwrites_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = InMemoryIngestionJobStore(ttl_seconds_after_completion=60, max_jobs=100)
        base_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(store, "_now", lambda: base_now)

        job1 = _make_job(job_id="j1", state_value="queued")
        job2 = _make_job(job_id="j1", state_value="running")

        await store.create(job1)
        await store.update(job2)

        got = await store.get("j1")
        assert got is job2

    async def test_completed_jobs_expire_after_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = InMemoryIngestionJobStore(ttl_seconds_after_completion=10, max_jobs=100)

        base_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(store, "_now", lambda: base_now)

        completed = _make_job(job_id="j_done", state_value="succeeded")
        await store.create(completed)

        monkeypatch.setattr(store, "_now", lambda: base_now + timedelta(seconds=11))

        with pytest.raises(KeyError):
            await store.get("j_done")

    async def test_in_flight_jobs_do_not_expire_by_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = InMemoryIngestionJobStore(ttl_seconds_after_completion=10, max_jobs=100)

        base_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(store, "_now", lambda: base_now)

        inflight = _make_job(job_id="j_inflight", state_value="running")
        await store.create(inflight)

        monkeypatch.setattr(store, "_now", lambda: base_now + timedelta(seconds=10_000))

        got = await store.get("j_inflight")
        assert got is inflight

    @pytest.mark.parametrize("state_value", ["succeeded", "failed", "partial"])
    async def test_create_sets_completed_at_for_completed_states(
        self, monkeypatch: pytest.MonkeyPatch, state_value: str
    ) -> None:
        store = InMemoryIngestionJobStore(ttl_seconds_after_completion=60, max_jobs=100)

        base_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(store, "_now", lambda: base_now)

        job = _make_job(job_id=f"j_{state_value}", state_value=state_value)
        await store.create(job)

        rec = store._jobs[job.job_id]
        assert rec.completed_at == base_now

    @pytest.mark.parametrize("state_value", ["queued", "running", "accepted"])
    async def test_create_does_not_set_completed_at_for_non_completed_states(
        self, monkeypatch: pytest.MonkeyPatch, state_value: str
    ) -> None:
        store = InMemoryIngestionJobStore(ttl_seconds_after_completion=60, max_jobs=100)

        base_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(store, "_now", lambda: base_now)

        job = _make_job(job_id=f"j_{state_value}", state_value=state_value)
        await store.create(job)

        rec = store._jobs[job.job_id]
        assert rec.completed_at is None

    async def test_update_marks_completed_at_when_state_transitions_to_completed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = InMemoryIngestionJobStore(ttl_seconds_after_completion=60, max_jobs=100)

        base_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(store, "_now", lambda: base_now)

        job_running = _make_job(job_id="j1", state_value="running")
        await store.create(job_running)

        later = base_now + timedelta(seconds=5)
        monkeypatch.setattr(store, "_now", lambda: later)

        job_done = _make_job(job_id="j1", state_value="succeeded")
        await store.update(job_done)

        rec = store._jobs["j1"]
        assert rec.completed_at == later

    async def test_max_jobs_enforced_on_next_operation_by_evicting_oldest_completed_only(
            self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = InMemoryIngestionJobStore(ttl_seconds_after_completion=10_000, max_jobs=2)

        t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=1)
        t2 = t0 + timedelta(seconds=2)

        monkeypatch.setattr(store, "_now", lambda: t0)
        await store.create(_make_job(job_id="c0", state_value="succeeded"))

        monkeypatch.setattr(store, "_now", lambda: t1)
        await store.create(_make_job(job_id="c1", state_value="failed"))

        monkeypatch.setattr(store, "_now", lambda: t2)
        await store.create(_make_job(job_id="inflight", state_value="running"))

        # At this point, cleanup has not run after the 3rd insert.
        assert len(store._jobs) == 3

        # Trigger cleanup, any store operation does it.
        # Use get on an existing job so it does not raise.
        await store.get("inflight")

        # Now max_jobs is enforced, oldest completed is evicted first.
        assert len(store._jobs) == 2
        assert "inflight" in store._jobs
        assert "c0" not in store._jobs
        assert "c1" in store._jobs

    async def test_cleanup_evicts_completed_with_naive_completed_at(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = InMemoryIngestionJobStore(ttl_seconds_after_completion=10, max_jobs=100)

        # Create a completed job first so the record exists, then force completed_at naive
        base_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(store, "_now", lambda: base_now)

        await store.create(_make_job(job_id="j1", state_value="succeeded"))
        store._jobs["j1"] = store._jobs["j1"].__class__(job=store._jobs["j1"].job, completed_at=datetime(2026, 1, 1, 0, 0, 0))

        monkeypatch.setattr(store, "_now", lambda: base_now + timedelta(seconds=11))

        with pytest.raises(KeyError):
            await store.get("j1")

    def test_now_returns_timezone_aware_utc_datetime(self) -> None:
        store = InMemoryIngestionJobStore(ttl_seconds_after_completion=60, max_jobs=10)

        before = datetime.now(timezone.utc)
        got = store._now()
        after = datetime.now(timezone.utc)

        assert got.tzinfo == timezone.utc
        assert before <= got <= after
