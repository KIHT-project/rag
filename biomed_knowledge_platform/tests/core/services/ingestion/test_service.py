from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest

import biomed_platform.core.services.ingestion.service as service_mod
from biomed_platform.core.domains.ingestion import (
    IngestBatchAccepted,
    IngestItemState,
    JobState,
)
from biomed_platform.core.errors.errors import AppError
from biomed_platform.core.services.ingestion.service import DefaultIngestionService
from biomed_platform.core.services.ingestion_ports import (
    BackpressurePolicy,
    DocumentRegistry,
    IdempotencyStore,
    IngestPayloadStore,
    IngestionJobStore,
    IngestionQueue,
)

pytestmark = pytest.mark.asyncio


@dataclass
class _PeekRec:
    job_id: str


class _ErrorJobStore:
    def __init__(self, err: AppError) -> None:
        self.err = err
        self.calls: list[str] = []

    async def create(self, job: Any) -> None:
        raise AssertionError("not used")

    async def get(self, job_id: str) -> Any:
        self.calls.append(job_id)
        raise self.err

    async def update(self, job: Any) -> None:
        raise AssertionError("not used")

    async def delete(self, job_id: str) -> None:
        raise AssertionError("not used")


class _NoopQueue:
    def max_size(self) -> int:
        return 1

    def size(self) -> int:
        return 0

    async def enqueue(self, job_id: str) -> None:
        raise AssertionError("not used")

    async def dequeue(self) -> str:
        raise AssertionError("not used")


class _NoopIdem:
    async def get_job_id(self, *, key: str, body_hash: str) -> str | None:
        raise AssertionError("not used")

    async def put(self, *, key: str, body_hash: str, job_id: str, created_at) -> None:
        raise AssertionError("not used")


class _NoopRegistry:
    async def reserve(self, *, embedding_model_id: str, doc_id: str) -> None:
        raise AssertionError("not used")

    async def commit(self, *, embedding_model_id: str, doc_id: str) -> None:
        raise AssertionError("not used")

    async def release(self, *, embedding_model_id: str, doc_id: str) -> None:
        raise AssertionError("not used")


class _NoopBackpressure:
    def retry_after(self, *, queue_depth: int, queue_max_size: int, worker_count: int):
        raise AssertionError("not used")


class _NoopPayloadStore:
    async def put(self, *, job_id: str, items: list[Any]) -> None:
        raise AssertionError("not used")

    async def get(self, job_id: str) -> list[Any]:
        raise AssertionError("not used")

    async def delete(self, job_id: str) -> None:
        raise AssertionError("not used")


class _FakeQueue:
    def __init__(self, *, max_size: int, fail_full: bool = False) -> None:
        self._max = max_size
        self._items: list[str] = []
        self._fail_full = fail_full

    def max_size(self) -> int:
        return self._max

    def size(self) -> int:
        return len(self._items)

    async def enqueue(self, job_id: str) -> None:
        if self._fail_full or len(self._items) >= self._max:
            raise asyncio.QueueFull()
        self._items.append(job_id)

    async def dequeue(self) -> str:
        return self._items.pop(0)


class _FakeJobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, Any] = {}
        self.created: list[str] = []
        self.deleted: list[str] = []

    async def create(self, job: Any) -> None:
        self.jobs[job.job_id] = job
        self.created.append(job.job_id)

    async def get(self, job_id: str) -> Any:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    async def update(self, job: Any) -> None:
        self.jobs[job.job_id] = job

    async def delete(self, job_id: str) -> None:
        self.jobs.pop(job_id, None)
        self.deleted.append(job_id)


class _FakeIdempotencyStore:
    def __init__(self) -> None:
        self.map: dict[tuple[str, str], str] = {}
        self.by_key: dict[str, _PeekRec] = {}
        self.put_calls: list[tuple[str, str, str, datetime]] = []

    async def get_job_id(self, *, key: str, body_hash: str) -> str | None:
        return self.map.get((key, body_hash))

    async def put(self, *, key: str, body_hash: str, job_id: str, created_at: datetime) -> None:
        self.map[(key, body_hash)] = job_id
        self.by_key[key] = _PeekRec(job_id=job_id)
        self.put_calls.append((key, body_hash, job_id, created_at))

    async def peek_record(self, *, key: str) -> _PeekRec | None:
        return self.by_key.get(key)


class _FakeDocumentRegistry:
    def __init__(self) -> None:
        self.reserved: dict[str, set[str]] = {}
        self.committed: dict[str, set[str]] = {}
        self.reserve_calls: list[tuple[str, str]] = []
        self.release_calls: list[tuple[str, str]] = []

    def _rspace(self, embedding_model_id: str) -> set[str]:
        return self.reserved.setdefault(embedding_model_id, set())

    def _cspace(self, embedding_model_id: str) -> set[str]:
        return self.committed.setdefault(embedding_model_id, set())

    async def reserve(self, *, embedding_model_id: str, doc_id: str) -> None:
        self.reserve_calls.append((embedding_model_id, doc_id))
        if doc_id in self._rspace(embedding_model_id) or doc_id in self._cspace(embedding_model_id):
            raise KeyError(doc_id)
        self._rspace(embedding_model_id).add(doc_id)

    async def commit(self, *, embedding_model_id: str, doc_id: str) -> None:
        self._rspace(embedding_model_id).discard(doc_id)
        self._cspace(embedding_model_id).add(doc_id)

    async def release(self, *, embedding_model_id: str, doc_id: str) -> None:
        self.release_calls.append((embedding_model_id, doc_id))
        self._rspace(embedding_model_id).discard(doc_id)


class _FakePayloadStore:
    def __init__(self) -> None:
        self.data: dict[str, list[Any]] = {}
        self.put_calls: list[tuple[str, int]] = []
        self.delete_calls: list[str] = []

    async def put(self, *, job_id: str, items: list[Any]) -> None:
        self.data[job_id] = list(items)
        self.put_calls.append((job_id, len(items)))

    async def get(self, job_id: str) -> list[Any]:
        return list(self.data[job_id])

    async def delete(self, job_id: str) -> None:
        self.data.pop(job_id, None)
        self.delete_calls.append(job_id)


class _FakeBackpressure:
    def __init__(self, *, seconds: int) -> None:
        self.seconds = seconds
        self.calls: list[tuple[int, int, int]] = []

    def retry_after(self, *, queue_depth: int, queue_max_size: int, worker_count: int):
        self.calls.append((queue_depth, queue_max_size, worker_count))
        return SimpleNamespace(seconds=self.seconds)


def _cmd(
    *,
    embedding_model_id: str = "e1",
    idempotency_key: str | None = None,
    body_hash: str = "bh1",
    items: list[tuple[str, str]] | None = None,
) -> Any:
    if items is None:
        items = [("10.1/a", "10.1/A"), ("10.2/b", "10.2/B")]
    cmd_items = [SimpleNamespace(doi_normalized=n, doi_original=o) for (n, o) in items]
    return SimpleNamespace(
        effective_embedding_model_id=embedding_model_id,
        idempotency_key=idempotency_key,
        body_hash=body_hash,
        items=cmd_items,
    )


def _svc(
    *,
    queue: _FakeQueue,
    store: _FakeJobStore,
    idem: _FakeIdempotencyStore,
    reg: _FakeDocumentRegistry,
    backpressure: _FakeBackpressure,
    worker_count: int,
    payload: _FakePayloadStore,
) -> DefaultIngestionService:
    return DefaultIngestionService(
        queue=cast(IngestionQueue, queue),
        job_store=cast(IngestionJobStore, store),
        idempotency_store=cast(IdempotencyStore, idem),
        document_registry=cast(DocumentRegistry, reg),
        backpressure=cast(BackpressurePolicy, backpressure),
        worker_count=worker_count,
        payload_store=cast(IngestPayloadStore, payload),
    )


class TestDefaultIngestionService:
    async def test_ingest_batch_idempotency_fast_path_returns_existing(self) -> None:
        # Given
        queue = _FakeQueue(max_size=10)
        store = _FakeJobStore()
        idem = _FakeIdempotencyStore()
        reg = _FakeDocumentRegistry()
        payload = _FakePayloadStore()
        backpressure = _FakeBackpressure(seconds=7)

        idem.map[("k1", "bh1")] = "existing_job"

        svc = _svc(
            queue=queue,
            store=store,
            idem=idem,
            reg=reg,
            backpressure=backpressure,
            worker_count=2,
            payload=payload,
        )

        # When
        got = await svc.ingest_batch(_cmd(idempotency_key="k1", body_hash="bh1"))

        # Then
        assert got == IngestBatchAccepted(job_id="existing_job", state=JobState.queued)
        assert reg.reserve_calls == []
        assert store.created == []
        assert queue.size() == 0
        assert idem.put_calls == []
        assert payload.put_calls == []
        assert payload.delete_calls == []

    async def test_ingest_batch_duplicate_doi_in_same_request_is_skipped_and_job_is_enqueued(
            self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        queue = _FakeQueue(max_size=10)
        store = _FakeJobStore()
        idem = _FakeIdempotencyStore()
        reg = _FakeDocumentRegistry()
        payload = _FakePayloadStore()
        backpressure = _FakeBackpressure(seconds=7)

        svc = _svc(
            queue=queue,
            store=store,
            idem=idem,
            reg=reg,
            backpressure=backpressure,
            worker_count=2,
            payload=payload,
        )

        fixed_now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        monkeypatch.setattr(svc, "_now", lambda: fixed_now)

        class _UUID:
            hex = "job_dup"

        monkeypatch.setattr(service_mod.uuid, "uuid4", lambda: _UUID)

        cmd = _cmd(items=[("10.1/a", "10.1/A"), ("10.1/a", "10.1/A again")])

        # When
        got = await svc.ingest_batch(cmd)

        # Then
        assert got == IngestBatchAccepted(job_id="job_dup", state=JobState.queued)

        assert store.created == ["job_dup"]
        assert store.deleted == []
        assert queue._items == ["job_dup"]

        assert len(reg.reserve_calls) == 1
        assert reg.release_calls == []

        assert payload.put_calls == [("job_dup", 1)]
        assert payload.delete_calls == []

        created_job = store.jobs["job_dup"]
        assert created_job.job_id == "job_dup"
        assert created_job.state == JobState.queued
        assert created_job.counts.total == 2
        assert created_job.counts.failed == 0
        assert created_job.counts.succeeded == 0
        assert created_job.counts.skipped_duplicate == 1

        assert len(created_job.items) == 2
        states = [it.state for it in created_job.items]
        assert states.count(IngestItemState.queued) == 1
        assert states.count(getattr(IngestItemState, "skipped_duplicate", IngestItemState.failed)) == 1

    async def test_ingest_batch_idempotency_conflict_rolls_back_and_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        queue = _FakeQueue(max_size=10)
        store = _FakeJobStore()
        idem = _FakeIdempotencyStore()
        reg = _FakeDocumentRegistry()
        payload = _FakePayloadStore()
        backpressure = _FakeBackpressure(seconds=7)

        svc = DefaultIngestionService(
            queue=queue,
            job_store=store,
            idempotency_store=idem,
            document_registry=reg,
            backpressure=backpressure,
            worker_count=2,
            payload_store=payload,
        )

        fixed_now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        monkeypatch.setattr(svc, "_now", lambda: fixed_now)

        class _UUID:
            hex = "job_conflict"

        monkeypatch.setattr(service_mod.uuid, "uuid4", lambda: _UUID)

        idem.by_key["k1"] = _PeekRec(job_id="other_job")

        # When
        with pytest.raises(AppError) as exc:
            await svc.ingest_batch(_cmd(idempotency_key="k1", body_hash="bh1"))

        # Then
        assert exc.value.code == "validation_error"
        assert store.created == ["job_conflict"]
        assert store.deleted == ["job_conflict"]
        assert payload.put_calls == [("job_conflict", 2)]
        assert payload.delete_calls == ["job_conflict"]
        assert queue.size() == 0
        assert len(reg.release_calls) >= 1
        assert idem.put_calls == []

    async def test_ingest_batch_queue_full_rolls_back_and_raises_with_retry_after(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        queue = _FakeQueue(max_size=1, fail_full=True)
        store = _FakeJobStore()
        idem = _FakeIdempotencyStore()
        reg = _FakeDocumentRegistry()
        payload = _FakePayloadStore()
        backpressure = _FakeBackpressure(seconds=13)

        svc = DefaultIngestionService(
            queue=queue,
            job_store=store,
            idempotency_store=idem,
            document_registry=reg,
            backpressure=backpressure,
            worker_count=5,
            payload_store=payload,
        )

        fixed_now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        monkeypatch.setattr(svc, "_now", lambda: fixed_now)

        class _UUID:
            hex = "job_qfull"

        monkeypatch.setattr(service_mod.uuid, "uuid4", lambda: _UUID)

        # When
        with pytest.raises(AppError) as exc:
            await svc.ingest_batch(_cmd(idempotency_key=None))

        # Then
        assert exc.value.code == "too_many_requests"
        assert store.deleted == ["job_qfull"]
        assert payload.put_calls == [("job_qfull", 2)]
        assert payload.delete_calls == ["job_qfull"]
        assert len(reg.release_calls) >= 1
        assert backpressure.calls
        assert exc.value.details and exc.value.details.get("retry_after_seconds") == 13
        assert idem.put_calls == []

    async def test_ingest_batch_success_creates_job_enqueues_and_puts_idempotency_after_enqueue(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        queue = _FakeQueue(max_size=10)
        store = _FakeJobStore()
        idem = _FakeIdempotencyStore()
        reg = _FakeDocumentRegistry()
        payload = _FakePayloadStore()
        backpressure = _FakeBackpressure(seconds=7)

        svc = _svc(
            queue=queue,
            store=store,
            idem=idem,
            reg=reg,
            backpressure=backpressure,
            worker_count=2,
            payload=payload,
        )

        fixed_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(svc, "_now", lambda: fixed_now)

        class _UUID:
            hex = "job_ok"

        monkeypatch.setattr(service_mod.uuid, "uuid4", lambda: _UUID)

        # When
        got = await svc.ingest_batch(_cmd(idempotency_key="k1", body_hash="bh1"))

        # Then
        assert got == IngestBatchAccepted(job_id="job_ok", state=JobState.queued)
        assert store.created == ["job_ok"]
        assert store.deleted == []
        assert queue._items == ["job_ok"]
        assert len(reg.reserve_calls) == 2
        assert idem.put_calls == [("k1", "bh1", "job_ok", fixed_now)]
        assert payload.put_calls == [("job_ok", 2)]
        assert payload.delete_calls == []

        created_job = store.jobs["job_ok"]
        assert created_job.job_id == "job_ok"
        assert created_job.state == JobState.queued
        assert created_job.counts.total == 2
        assert all(it.state == IngestItemState.queued for it in created_job.items)

    async def test_get_job_status_maps_missing_job_to_not_found_app_error(self) -> None:
        # Given
        queue = _FakeQueue(max_size=10)
        store = _FakeJobStore()
        idem = _FakeIdempotencyStore()
        reg = _FakeDocumentRegistry()
        payload = _FakePayloadStore()
        backpressure = _FakeBackpressure(seconds=7)

        svc = _svc(
            queue=queue,
            store=store,
            idem=idem,
            reg=reg,
            backpressure=backpressure,
            worker_count=2,
            payload=payload,
        )

        # When
        with pytest.raises(AppError) as exc:
            await svc.get_job_status(job_id="missing")

        # Then
        assert exc.value.code == "not_found"

    async def test_get_job_status_reraises_app_error_unchanged(self) -> None:
        # Given
        err = AppError(code="validation_error", message="boom", details={"x": 1}, retryable=False)
        store = _ErrorJobStore(err)

        svc = DefaultIngestionService(
            queue=cast(IngestionQueue, _NoopQueue()),
            job_store=cast(IngestionJobStore, store),
            idempotency_store=cast(IdempotencyStore, _NoopIdem()),
            document_registry=cast(DocumentRegistry, _NoopRegistry()),
            backpressure=cast(BackpressurePolicy, _NoopBackpressure()),
            worker_count=1,
            payload_store=cast(IngestPayloadStore, _NoopPayloadStore()),
        )

        # When
        with pytest.raises(AppError) as exc:
            await svc.get_job_status(job_id="j1")

        # Then
        assert exc.value is err
        assert store.calls == ["j1"]
