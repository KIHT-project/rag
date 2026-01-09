from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from biomed_platform.core.domains.ingestion import (
    IngestItemState,
    IngestItemStatus,
    IngestionJob,
    JobCounts,
    JobState,
)
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.services.ingestion.worker import IngestionWorker

pytestmark = pytest.mark.asyncio


class _FakeQueue:
    def __init__(self, job_ids: list[str]) -> None:
        self._q: asyncio.Queue[str] = asyncio.Queue()
        for jid in job_ids:
            self._q.put_nowait(jid)

    def max_size(self) -> int:
        return 10

    def size(self) -> int:
        return self._q.qsize()

    async def enqueue(self, job_id: str) -> None:
        self._q.put_nowait(job_id)

    async def dequeue(self) -> str:
        return await self._q.get()


class _FakeJobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, Any] = {}
        self.updates: list[Any] = []

    async def get(self, job_id: str) -> Any:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    async def update(self, job: Any) -> None:
        self.jobs[job.job_id] = job
        self.updates.append(copy.deepcopy(job))


class _FakeDocumentRegistry:
    def __init__(self) -> None:
        self.commits: list[tuple[str, str]] = []
        self.releases: list[tuple[str, str]] = []

    async def reserve(self, *, embedding_model_id: str, doc_id: str) -> None:
        raise NotImplementedError

    async def commit(self, *, embedding_model_id: str, doc_id: str) -> None:
        self.commits.append((embedding_model_id, doc_id))

    async def release(self, *, embedding_model_id: str, doc_id: str) -> None:
        self.releases.append((embedding_model_id, doc_id))


class _FakePayloadStore:
    def __init__(self) -> None:
        self.payload_by_job: dict[str, list[Any]] = {}
        self.get_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.fail_get_missing: set[str] = set()
        self.fail_delete: bool = False

    async def put(self, *, job_id: str, items: list[Any]) -> None:
        self.payload_by_job[job_id] = list(items)

    async def get(self, *, job_id: str) -> list[Any]:
        self.get_calls.append(job_id)
        if job_id in self.fail_get_missing:
            raise KeyError(job_id)
        if job_id not in self.payload_by_job:
            raise KeyError(job_id)
        return list(self.payload_by_job[job_id])

    async def delete(self, *, job_id: str) -> None:
        self.delete_calls.append(job_id)
        if self.fail_delete:
            raise RuntimeError("delete_failed")
        self.payload_by_job.pop(job_id, None)


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, Any]] = []
        self.fail_by_doc_id: dict[str, Exception] = {}

    async def ingest_item(
        self,
        *,
        job_id: str,
        embedding_model_id: str,
        doc_id: str,
        item: Any,
    ) -> None:
        self.calls.append((job_id, embedding_model_id, doc_id, item))
        exc = self.fail_by_doc_id.get(doc_id)
        if exc is not None:
            raise exc


def _job(*, job_id: str, embedding_model_id: str, items: list[tuple[str, str]]) -> IngestionJob:
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    statuses = [
        IngestItemStatus(
            doi_original=doi,
            doc_id=doc_id,
            state=IngestItemState.queued,
            message=None,
        )
        for (doi, doc_id) in items
    ]
    return IngestionJob(
        job_id=job_id,
        state=JobState.queued,
        created_at=now,
        updated_at=now,
        effective_embedding_model_id=embedding_model_id,
        items=statuses,
        counts=JobCounts(total=len(statuses), succeeded=0, failed=0, skipped_duplicate=0),
    )


def _payload_items(dois: list[str]) -> list[Any]:
    return [SimpleNamespace(doi_original=d) for d in dois]


class TestIngestionWorker:
    async def test_process_job_returns_when_job_missing_and_deletes_payload_best_effort(self) -> None:
        queue = _FakeQueue(job_ids=[])
        store = _FakeJobStore()
        reg = _FakeDocumentRegistry()
        payload = _FakePayloadStore()
        pipeline = _FakePipeline()

        worker = IngestionWorker(
            queue=queue,
            job_store=store,
            document_registry=reg,
            payload_store=payload,
            pipeline=pipeline,
        )

        await worker._process_job("missing")

        assert store.updates == []
        assert reg.commits == []
        assert reg.releases == []
        assert payload.delete_calls == ["missing"]

    async def test_process_job_marks_failed_when_payload_missing_and_releases_all_docs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        queue = _FakeQueue(job_ids=[])
        store = _FakeJobStore()
        reg = _FakeDocumentRegistry()
        payload = _FakePayloadStore()
        pipeline = _FakePipeline()

        job = _job(job_id="j1", embedding_model_id="e1", items=[("10.1/A", "d1"), ("10.2/B", "d2")])
        store.jobs["j1"] = job
        payload.fail_get_missing.add("j1")

        worker = IngestionWorker(
            queue=queue,
            job_store=store,
            document_registry=reg,
            payload_store=payload,
            pipeline=pipeline,
        )

        t0 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        monkeypatch.setattr(worker, "_now", lambda: t0)

        await worker._process_job("j1")

        assert len(store.updates) == 1
        upd = store.updates[0]
        assert upd.state == JobState.failed
        assert upd.updated_at == t0
        assert upd.counts == JobCounts(total=2, succeeded=0, failed=2, skipped_duplicate=0)
        assert [it.state for it in upd.items] == [IngestItemState.failed, IngestItemState.failed]
        assert [it.message for it in upd.items] == ["missing_payload", "missing_payload"]

        assert reg.commits == []
        assert reg.releases == [("e1", "d1"), ("e1", "d2")]
        assert payload.delete_calls == ["j1"]

    async def test_process_job_sets_running_then_succeeded_commits_all_and_deletes_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        queue = _FakeQueue(job_ids=[])
        store = _FakeJobStore()
        reg = _FakeDocumentRegistry()
        payload = _FakePayloadStore()
        pipeline = _FakePipeline()

        job = _job(
            job_id="j1",
            embedding_model_id="e1",
            items=[("10.1/A", "d1"), ("10.2/B", "d2")],
        )
        store.jobs["j1"] = job
        payload.payload_by_job["j1"] = _payload_items(["10.1/A", "10.2/B"])

        worker = IngestionWorker(
            queue=queue,
            job_store=store,
            document_registry=reg,
            payload_store=payload,
            pipeline=pipeline,
        )

        t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 0, 0, 9, tzinfo=timezone.utc)
        times = iter([t0, t1, t2])
        monkeypatch.setattr(worker, "_now", lambda: next(times))

        await worker._process_job("j1")

        assert len(store.updates) == 2

        running_update = store.updates[0]
        assert running_update.state == JobState.running
        assert running_update.updated_at == t0
        assert all(it.state == IngestItemState.running for it in running_update.items)

        final_update = store.updates[1]
        assert final_update.state == JobState.succeeded
        assert final_update.updated_at == t1
        assert final_update.counts == JobCounts(total=2, succeeded=2, failed=0, skipped_duplicate=0)
        assert all(it.state == IngestItemState.succeeded for it in final_update.items)

        assert pipeline.calls and len(pipeline.calls) == 2
        assert reg.commits == [("e1", "d1"), ("e1", "d2")]
        assert reg.releases == []
        assert payload.delete_calls == ["j1"]

    async def test_process_job_partial_when_pipeline_system_error_releases_failed_doc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        queue = _FakeQueue(job_ids=[])
        store = _FakeJobStore()
        reg = _FakeDocumentRegistry()
        payload = _FakePayloadStore()
        pipeline = _FakePipeline()

        job = _job(
            job_id="j1",
            embedding_model_id="e1",
            items=[("10.1/A", "d1"), ("10.2/B", "d2")],
        )
        store.jobs["j1"] = job
        payload.payload_by_job["j1"] = _payload_items(["10.1/A", "10.2/B"])

        pipeline.fail_by_doc_id["d1"] = SystemError(
            code="boom",
            message="x",
            details=None,
            retryable=False,
        )

        worker = IngestionWorker(
            queue=queue,
            job_store=store,
            document_registry=reg,
            payload_store=payload,
            pipeline=pipeline,
        )

        t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 0, 0, 4, tzinfo=timezone.utc)
        times = iter([t0, t1, t2])
        monkeypatch.setattr(worker, "_now", lambda: next(times))

        await worker._process_job("j1")

        assert len(store.updates) == 2
        final_update = store.updates[1]
        assert final_update.state == JobState.partial
        assert final_update.counts == JobCounts(total=2, succeeded=1, failed=1, skipped_duplicate=0)

        by_doi = {it.doi_original: it for it in final_update.items}
        assert by_doi["10.1/A"].state == IngestItemState.failed
        assert by_doi["10.1/A"].message == "boom"
        assert by_doi["10.2/B"].state == IngestItemState.succeeded

        assert reg.commits == [("e1", "d2")]
        assert reg.releases == [("e1", "d1")]
        assert payload.delete_calls == ["j1"]

    async def test_process_job_fails_item_when_doc_id_mapping_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        queue = _FakeQueue(job_ids=[])
        store = _FakeJobStore()
        reg = _FakeDocumentRegistry()
        payload = _FakePayloadStore()
        pipeline = _FakePipeline()

        job = _job(
            job_id="j1",
            embedding_model_id="e1",
            items=[("10.1/A", "d1")],
        )
        store.jobs["j1"] = job

        payload.payload_by_job["j1"] = _payload_items(["10.404/MISSING"])

        worker = IngestionWorker(
            queue=queue,
            job_store=store,
            document_registry=reg,
            payload_store=payload,
            pipeline=pipeline,
        )

        t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
        times = iter([t0, t1, t2])
        monkeypatch.setattr(worker, "_now", lambda: next(times))

        await worker._process_job("j1")

        final_update = store.updates[1]
        assert final_update.state == JobState.failed
        assert final_update.counts == JobCounts(total=1, succeeded=0, failed=1, skipped_duplicate=0)

        assert final_update.items[0].doi_original == "10.1/A"
        assert final_update.items[0].state == IngestItemState.running or final_update.items[0].state == IngestItemState.queued

        assert pipeline.calls == []
        assert reg.commits == []
        assert reg.releases == []
        assert payload.delete_calls == ["j1"]

    async def test_run_forever_dequeues_and_processes_jobs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        queue = _FakeQueue(job_ids=["j1"])
        store = _FakeJobStore()
        reg = _FakeDocumentRegistry()
        payload = _FakePayloadStore()
        pipeline = _FakePipeline()

        job = _job(job_id="j1", embedding_model_id="e1", items=[("10.1/A", "d1")])
        store.jobs["j1"] = job
        payload.payload_by_job["j1"] = _payload_items(["10.1/A"])

        worker = IngestionWorker(
            queue=queue,
            job_store=store,
            document_registry=reg,
            payload_store=payload,
            pipeline=pipeline,
        )

        async def fake_process(job_id: str) -> None:
            assert job_id == "j1"
            raise asyncio.CancelledError

        monkeypatch.setattr(worker, "_process_job", fake_process)

        with pytest.raises(asyncio.CancelledError):
            await worker.run_forever()

    def test_now_returns_timezone_aware_utc_datetime(self) -> None:
        worker = IngestionWorker(
            queue=_FakeQueue(job_ids=[]),
            job_store=_FakeJobStore(),
            document_registry=_FakeDocumentRegistry(),
            payload_store=_FakePayloadStore(),
            pipeline=_FakePipeline(),
        )

        now = worker._now()

        assert isinstance(now, datetime)
        assert now.tzinfo is timezone.utc
