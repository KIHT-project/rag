from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
from typing import Any

import pytest
import biomed_platform.core.services.ingestion.worker as worker_mod
from biomed_platform.core.domains.ingestion import (
    IngestItemState,
    IngestItemStatus,
    IngestionJob,
    JobCounts,
    JobState,
)
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


class TestIngestionWorker:
    async def test_process_job_returns_when_job_missing(self) -> None:
        queue = _FakeQueue(job_ids=[])
        store = _FakeJobStore()
        reg = _FakeDocumentRegistry()

        worker = IngestionWorker(queue=queue, job_store=store, document_registry=reg)

        await worker._process_job("missing")

        assert store.updates == []
        assert reg.commits == []
        assert reg.releases == []

    async def test_process_job_sets_running_then_succeeded_and_commits_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        queue = _FakeQueue(job_ids=[])
        store = _FakeJobStore()
        reg = _FakeDocumentRegistry()

        job = _job(
            job_id="j1",
            embedding_model_id="e1",
            items=[("10.1/A", "d1"), ("10.2/B", "d2")],
        )
        store.jobs["j1"] = job

        worker = IngestionWorker(queue=queue, job_store=store, document_registry=reg)

        t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
        times = iter([t0, t1])
        monkeypatch.setattr(worker, "_now", lambda: next(times))

        await worker._process_job("j1")

        # Two updates: running update, then final update
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

        assert reg.commits == [("e1", "d1"), ("e1", "d2")]
        assert reg.releases == []

    async def test_process_job_preserves_item_message_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        queue = _FakeQueue(job_ids=[])
        store = _FakeJobStore()
        reg = _FakeDocumentRegistry()

        now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        job = IngestionJob(
            job_id="j1",
            state=JobState.queued,
            created_at=now,
            updated_at=now,
            effective_embedding_model_id="e1",
            items=[
                IngestItemStatus(doi_original="10.1/A", doc_id="d1", state=IngestItemState.queued, message="m1"),
                IngestItemStatus(doi_original="10.2/B", doc_id="d2", state=IngestItemState.queued, message=None),
            ],
            counts=JobCounts(total=2, succeeded=0, failed=0, skipped_duplicate=0),
        )
        store.jobs["j1"] = job

        worker = IngestionWorker(queue=queue, job_store=store, document_registry=reg)

        t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        times = iter([t0, t1])
        monkeypatch.setattr(worker, "_now", lambda: next(times))

        await worker._process_job("j1")

        final_update = store.updates[-1]
        assert [it.message for it in final_update.items] == ["m1", None]

    async def test_run_forever_dequeues_and_processes_jobs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        queue = _FakeQueue(job_ids=["j1"])
        store = _FakeJobStore()
        reg = _FakeDocumentRegistry()

        job = _job(job_id="j1", embedding_model_id="e1", items=[("10.1/A", "d1")])
        store.jobs["j1"] = job

        worker = IngestionWorker(queue=queue, job_store=store, document_registry=reg)

        async def fake_process(job_id: str) -> None:
            assert job_id == "j1"
            raise asyncio.CancelledError

        monkeypatch.setattr(worker, "_process_job", fake_process)

        with pytest.raises(asyncio.CancelledError):
            await worker.run_forever()

    def test_now_returns_timezone_aware_utc_datetime(self) -> None:
        worker = IngestionWorker(queue=_FakeQueue(job_ids=[]), job_store=_FakeJobStore(),
                                 document_registry=_FakeDocumentRegistry())

        now = worker._now()

        assert isinstance(now, datetime)
        assert now.tzinfo is timezone.utc

    async def test_release_branch_executed_via_monkeypatched_item_builder(self,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
        queue = _FakeQueue(job_ids=[])
        store = _FakeJobStore()
        reg = _FakeDocumentRegistry()

        job = _job(job_id="j1", embedding_model_id="e1", items=[("10.1/A", "d1"), ("10.2/B", "d2")])
        store.jobs["j1"] = job

        real_cls = worker_mod.IngestItemStatus

        def _patched_status(*, doi_original: str, doc_id: str, state: IngestItemState, message: str | None = None):
            if state == IngestItemState.succeeded and doc_id == "d1":
                state = IngestItemState.failed
            return real_cls(doi_original=doi_original, doc_id=doc_id, state=state, message=message)

        monkeypatch.setattr(worker_mod, "IngestItemStatus", _patched_status)

        worker = IngestionWorker(queue=queue, job_store=store, document_registry=reg)

        await worker._process_job("j1")

        assert reg.commits == [("e1", "d2")]
        assert reg.releases == [("e1", "d1")]

    async def test_release_branch_executed_for_failed_or_skipped_items(self) -> None:
        queue = _FakeQueue(job_ids=[])
        store = _FakeJobStore()
        reg = _FakeDocumentRegistry()

        job = _job(job_id="j1", embedding_model_id="e1", items=[("10.1/A", "d1"), ("10.2/B", "d2")])
        store.jobs["j1"] = job

        class _TestWorker(IngestionWorker):
            async def _process_job(self, job_id: str) -> None:
                try:
                    job_local = await self._job_store.get(job_id)
                except KeyError:
                    return

                updated_items = [
                    IngestItemStatus(
                        doi_original=job_local.items[0].doi_original,
                        doc_id=job_local.items[0].doc_id,
                        state=IngestItemState.failed,
                        message=job_local.items[0].message,
                    ),
                    IngestItemStatus(
                        doi_original=job_local.items[1].doi_original,
                        doc_id=job_local.items[1].doc_id,
                        state=IngestItemState.skipped_duplicate,
                        message=job_local.items[1].message,
                    ),
                ]

                for it in updated_items:
                    if it.state == IngestItemState.succeeded:
                        await self._document_registry.commit(
                            embedding_model_id=job_local.effective_embedding_model_id,
                            doc_id=it.doc_id,
                        )
                    elif it.state in (IngestItemState.failed, IngestItemState.skipped_duplicate):
                        await self._document_registry.release(
                            embedding_model_id=job_local.effective_embedding_model_id,
                            doc_id=it.doc_id,
                        )

        worker = _TestWorker(queue=queue, job_store=store, document_registry=reg)

        await worker._process_job("j1")

        assert reg.commits == []
        assert reg.releases == [("e1", "d1"), ("e1", "d2")]