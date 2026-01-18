from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from biomed_platform.common.middleware.trace import request_id_ctx
from biomed_platform.core.domains.ingestion import (
    IngestItem,
    IngestItemState,
    IngestItemStatus,
    JobCounts,
    JobState,
)
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.services.ingestion.worker import IngestionWorker


@dataclass(slots=True)
class _Job:
    job_id: str
    state: JobState
    effective_embedding_model_id: str
    items: list[IngestItemStatus]
    counts: JobCounts | None = None
    updated_at: datetime | None = None
    correlation_id: str | None = None


def _mk_queue(*, dequeue_side_effect) -> object:
    return SimpleNamespace(
        dequeue=AsyncMock(side_effect=dequeue_side_effect),
        size=lambda: 0,
        max_size=lambda: 10,
    )


@pytest.mark.anyio
async def test_run_forever_processes_one_job_then_cancelled() -> None:
    # Given
    queue = _mk_queue(dequeue_side_effect=["j1", asyncio.CancelledError()])
    job_store = AsyncMock()
    registry = AsyncMock()
    payload_store = AsyncMock()
    pipeline = AsyncMock()

    worker = IngestionWorker(
        queue=queue,
        job_store=job_store,
        document_registry=registry,
        payload_store=payload_store,
        pipeline=pipeline,
    )

    worker._process_job = AsyncMock()

    # When, Then
    with pytest.raises(asyncio.CancelledError):
        await worker.run_forever()

    worker._process_job.assert_awaited_once_with("j1")


@pytest.mark.anyio
async def test_run_forever_crashes_on_unexpected_exception() -> None:
    # Given
    queue = _mk_queue(dequeue_side_effect=RuntimeError("boom"))
    worker = IngestionWorker(
        queue=queue,
        job_store=AsyncMock(),
        document_registry=AsyncMock(),
        payload_store=AsyncMock(),
        pipeline=AsyncMock(),
    )

    # When, Then
    with pytest.raises(RuntimeError, match="boom"):
        await worker.run_forever()


@pytest.mark.anyio
async def test_process_job_skips_when_job_missing_and_deletes_payload() -> None:
    # Given
    queue = _mk_queue(dequeue_side_effect=["j1"])
    job_store = AsyncMock()
    job_store.get = AsyncMock(side_effect=KeyError("missing"))

    payload_store = AsyncMock()
    payload_store.delete = AsyncMock()

    worker = IngestionWorker(
        queue=queue,
        job_store=job_store,
        document_registry=AsyncMock(),
        payload_store=payload_store,
        pipeline=AsyncMock(),
    )

    assert request_id_ctx.get(None) is None

    # When
    await worker._process_job("j1")

    # Then
    payload_store.delete.assert_awaited_once_with(job_id="j1")
    assert request_id_ctx.get(None) is None


@pytest.mark.anyio
async def test_process_job_fails_when_payload_missing_and_marks_job_failed() -> None:
    # Given
    queue = _mk_queue(dequeue_side_effect=["j1"])
    registry = AsyncMock()
    registry.release = AsyncMock()

    items = [
        IngestItemStatus(
            doi_original="10.1/a",
            doc_id="doc1",
            state=IngestItemState.queued,
            message=None,
        )
    ]
    job = _Job(
        job_id="j1",
        state=JobState.queued,
        effective_embedding_model_id="m",
        items=list(items),
    )

    job_store = AsyncMock()
    job_store.get = AsyncMock(return_value=job)
    job_store.update = AsyncMock()

    payload_store = AsyncMock()
    payload_store.get = AsyncMock(side_effect=KeyError("no payload"))
    payload_store.delete = AsyncMock()

    worker = IngestionWorker(
        queue=queue,
        job_store=job_store,
        document_registry=registry,
        payload_store=payload_store,
        pipeline=AsyncMock(),
    )

    # When
    await worker._process_job("j1")

    # Then
    assert job.state == JobState.failed
    assert job.counts is not None
    assert job.counts.failed == 1
    assert job.items[0].state == IngestItemState.failed
    registry.release.assert_awaited()
    payload_store.delete.assert_awaited_with(job_id="j1")
    assert request_id_ctx.get(None) is None


@pytest.mark.anyio
async def test_process_job_happy_path_with_correlation_id_sets_context_and_succeeds() -> None:
    # Given
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    queue = _mk_queue(dequeue_side_effect=["j1"])
    registry = AsyncMock()
    registry.commit = AsyncMock()
    registry.release = AsyncMock()

    original_items = [
        IngestItemStatus(
            doi_original="10.1/a",
            doc_id="doc1",
            state=IngestItemState.queued,
            message=None,
        )
    ]
    job = _Job(
        job_id="j1",
        state=JobState.queued,
        effective_embedding_model_id="m",
        items=list(original_items),
        correlation_id="corr123",
    )

    job_store = AsyncMock()
    job_store.get = AsyncMock(return_value=job)
    job_store.update = AsyncMock()

    payload_store = AsyncMock()
    payload_store.get = AsyncMock(
        return_value=[
            IngestItem(
                doi_original="10.1/a",
                doi_normalized="10.1/a",
                title=None,
                journal=None,
                year=None,
                authors=[],
                disease=None,
                source_type=None,
                content_text="x",
            )
        ]
    )
    payload_store.delete = AsyncMock()

    pipeline = AsyncMock()
    pipeline.ingest_item = AsyncMock(return_value=None)

    worker = IngestionWorker(
        queue=queue,
        job_store=job_store,
        document_registry=registry,
        payload_store=payload_store,
        pipeline=pipeline,
    )

    worker._now = MagicMock(return_value=now)

    # When
    await worker._process_job("j1")

    # Then
    assert job.state == JobState.succeeded
    assert job.counts is not None
    assert job.counts.total == 1
    assert job.counts.succeeded == 1
    assert job.items[0].state == IngestItemState.succeeded
    registry.commit.assert_awaited()
    payload_store.delete.assert_awaited_with(job_id="j1")
    assert request_id_ctx.get(None) is None


@pytest.mark.anyio
async def test_process_items_handles_missing_doc_id_mapping_branch() -> None:
    # Given
    worker = IngestionWorker(
        queue=_mk_queue(dequeue_side_effect=["j"]),
        job_store=AsyncMock(),
        document_registry=AsyncMock(),
        payload_store=AsyncMock(),
        pipeline=AsyncMock(),
    )

    payload_items = [
        IngestItem(
            doi_original="10.1/a",
            doi_normalized="10.1/a",
            title=None,
            journal=None,
            year=None,
            authors=[],
            disease=None,
            source_type=None,
            content_text="x",
        )
    ]

    stats = SimpleNamespace(succeeded=0, failed=0, skipped=0)

    # When
    out = await worker._process_items(
        job_id="j",
        embedding_model_id="m",
        items=payload_items,
        doc_id_by_doi={},
        stats=stats,
    )

    # Then
    assert len(out) == 1
    assert out[0].state == IngestItemState.failed
    assert out[0].message == "missing_doc_id_mapping"
    assert stats.failed == 1


def test_build_doc_id_index_covers_duplicate_doi_branch() -> None:
    # Given
    worker = IngestionWorker(
        queue=_mk_queue(dequeue_side_effect=["j"]),
        job_store=AsyncMock(),
        document_registry=AsyncMock(),
        payload_store=AsyncMock(),
        pipeline=AsyncMock(),
    )

    job_items = [
        IngestItemStatus(doi_original="a", doc_id="d1", state=IngestItemState.queued, message=None),
        IngestItemStatus(doi_original="a", doc_id="d2", state=IngestItemState.queued, message=None),
        IngestItemStatus(doi_original="b", doc_id="", state=IngestItemState.queued, message=None),
    ]

    # When
    mapping = worker._build_doc_id_index(job_id="j", job_items=job_items)

    # Then
    assert mapping["a"] == "d2"
    assert mapping["b"] == ""


@pytest.mark.anyio
async def test_safe_commit_and_release_swallow_exceptions() -> None:
    # Given
    registry = AsyncMock()
    registry.commit = AsyncMock(side_effect=RuntimeError("c"))
    registry.release = AsyncMock(side_effect=RuntimeError("r"))

    worker = IngestionWorker(
        queue=_mk_queue(dequeue_side_effect=["j"]),
        job_store=AsyncMock(),
        document_registry=registry,
        payload_store=AsyncMock(),
        pipeline=AsyncMock(),
    )

    # When, Then
    await worker._safe_commit(job_embedding_model_id="m", doc_id="doc")
    await worker._safe_release(job_embedding_model_id="m", doc_id="doc")

    # And, empty doc id is a no op
    await worker._safe_commit(job_embedding_model_id="m", doc_id="")
    await worker._safe_release(job_embedding_model_id="m", doc_id="")


@pytest.mark.anyio
async def test_safe_payload_delete_swallow_exceptions() -> None:
    # Given
    payload_store = AsyncMock()
    payload_store.delete = AsyncMock(side_effect=RuntimeError("x"))

    worker = IngestionWorker(
        queue=_mk_queue(dequeue_side_effect=["j"]),
        job_store=AsyncMock(),
        document_registry=AsyncMock(),
        payload_store=payload_store,
        pipeline=AsyncMock(),
    )

    # When, Then
    await worker._safe_payload_delete("j")


@pytest.mark.anyio
async def test_load_job_or_skip_raises_on_job_store_exception() -> None:
    # Given, job store get raises unexpected exception
    job_store = AsyncMock()
    job_store.get = AsyncMock(side_effect=RuntimeError("boom"))

    worker = IngestionWorker(
        queue=_mk_queue(dequeue_side_effect=["j"]),
        job_store=job_store,
        document_registry=AsyncMock(),
        payload_store=AsyncMock(),
        pipeline=AsyncMock(),
    )

    # When, loading job
    # Then, exception is raised
    with pytest.raises(RuntimeError, match="boom"):
        await worker._load_job_or_skip("j1")


@pytest.mark.anyio
async def test_load_payload_or_fail_raises_on_payload_store_exception() -> None:
    # Given, payload store get raises unexpected exception
    payload_store = AsyncMock()
    payload_store.get = AsyncMock(side_effect=RuntimeError("boom"))

    worker = IngestionWorker(
        queue=_mk_queue(dequeue_side_effect=["j"]),
        job_store=AsyncMock(),
        document_registry=AsyncMock(),
        payload_store=payload_store,
        pipeline=AsyncMock(),
    )

    # When, loading payload
    # Then, exception is raised
    with pytest.raises(RuntimeError, match="boom"):
        await worker._load_payload_or_fail("j1")


@pytest.mark.anyio
async def test_mark_job_failed_missing_payload_handles_job_missing_and_job_store_exception() -> None:
    # Given, KeyError means job missing
    job_store = AsyncMock()
    job_store.get = AsyncMock(side_effect=KeyError("no job"))

    worker = IngestionWorker(
        queue=_mk_queue(dequeue_side_effect=["j"]),
        job_store=job_store,
        document_registry=AsyncMock(),
        payload_store=AsyncMock(),
        pipeline=AsyncMock(),
    )

    # When, marking failed for missing payload
    await worker._mark_job_failed_missing_payload(job_id="j1")

    # Then, returns without raising
    assert True

    # Given, unexpected exception from job store get
    job_store.get = AsyncMock(side_effect=RuntimeError("boom"))

    # When, marking failed for missing payload
    # Then, exception is raised
    with pytest.raises(RuntimeError, match="boom"):
        await worker._mark_job_failed_missing_payload(job_id="j2")


@pytest.mark.anyio
async def test_process_items_counts_succeeded_failed_and_skipped_paths() -> None:
    # Given, one item missing mapping, one succeeds, one skipped duplicate
    registry = AsyncMock()
    payload_store = AsyncMock()

    worker = IngestionWorker(
        queue=_mk_queue(dequeue_side_effect=["j"]),
        job_store=AsyncMock(),
        document_registry=registry,
        payload_store=payload_store,
        pipeline=AsyncMock(),
    )

    item_missing = IngestItem(
        doi_original="10.1/missing",
        doi_normalized="10.1/missing",
        title=None,
        journal=None,
        year=None,
        authors=[],
        disease=None,
        source_type=None,
        content_text="x",
    )
    item_ok = IngestItem(
        doi_original="10.1/ok",
        doi_normalized="10.1/ok",
        title=None,
        journal=None,
        year=None,
        authors=[],
        disease=None,
        source_type=None,
        content_text="x",
    )
    item_dup = IngestItem(
        doi_original="10.1/dup",
        doi_normalized="10.1/dup",
        title=None,
        journal=None,
        year=None,
        authors=[],
        disease=None,
        source_type=None,
        content_text="x",
    )

    async def _process_single_item(*, job_id: str, embedding_model_id: str, doc_id: str, item: IngestItem):
        if item.doi_original.endswith("/ok"):
            return IngestItemStatus(
                doi_original=item.doi_original,
                doc_id=doc_id,
                state=IngestItemState.succeeded,
                message=None,
            )
        return IngestItemStatus(
            doi_original=item.doi_original,
            doc_id=doc_id,
            state=IngestItemState.skipped_duplicate,
            message="already_indexed",
        )

    worker._process_single_item = AsyncMock(side_effect=_process_single_item)

    stats = SimpleNamespace(succeeded=0, failed=0, skipped=0)

    # When, processing items
    out = await worker._process_items(
        job_id="j1",
        embedding_model_id="m",
        items=[item_missing, item_ok, item_dup],
        doc_id_by_doi={
            "10.1/ok": "doc_ok",
            "10.1/dup": "doc_dup",
        },
        stats=stats,
    )

    # Then, it covers failed, succeeded, skipped branches
    assert [x.state for x in out] == [
        IngestItemState.failed,
        IngestItemState.succeeded,
        IngestItemState.skipped_duplicate,
    ]
    assert stats.failed == 1
    assert stats.succeeded == 1
    assert stats.skipped == 1


@pytest.mark.anyio
async def test_finalize_job_merges_missing_replacements_and_state_failed_branch() -> None:
    # Given, original items have two dois, processed provides only one replacement and no unexpected dois
    job = _Job(
        job_id="j1",
        state=JobState.running,
        effective_embedding_model_id="m",
        items=[
            IngestItemStatus(doi_original="a", doc_id="d1", state=IngestItemState.queued, message=None),
            IngestItemStatus(doi_original="b", doc_id="d2", state=IngestItemState.queued, message=None),
        ],
    )

    job_store = AsyncMock()
    job_store.get = AsyncMock(return_value=job)
    job_store.update = AsyncMock()

    worker = IngestionWorker(
        queue=_mk_queue(dequeue_side_effect=["j"]),
        job_store=job_store,
        document_registry=AsyncMock(),
        payload_store=AsyncMock(),
        pipeline=AsyncMock(),
    )

    processed = [
        IngestItemStatus(doi_original="a", doc_id="d1", state=IngestItemState.failed, message="x"),
    ]

    # When, finalize runs
    await worker._finalize_job(
        job_id="j1",
        original_items=list(job.items),
        processed_statuses=processed,
    )

    # Then, missing replacement keeps original for b, counts reflect failed with zero succeeded, final state failed
    assert job.items[0].doi_original == "a"
    assert job.items[0].state == IngestItemState.failed
    assert job.items[1].doi_original == "b"
    assert job.items[1].state == IngestItemState.queued

    assert job.counts is not None
    assert job.counts.total == 2
    assert job.counts.succeeded == 0
    assert job.counts.failed == 1
    assert job.counts.skipped_duplicate == 0

    assert job.state == JobState.failed
    job_store.update.assert_awaited_once()
