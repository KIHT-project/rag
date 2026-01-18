from __future__ import annotations

import pytest

from biomed_platform.core.domains.ingestion import IngestItem, IngestItemState
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.services.ingestion.in_memory_document_registry import InMemoryDocumentRegistry
from biomed_platform.core.services.ingestion.in_memory_job_store import InMemoryIngestionJobStore
from biomed_platform.core.services.ingestion.in_memory_payload_store import InMemoryIngestPayloadStore
from biomed_platform.core.services.ingestion.in_memory_queue import InMemoryIngestionQueue
from biomed_platform.core.services.ingestion.worker import IngestionWorker


class _Pipeline:
    def __init__(self, *, exc: Exception | None = None) -> None:
        self.exc = exc
        self.calls: list[tuple[str, str, str]] = []

    async def ingest_item(self, *, job_id: str, embedding_model_id: str, doc_id: str, item: IngestItem) -> None:
        self.calls.append((job_id, embedding_model_id, doc_id))
        if self.exc is not None:
            raise self.exc


def _item(doi: str) -> IngestItem:
    return IngestItem(
        doi_original=doi,
        doi_normalized=doi,
        disease="d",
        source_type="t",
        content_text="x",
    )


@pytest.mark.asyncio
async def test_process_single_item_succeeds_and_commits_doc_id() -> None:
    # Given a worker with a pipeline that succeeds
    reg = InMemoryDocumentRegistry()
    await reg.reserve(embedding_model_id="m", doc_id="d")

    worker = IngestionWorker(
        queue=InMemoryIngestionQueue(max_size=1),
        job_store=InMemoryIngestionJobStore(ttl_seconds_after_completion=10, max_jobs=10),
        document_registry=reg,
        payload_store=InMemoryIngestPayloadStore(ttl_seconds=10, max_jobs=10),
        pipeline=_Pipeline(),
    )

    # When processing one item
    status = await worker._process_single_item(
        job_id="j",
        embedding_model_id="m",
        doc_id="d",
        item=_item("10.1/a"),
    )

    # Then it is marked succeeded
    assert status.state == IngestItemState.succeeded

    # Then doc id is committed, reserve again should fail
    with pytest.raises(KeyError):
        await reg.reserve(embedding_model_id="m", doc_id="d")


@pytest.mark.asyncio
async def test_process_single_item_duplicate_doc_skips_and_commits_doc_id() -> None:
    # Given a worker with a pipeline that raises duplicate_doc
    reg = InMemoryDocumentRegistry()
    await reg.reserve(embedding_model_id="m", doc_id="d")

    dup = SystemError(code="duplicate_doc", message="dup", details=None, retryable=False)

    worker = IngestionWorker(
        queue=InMemoryIngestionQueue(max_size=1),
        job_store=InMemoryIngestionJobStore(ttl_seconds_after_completion=10, max_jobs=10),
        document_registry=reg,
        payload_store=InMemoryIngestPayloadStore(ttl_seconds=10, max_jobs=10),
        pipeline=_Pipeline(exc=dup),
    )

    # When processing one item
    status = await worker._process_single_item(
        job_id="j",
        embedding_model_id="m",
        doc_id="d",
        item=_item("10.1/a"),
    )

    # Then it is marked skipped_duplicate
    assert status.state == IngestItemState.skipped_duplicate

    # Then doc id is committed
    with pytest.raises(KeyError):
        await reg.reserve(embedding_model_id="m", doc_id="d")


@pytest.mark.asyncio
async def test_process_single_item_other_system_error_releases_doc_id() -> None:
    # Given a worker with a pipeline that raises a system error
    reg = InMemoryDocumentRegistry()
    await reg.reserve(embedding_model_id="m", doc_id="d")

    err = SystemError(code="boom", message="x", details=None, retryable=True)

    worker = IngestionWorker(
        queue=InMemoryIngestionQueue(max_size=1),
        job_store=InMemoryIngestionJobStore(ttl_seconds_after_completion=10, max_jobs=10),
        document_registry=reg,
        payload_store=InMemoryIngestPayloadStore(ttl_seconds=10, max_jobs=10),
        pipeline=_Pipeline(exc=err),
    )

    # When processing one item
    status = await worker._process_single_item(
        job_id="j",
        embedding_model_id="m",
        doc_id="d",
        item=_item("10.1/a"),
    )

    # Then it is marked failed
    assert status.state == IngestItemState.failed

    # Then doc id was released and can be reserved again
    await reg.reserve(embedding_model_id="m", doc_id="d")
