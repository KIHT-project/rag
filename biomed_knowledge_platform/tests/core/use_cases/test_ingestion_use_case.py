from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from biomed_platform.core.domains.ingestion import (
    IngestBatchCommand,
    IngestItem,
    JobState,
)
from biomed_platform.core.errors.errors import AppError, BusinessError, SystemError
from biomed_platform.core.use_cases import ingestion as ingestion_mod
from biomed_platform.core.use_cases.ingestion import IngestionUseCase


def _item(*, doi: str) -> IngestItem:
    return IngestItem(
        doi_original=doi,
        doi_normalized=doi,
        disease="d",
        source_type="t",
        content_text="x",
    )


@dataclass(slots=True)
class _IdRec:
    job_id: str


@pytest.mark.anyio
async def test_ingest_batch_raises_when_no_valid_items() -> None:
    # Given
    uc = IngestionUseCase(
        queue=AsyncMock(),
        job_store=AsyncMock(),
        idempotency_store=AsyncMock(),
        document_registry=AsyncMock(),
        backpressure=AsyncMock(),
        worker_count=1,
        payload_store=AsyncMock(),
    )

    cmd = IngestBatchCommand(
        effective_embedding_model_id="m",
        items=(
            _item(doi=""),
            _item(doi="   "),
        ),
        idempotency_key=None,
        body_hash="h",
        correlation_id=None,
    )

    # When, Then
    with pytest.raises(BusinessError) as exc:
        await uc.ingest_batch(cmd)

    assert exc.value.code == "validation_error"


@pytest.mark.anyio
async def test_ingest_batch_returns_existing_job_on_idempotency_hit(monkeypatch) -> None:
    # Given
    queue = AsyncMock()
    job_store = AsyncMock()
    id_store = AsyncMock()
    id_store.get_job_id = AsyncMock(return_value="existing")
    registry = AsyncMock()
    backpressure = AsyncMock()
    payload_store = AsyncMock()

    uc = IngestionUseCase(
        queue=queue,
        job_store=job_store,
        idempotency_store=id_store,
        document_registry=registry,
        backpressure=backpressure,
        worker_count=1,
        payload_store=payload_store,
    )

    cmd = IngestBatchCommand(
        effective_embedding_model_id="m",
        items=(_item(doi="10.1/a"),),
        idempotency_key="k",
        body_hash="h",
        correlation_id=None,
    )

    # When
    res = await uc.ingest_batch(cmd)

    # Then
    assert res.job_id == "existing"
    assert res.state == JobState.queued
    job_store.create.assert_not_called()
    queue.enqueue.assert_not_called()


@pytest.mark.anyio
async def test_ingest_batch_reserves_docs_creates_job_puts_payload_enqueues_writes_idempotency(monkeypatch) -> None:
    # Given
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    queue = AsyncMock()
    job_store = AsyncMock()
    id_store = AsyncMock()
    id_store.get_job_id = AsyncMock(return_value=None)
    id_store.peek_record = AsyncMock(return_value=None)

    registry = AsyncMock()
    registry.reserve = AsyncMock(return_value=None)
    registry.release = AsyncMock()

    payload_store = AsyncMock()
    payload_store.put = AsyncMock()
    payload_store.delete = AsyncMock()

    backpressure = AsyncMock()

    uc = IngestionUseCase(
        queue=queue,
        job_store=job_store,
        idempotency_store=id_store,
        document_registry=registry,
        backpressure=backpressure,
        worker_count=2,
        payload_store=payload_store,
    )

    monkeypatch.setattr(uc, "_now", lambda: now, raising=True)

    monkeypatch.setattr(
        ingestion_mod,
        "compute_doc_id",
        lambda *, doi_normalized: f"doc_{doi_normalized}",
        raising=True,
    )

    monkeypatch.setattr(
        ingestion_mod.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="job123"),
        raising=True,
    )

    cmd = IngestBatchCommand(
        effective_embedding_model_id="m",
        items=(_item(doi="10.1/a"), _item(doi="10.1/b")),
        idempotency_key="k",
        body_hash="h",
        correlation_id="c",
    )

    # When
    res = await uc.ingest_batch(cmd)

    # Then
    assert res.job_id == "job123"
    assert res.state == JobState.queued

    assert job_store.create.await_count == 1
    assert payload_store.put.await_count == 1
    assert queue.enqueue.await_count == 1
    assert id_store.put.await_count == 1


@pytest.mark.anyio
async def test_ingest_batch_queue_full_triggers_rollback(monkeypatch) -> None:
    # Given
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    queue = AsyncMock()
    queue.enqueue = AsyncMock(side_effect=asyncio.QueueFull())
    # size and max_size are sync in production code, so make them sync here
    queue.size = MagicMock(return_value=10)
    queue.max_size = MagicMock(return_value=10)

    job_store = AsyncMock()
    job_store.create = AsyncMock()
    job_store.delete = AsyncMock()

    id_store = AsyncMock()
    id_store.get_job_id = AsyncMock(return_value=None)
    id_store.peek_record = AsyncMock(return_value=None)

    registry = AsyncMock()
    registry.reserve = AsyncMock(return_value=None)
    registry.release = AsyncMock()

    payload_store = AsyncMock()
    payload_store.put = AsyncMock()
    payload_store.delete = AsyncMock()

    backpressure = AsyncMock()
    backpressure.retry_after = MagicMock(return_value=SimpleNamespace(seconds=7))

    uc = IngestionUseCase(
        queue=queue,
        job_store=job_store,
        idempotency_store=id_store,
        document_registry=registry,
        backpressure=backpressure,
        worker_count=1,
        payload_store=payload_store,
    )

    monkeypatch.setattr(uc, "_now", lambda: now, raising=True)
    monkeypatch.setattr(
        ingestion_mod,
        "compute_doc_id",
        lambda *, doi_normalized: f"doc_{doi_normalized}",
        raising=True,
    )
    monkeypatch.setattr(
        ingestion_mod.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="jobqfull"),
        raising=True,
    )

    cmd = IngestBatchCommand(
        effective_embedding_model_id="m",
        items=(_item(doi="10.1/a"),),
        idempotency_key=None,
        body_hash="h",
        correlation_id=None,
    )

    # When, Then
    with pytest.raises(AppError) as exc:
        await uc.ingest_batch(cmd)

    assert exc.value.code == "too_many_requests"

    # rollback happened
    assert registry.release.await_count == 1
    assert payload_store.delete.await_count == 1
    assert job_store.delete.await_count == 1


@pytest.mark.anyio
async def test_ingest_batch_idempotency_conflict_triggers_rollback(monkeypatch) -> None:
    # Given
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    queue = AsyncMock()
    queue.enqueue = AsyncMock()
    queue.size = MagicMock(return_value=0)
    queue.max_size = MagicMock(return_value=10)

    job_store = AsyncMock()
    job_store.create = AsyncMock()
    job_store.delete = AsyncMock()

    id_store = AsyncMock()
    id_store.get_job_id = AsyncMock(return_value=None)
    id_store.peek_record = AsyncMock(return_value=_IdRec(job_id="other"))

    registry = AsyncMock()
    registry.reserve = AsyncMock(return_value=None)
    registry.release = AsyncMock()

    payload_store = AsyncMock()
    payload_store.put = AsyncMock()
    payload_store.delete = AsyncMock()

    backpressure = AsyncMock()

    uc = IngestionUseCase(
        queue=queue,
        job_store=job_store,
        idempotency_store=id_store,
        document_registry=registry,
        backpressure=backpressure,
        worker_count=1,
        payload_store=payload_store,
    )

    monkeypatch.setattr(uc, "_now", lambda: now, raising=True)
    monkeypatch.setattr(
        ingestion_mod,
        "compute_doc_id",
        lambda *, doi_normalized: f"doc_{doi_normalized}",
        raising=True,
    )
    monkeypatch.setattr(
        ingestion_mod.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="newjob"),
        raising=True,
    )

    cmd = IngestBatchCommand(
        effective_embedding_model_id="m",
        items=(_item(doi="10.1/a"),),
        idempotency_key="k",
        body_hash="h",
        correlation_id=None,
    )

    # When, Then
    with pytest.raises(AppError) as exc:
        await uc.ingest_batch(cmd)

    # this error is generated by idempotency_conflict_error
    assert exc.value.code == "validation_error"
    assert exc.value.details is not None
    assert exc.value.details.get("idempotency_key") == "k"

    # rollback happened
    assert registry.release.await_count == 1
    assert payload_store.delete.await_count == 1
    assert job_store.delete.await_count == 1


@pytest.mark.anyio
async def test_ingest_batch_no_reserved_docs_returns_without_enqueue_and_writes_idempotency(monkeypatch) -> None:
    # Given, registry always says duplicate via KeyError
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    queue = AsyncMock()
    job_store = AsyncMock()
    id_store = AsyncMock()
    id_store.get_job_id = AsyncMock(return_value=None)
    id_store.peek_record = AsyncMock(return_value=None)
    id_store.put = AsyncMock()

    registry = AsyncMock()
    registry.reserve = AsyncMock(side_effect=KeyError("dup"))
    registry.release = AsyncMock()

    payload_store = AsyncMock()
    payload_store.put = AsyncMock()
    payload_store.delete = AsyncMock()

    backpressure = AsyncMock()

    uc = IngestionUseCase(
        queue=queue,
        job_store=job_store,
        idempotency_store=id_store,
        document_registry=registry,
        backpressure=backpressure,
        worker_count=1,
        payload_store=payload_store,
    )

    monkeypatch.setattr(uc, "_now", lambda: now, raising=True)
    monkeypatch.setattr(
        ingestion_mod,
        "compute_doc_id",
        lambda *, doi_normalized: "doc_same",
        raising=True,
    )
    monkeypatch.setattr(
        ingestion_mod.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="job_nores"),
        raising=True,
    )

    cmd = IngestBatchCommand(
        effective_embedding_model_id="m",
        items=(_item(doi="10.1/a"),),
        idempotency_key="k",
        body_hash="h",
        correlation_id=None,
    )

    # When
    res = await uc.ingest_batch(cmd)

    # Then
    assert res.job_id == "job_nores"
    assert res.state == JobState.succeeded
    queue.enqueue.assert_not_called()
    payload_store.put.assert_not_called()
    assert id_store.put.await_count == 1


@pytest.mark.anyio
async def test_build_job_mismatch_raises_system_error() -> None:
    # Given
    uc = IngestionUseCase(
        queue=AsyncMock(),
        job_store=AsyncMock(),
        idempotency_store=AsyncMock(),
        document_registry=AsyncMock(),
        backpressure=AsyncMock(),
        worker_count=1,
        payload_store=AsyncMock(),
    )

    cmd = IngestBatchCommand(
        effective_embedding_model_id="m",
        items=(_item(doi="10.1/a"), _item(doi="10.1/b")),
        idempotency_key=None,
        body_hash="h",
        correlation_id=None,
    )

    # Given, reserved missing one requested item so statuses will not match request size
    reserved = SimpleNamespace(items_with_doc_id=[(cmd.items[0], "doc1")], reserved_doc_ids=["doc1"])
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # When, Then
    with pytest.raises(SystemError) as exc:
        uc._build_job(
            cmd=cmd,
            job_id="j",
            now=now,
            reserved=reserved,
            invalid_items=[],
            skipped_duplicates=[],
        )

    assert exc.value.code == "job_build_mismatch"


@pytest.mark.anyio
async def test_get_job_status_success_and_not_found_and_app_error_passthrough() -> None:
    # Given
    job_store = AsyncMock()
    uc = IngestionUseCase(
        queue=AsyncMock(),
        job_store=job_store,
        idempotency_store=AsyncMock(),
        document_registry=AsyncMock(),
        backpressure=AsyncMock(),
        worker_count=1,
        payload_store=AsyncMock(),
    )

    job_store.get = AsyncMock(return_value=SimpleNamespace(job_id="j", state=SimpleNamespace(value="queued")))

    # When
    job = await uc.get_job_status(job_id="j")

    # Then
    assert job.job_id == "j"

    # Given, not found
    job_store.get = AsyncMock(side_effect=KeyError("missing"))

    # When, Then
    with pytest.raises(AppError) as exc:
        await uc.get_job_status(job_id="missing")

    assert exc.value.code == "not_found"

    # Given, AppError passthrough
    ae = AppError(code="system_error", message="x", details=None, retryable=False)
    job_store.get = AsyncMock(side_effect=ae)

    # When, Then
    with pytest.raises(AppError) as exc2:
        await uc.get_job_status(job_id="j")

    assert exc2.value is ae
