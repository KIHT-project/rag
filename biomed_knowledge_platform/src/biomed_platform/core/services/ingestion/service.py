from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from biomed_platform.common.logging import get_logger
from biomed_platform.common.utils import compute_doc_id
from biomed_platform.core.domains.ingestion import (
    IngestBatchAccepted,
    IngestBatchCommand,
    IngestItemState,
    IngestItemStatus,
    IngestionJob,
    JobCounts,
    JobState,
)
from biomed_platform.core.errors.errors import (
    AppError,
    duplicate_doi_error,
    idempotency_conflict_error,
    job_not_found_error,
    queue_full_error,
)
from biomed_platform.core.services.ingestion_ports import (
    BackpressurePolicy,
    DocumentRegistry,
    IdempotencyStore,
    IngestionJobStore,
    IngestionQueue,
    IngestionService,
)

log = get_logger(__name__)


@dataclass(frozen=True)
class _ReservedDocs:
    items_with_doc_id: list[tuple[object, str]]
    reserved_doc_ids: list[str]


class DefaultIngestionService(IngestionService):
    def __init__(
        self,
        *,
        queue: IngestionQueue,
        job_store: IngestionJobStore,
        idempotency_store: IdempotencyStore,
        document_registry: DocumentRegistry,
        backpressure: BackpressurePolicy,
        worker_count: int,
    ) -> None:
        self._queue = queue
        self._job_store = job_store
        self._idempotency_store = idempotency_store
        self._document_registry = document_registry
        self._backpressure = backpressure
        self._worker_count = worker_count

        log.debug(
            "DefaultIngestionService initialized, worker_count=%d, queue_max_size=%d",
            worker_count,
            queue.max_size(),
        )

    class _IngestRollback:
        def __init__(
            self,
            *,
            job_store: IngestionJobStore,
            document_registry: DocumentRegistry,
            embedding_model_id: str,
            job_id: str,
            reserved_doc_ids: Iterable[str],
        ) -> None:
            self._job_store = job_store
            self._document_registry = document_registry
            self._embedding_model_id = embedding_model_id
            self._job_id = job_id
            self._reserved_doc_ids = list(reserved_doc_ids)
            self._armed = True

        async def run(self) -> None:
            if not self._armed:
                return

            log.warning(
                "Ingest rollback running, job_id=%s, embedding_model_id=%s, reserved_docs=%d",
                self._job_id,
                self._embedding_model_id,
                len(self._reserved_doc_ids),
            )

            for doc_id in self._reserved_doc_ids:
                await self._document_registry.release(
                    embedding_model_id=self._embedding_model_id,
                    doc_id=doc_id,
                )
            await self._job_store.delete(self._job_id)

            log.debug(
                "Ingest rollback completed, job_id=%s",
                self._job_id,
            )

        def disarm(self) -> None:
            self._armed = False

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    async def ingest_batch(self, cmd: IngestBatchCommand) -> IngestBatchAccepted:
        now = self._now()

        log.info(
            "Ingest batch received, items_count=%d, embedding_model_id=%s,"
            "idempotency_key_present=%s",
            len(cmd.items),
            cmd.effective_embedding_model_id,
            cmd.idempotency_key is not None,
        )

        existing_job_id = await self._get_idempotent_job_id_if_any(cmd)
        if existing_job_id is not None:
            log.info(
                "Idempotent ingest hit, returning existing job, job_id=%s",
                existing_job_id,
            )
            return IngestBatchAccepted(job_id=existing_job_id, state=JobState.queued)

        reserved = await self._reserve_all_docs(cmd)

        job_id = uuid.uuid4().hex
        job = self._build_job(
            cmd=cmd, job_id=job_id, now=now, items_with_doc_id=reserved.items_with_doc_id
        )
        await self._job_store.create(job)

        log.debug(
            "Ingestion job created, job_id=%s, items_count=%d, embedding_model_id=%s",
            job_id,
            len(reserved.items_with_doc_id),
            cmd.effective_embedding_model_id,
        )

        rollback = self._IngestRollback(
            job_store=self._job_store,
            document_registry=self._document_registry,
            embedding_model_id=cmd.effective_embedding_model_id,
            job_id=job_id,
            reserved_doc_ids=reserved.reserved_doc_ids,
        )

        try:
            await self._enforce_idempotency_conflict_check(cmd=cmd, job_id=job_id)
            await self._enqueue_or_raise(cmd=cmd, job_id=job_id)
            await self._write_idempotency_after_enqueue(cmd=cmd, job_id=job_id, now=now)
        except Exception:
            log.warning(
                "Ingest batch failed, rolling back, job_id=%s",
                job_id,
            )
            await rollback.run()
            raise

        rollback.disarm()

        log.info(
            "Ingest batch accepted, job_id=%s",
            job_id,
        )
        return IngestBatchAccepted(job_id=job_id, state=JobState.queued)

    async def _get_idempotent_job_id_if_any(self, cmd: IngestBatchCommand) -> str | None:
        if not cmd.idempotency_key:
            return None

        job_id = await self._idempotency_store.get_job_id(
            key=cmd.idempotency_key,
            body_hash=cmd.body_hash,
        )

        log.debug(
            "Idempotency lookup completed, hit=%s",
            job_id is not None,
        )
        return job_id

    async def _reserve_all_docs(self, cmd: IngestBatchCommand) -> _ReservedDocs:
        items_with_doc_id: list[tuple[object, str]] = []
        reserved_doc_ids: list[str] = []

        log.debug(
            "Reserving documents, items_count=%d, embedding_model_id=%s",
            len(cmd.items),
            cmd.effective_embedding_model_id,
        )

        try:
            for item in cmd.items:
                doc_id = compute_doc_id(doi_normalized=item.doi_normalized)
                items_with_doc_id.append((item, doc_id))
                await self._document_registry.reserve(
                    embedding_model_id=cmd.effective_embedding_model_id,
                    doc_id=doc_id,
                )
                reserved_doc_ids.append(doc_id)
        except KeyError:
            log.warning(
                "Duplicate DOI detected during reservation, doi=%s, embedding_model_id=%s",
                item.doi_normalized,
                cmd.effective_embedding_model_id,
            )
            for doc_id in reserved_doc_ids:
                await self._document_registry.release(
                    embedding_model_id=cmd.effective_embedding_model_id,
                    doc_id=doc_id,
                )
            raise duplicate_doi_error(
                doi_normalized=item.doi_normalized,
                embedding_model_id=cmd.effective_embedding_model_id,
            )

        log.debug(
            "Documents reserved, count=%d, embedding_model_id=%s",
            len(reserved_doc_ids),
            cmd.effective_embedding_model_id,
        )

        return _ReservedDocs(items_with_doc_id=items_with_doc_id, reserved_doc_ids=reserved_doc_ids)

    def _build_job(
        self,
        *,
        cmd: IngestBatchCommand,
        job_id: str,
        now: datetime,
        items_with_doc_id: list[tuple[object, str]],
    ) -> IngestionJob:
        statuses = [
            IngestItemStatus(
                doi_original=item.doi_original,
                doc_id=doc_id,
                state=IngestItemState.queued,
            )
            for (item, doc_id) in items_with_doc_id
        ]

        log.debug(
            "Building ingestion job, job_id=%s, items_count=%d, embedding_model_id=%s",
            job_id,
            len(statuses),
            cmd.effective_embedding_model_id,
        )

        return IngestionJob(
            job_id=job_id,
            state=JobState.queued,
            created_at=now,
            updated_at=now,
            effective_embedding_model_id=cmd.effective_embedding_model_id,
            items=statuses,
            counts=JobCounts(
                total=len(statuses),
                succeeded=0,
                failed=0,
                skipped_duplicate=0,
            ),
        )

    async def _enforce_idempotency_conflict_check(
        self, *, cmd: IngestBatchCommand, job_id: str
    ) -> None:
        if not cmd.idempotency_key:
            return

        existing_any = await self._peek_any_idempotency_job_id(cmd.idempotency_key)
        if existing_any is None:
            return

        if existing_any == job_id:
            return

        log.warning(
            "Idempotency conflict, existing_job_id=%s, new_job_id=%s",
            existing_any,
            job_id,
        )
        raise idempotency_conflict_error(idempotency_key=cmd.idempotency_key)

    async def _peek_any_idempotency_job_id(self, key: str) -> str | None:
        peek = getattr(self._idempotency_store, "peek_record", None)
        if peek is None:
            return None

        rec = await peek(key=key)
        return rec.job_id if rec is not None else None

    async def _enqueue_or_raise(self, *, cmd: IngestBatchCommand, job_id: str) -> None:
        try:
            await self._queue.enqueue(job_id)
            log.debug(
                "Job enqueued, job_id=%s, queue_depth=%d, queue_max_size=%d",
                job_id,
                self._queue.size(),
                self._queue.max_size(),
            )
        except asyncio.QueueFull as exc:
            retry_hint = self._backpressure.retry_after(
                queue_depth=self._queue.size(),
                queue_max_size=self._queue.max_size(),
                worker_count=self._worker_count,
            )
            log.warning(
                "Queue full, job rejected, job_id=%s, queue_max_size=%d, retry_after_seconds=%d",
                job_id,
                self._queue.max_size(),
                retry_hint.seconds,
            )
            raise queue_full_error(
                queue_max_size=self._queue.max_size(),
                retry_after_seconds=retry_hint.seconds,
            ) from exc

    async def _write_idempotency_after_enqueue(
        self, *, cmd: IngestBatchCommand, job_id: str, now: datetime
    ) -> None:
        if not cmd.idempotency_key:
            return

        await self._idempotency_store.put(
            key=cmd.idempotency_key,
            body_hash=cmd.body_hash,
            job_id=job_id,
            created_at=now,
        )

        log.debug(
            "Idempotency record written, job_id=%s",
            job_id,
        )

    async def get_job_status(self, *, job_id: str) -> IngestionJob:
        try:
            job = await self._job_store.get(job_id)
            log.debug(
                "Job status retrieved, job_id=%s, state=%s",
                job_id,
                job.state.value,
            )
            return job
        except KeyError:
            log.debug(
                "Job status not found, job_id=%s",
                job_id,
            )
            raise job_not_found_error(job_id=job_id)
        except AppError:
            raise
