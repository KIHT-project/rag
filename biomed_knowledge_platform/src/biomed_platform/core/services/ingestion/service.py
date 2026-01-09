from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Iterable

from biomed_platform.common.logging import get_logger
from biomed_platform.common.utils import compute_doc_id
from biomed_platform.core.domains.ingestion import (
    IngestBatchAccepted,
    IngestBatchCommand,
    IngestItem,
    IngestItemState,
    IngestItemStatus,
    IngestionJob,
    JobCounts,
    JobState,
    SplitItems,
    SkippedDuplicate,
    ReserveResult,
    ReservedDocs,
)
from biomed_platform.core.errors.errors import (
    AppError,
    SystemError,
    idempotency_conflict_error,
    job_not_found_error,
    no_valid_items_error,
    queue_full_error,
)
from biomed_platform.core.services.ingestion_ports import (
    BackpressurePolicy,
    DocumentRegistry,
    IdempotencyStore,
    IngestionJobStore,
    IngestionQueue,
    IngestionService,
    IngestPayloadStore,
)

log = get_logger(__name__)


def _is_valid_item(item: IngestItem) -> bool:
    return bool((item.doi_normalized or "").strip())


def _skipped_state() -> IngestItemState:
    cand = getattr(IngestItemState, "skipped_duplicate", None)
    if cand is not None:
        return cand
    return IngestItemState.failed


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
        payload_store: IngestPayloadStore,
    ) -> None:
        self._queue = queue
        self._job_store = job_store
        self._idempotency_store = idempotency_store
        self._document_registry = document_registry
        self._backpressure = backpressure
        self._worker_count = worker_count
        self._payload_store = payload_store

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
            payload_store: IngestPayloadStore,
        ) -> None:
            self._job_store = job_store
            self._document_registry = document_registry
            self._embedding_model_id = embedding_model_id
            self._job_id = job_id
            self._reserved_doc_ids = list(reserved_doc_ids)
            self._armed = True
            self._payload_store = payload_store

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

            await self._payload_store.delete(job_id=self._job_id)
            await self._job_store.delete(self._job_id)

            log.debug("Ingest rollback completed, job_id=%s", self._job_id)

        def disarm(self) -> None:
            self._armed = False

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _split_items(self, cmd: IngestBatchCommand) -> SplitItems:
        valid: list[IngestItem] = []
        invalid: list[IngestItem] = []

        for it in cmd.items:
            if _is_valid_item(it):
                valid.append(it)
            else:
                invalid.append(it)

        log.debug(
            "Split ingest items, total=%d, valid=%d, invalid=%d",
            len(cmd.items),
            len(valid),
            len(invalid),
        )
        return SplitItems(valid=valid, invalid=invalid)

    async def ingest_batch(self, cmd: IngestBatchCommand) -> IngestBatchAccepted:
        now = self._now()

        split = self._split_items(cmd)
        if not split.valid:
            raise no_valid_items_error(embedding_model_id=cmd.effective_embedding_model_id)

        existing_job_id = await self._get_idempotent_job_id_if_any(cmd)
        if existing_job_id is not None:
            return IngestBatchAccepted(job_id=existing_job_id, state=JobState.queued)

        reserve_result = await self._reserve_unique_docs(
            items=split.valid,
            embedding_model_id=cmd.effective_embedding_model_id,
        )
        reserved = reserve_result.reserved
        skipped_duplicates = reserve_result.skipped_duplicates

        job_id = uuid.uuid4().hex
        job = self._build_job(
            cmd=cmd,
            job_id=job_id,
            now=now,
            reserved=reserved,
            invalid_items=split.invalid,
            skipped_duplicates=skipped_duplicates,
        )
        await self._job_store.create(job)

        reserved_items = [it for (it, _doc_id) in reserved.items_with_doc_id]
        if reserved_items:
            await self._payload_store.put(job_id=job_id, items=reserved_items)

        if not reserved.reserved_doc_ids:
            await self._enforce_idempotency_conflict_check(cmd=cmd, job_id=job_id)
            await self._write_idempotency_after_enqueue(cmd=cmd, job_id=job_id, now=now)
            return IngestBatchAccepted(job_id=job_id, state=job.state)

        rollback = self._IngestRollback(
            job_store=self._job_store,
            document_registry=self._document_registry,
            embedding_model_id=cmd.effective_embedding_model_id,
            job_id=job_id,
            reserved_doc_ids=reserved.reserved_doc_ids,
            payload_store=self._payload_store,
        )

        try:
            await self._enforce_idempotency_conflict_check(cmd=cmd, job_id=job_id)
            await self._enqueue_or_raise(job_id=job_id)
            await self._write_idempotency_after_enqueue(cmd=cmd, job_id=job_id, now=now)
        except Exception:
            await rollback.run()
            raise

        rollback.disarm()
        return IngestBatchAccepted(job_id=job_id, state=JobState.queued)

    async def _get_idempotent_job_id_if_any(self, cmd: IngestBatchCommand) -> str | None:
        if not cmd.idempotency_key:
            return None

        job_id = await self._idempotency_store.get_job_id(
            key=cmd.idempotency_key,
            body_hash=cmd.body_hash,
        )

        log.debug("Idempotency lookup completed, hit=%s", job_id is not None)
        return job_id

    async def _reserve_unique_docs(
        self,
        *,
        items: list[IngestItem],
        embedding_model_id: str,
    ) -> ReserveResult:
        items_with_doc_id: list[tuple[IngestItem, str]] = []
        reserved_doc_ids: list[str] = []
        skipped_duplicates: list[SkippedDuplicate] = []

        seen: set[str] = set()

        for item in items:
            doc_id = compute_doc_id(doi_normalized=item.doi_normalized)

            if doc_id in seen:
                skipped_duplicates.append(
                    SkippedDuplicate(item=item, doc_id=doc_id, message="duplicate_doi")
                )
                continue
            seen.add(doc_id)

            try:
                await self._document_registry.reserve(
                    embedding_model_id=embedding_model_id,
                    doc_id=doc_id,
                )
            except KeyError:
                skipped_duplicates.append(
                    SkippedDuplicate(item=item, doc_id=doc_id, message="duplicate_doi")
                )
                continue

            items_with_doc_id.append((item, doc_id))
            reserved_doc_ids.append(doc_id)

        log.debug(
            "Reserve unique docs completed, reserved=%d,"
            "skipped_duplicates=%d, embedding_model_id=%s",
            len(reserved_doc_ids),
            len(skipped_duplicates),
            embedding_model_id,
        )

        return ReserveResult(
            reserved=ReservedDocs(
                items_with_doc_id=items_with_doc_id,
                reserved_doc_ids=reserved_doc_ids,
            ),
            skipped_duplicates=skipped_duplicates,
        )

    def _build_job(
        self,
        *,
        cmd: IngestBatchCommand,
        job_id: str,
        now: datetime,
        reserved: ReservedDocs,
        invalid_items: list[IngestItem],
        skipped_duplicates: list[SkippedDuplicate],
    ) -> IngestionJob:
        statuses: list[IngestItemStatus] = []

        for it in invalid_items:
            statuses.append(
                IngestItemStatus(
                    doi_original=it.doi_original,
                    doc_id="",
                    state=IngestItemState.failed,
                    message="invalid_doi",
                )
            )

        for sd in skipped_duplicates:
            statuses.append(
                IngestItemStatus(
                    doi_original=sd.item.doi_original,
                    doc_id=sd.doc_id,
                    state=_skipped_state(),
                    message=sd.message,
                )
            )

        for item, doc_id in reserved.items_with_doc_id:
            statuses.append(
                IngestItemStatus(
                    doi_original=item.doi_original,
                    doc_id=doc_id,
                    state=IngestItemState.queued,
                    message=None,
                )
            )

        total_requested = len(cmd.items)
        total_statuses = len(statuses)

        if total_statuses != total_requested:
            log.error(
                "Job build mismatch, job_id=%s, requested=%d, statuses=%d",
                job_id,
                total_requested,
                total_statuses,
            )
            raise SystemError(
                code="job_build_mismatch",
                message="Job items do not match request size",
                details={"requested": total_requested, "statuses": total_statuses},
                retryable=False,
            )

        failed_count = len(invalid_items)
        skipped_count = len(skipped_duplicates)
        reserved_count = len(reserved.items_with_doc_id)

        if reserved_count > 0:
            initial_state = JobState.queued
        else:
            initial_state = JobState.failed if failed_count > 0 else JobState.succeeded

        log.debug(
            "Building ingestion job, job_id=%s, total=%d, reserved=%d,"
            "invalid=%d, skipped=%d, embedding_model_id=%s",
            job_id,
            total_requested,
            reserved_count,
            failed_count,
            skipped_count,
            cmd.effective_embedding_model_id,
        )

        return IngestionJob(
            job_id=job_id,
            state=initial_state,
            created_at=now,
            updated_at=now,
            effective_embedding_model_id=cmd.effective_embedding_model_id,
            items=statuses,
            counts=JobCounts(
                total=total_requested,
                succeeded=0,
                failed=failed_count,
                skipped_duplicate=skipped_count,
            ),
            correlation_id=cmd.correlation_id,
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

        log.warning("Idempotency conflict, existing_job_id=%s, new_job_id=%s", existing_any, job_id)
        raise idempotency_conflict_error(idempotency_key=cmd.idempotency_key)

    async def _peek_any_idempotency_job_id(self, key: str) -> str | None:
        peek = getattr(self._idempotency_store, "peek_record", None)
        if peek is None:
            return None

        rec = await peek(key=key)
        return rec.job_id if rec is not None else None

    async def _enqueue_or_raise(self, *, job_id: str) -> None:
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

        log.debug("Idempotency record written, job_id=%s", job_id)

    async def get_job_status(self, *, job_id: str) -> IngestionJob:
        try:
            job = await self._job_store.get(job_id)
            log.debug("Job status retrieved, job_id=%s, state=%s", job_id, job.state.value)
            return job
        except KeyError:
            log.debug("Job status not found, job_id=%s", job_id)
            raise job_not_found_error(job_id=job_id)
        except AppError:
            raise
