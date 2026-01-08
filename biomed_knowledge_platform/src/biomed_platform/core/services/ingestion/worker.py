from __future__ import annotations

from datetime import datetime, timezone

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.ingestion import (
    IngestItemState,
    IngestItemStatus,
    JobCounts,
    JobState,
)
from biomed_platform.core.services.ingestion_ports import (
    IngestionJobStore,
    IngestionQueue,
    DocumentRegistry,
)

log = get_logger(__name__)


class IngestionWorker:
    def __init__(
        self,
        *,
        queue: IngestionQueue,
        job_store: IngestionJobStore,
        document_registry: DocumentRegistry,
    ) -> None:
        self._queue = queue
        self._job_store = job_store
        self._document_registry = document_registry

        log.debug(
            "IngestionWorker initialized, queue_max_size=%d",
            queue.max_size(),
        )

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    async def run_forever(self) -> None:
        log.info("IngestionWorker started run loop")

        while True:
            job_id = await self._queue.dequeue()
            log.debug(
                "Dequeued job for processing, job_id=%s",
                job_id,
            )
            await self._process_job(job_id)

    async def _process_job(self, job_id: str) -> None:
        log.debug(
            "Processing job started, job_id=%s",
            job_id,
        )

        try:
            job = await self._job_store.get(job_id)
        except KeyError:
            log.warning(
                "Job not found in store, skipping processing, job_id=%s",
                job_id,
            )
            return

        now = self._now()
        job.state = JobState.running
        job.updated_at = now
        job.items = [
            IngestItemStatus(
                doi_original=it.doi_original,
                doc_id=it.doc_id,
                state=IngestItemState.running,
                message=it.message,
            )
            for it in job.items
        ]

        await self._job_store.update(job)

        log.debug(
            "Job marked as running, job_id=%s, items_count=%d",
            job_id,
            len(job.items),
        )

        succeeded = 0
        failed = 0
        skipped = 0

        updated_items: list[IngestItemStatus] = []

        for it in job.items:
            succeeded += 1
            updated_items.append(
                IngestItemStatus(
                    doi_original=it.doi_original,
                    doc_id=it.doc_id,
                    state=IngestItemState.succeeded,
                    message=it.message,
                )
            )

        log.debug(
            "Item processing completed, job_id=%s, succeeded=%d, failed=%d, skipped=%d",
            job_id,
            succeeded,
            failed,
            skipped,
        )

        for it in updated_items:
            if it.state == IngestItemState.succeeded:
                await self._document_registry.commit(
                    embedding_model_id=job.effective_embedding_model_id,
                    doc_id=it.doc_id,
                )
                log.debug(
                    "Document committed, job_id=%s, doc_id=%s",
                    job_id,
                    it.doc_id,
                )
            elif it.state in (IngestItemState.failed, IngestItemState.skipped_duplicate):
                await self._document_registry.release(
                    embedding_model_id=job.effective_embedding_model_id,
                    doc_id=it.doc_id,
                )
                log.debug(
                    "Document released, job_id=%s, doc_id=%s, state=%s",
                    job_id,
                    it.doc_id,
                    it.state.value,
                )

        now = self._now()
        job.items = updated_items
        job.counts = JobCounts(
            total=len(updated_items),
            succeeded=succeeded,
            failed=failed,
            skipped_duplicate=skipped,
        )
        job.state = JobState.succeeded if failed == 0 else JobState.partial
        job.updated_at = now

        await self._job_store.update(job)

        log.info(
            "Job processing completed, job_id=%s, final_state=%s, total=%d,"
            "succeeded=%d, failed=%d, skipped=%d",
            job_id,
            job.state.value,
            job.counts.total,
            job.counts.succeeded,
            job.counts.failed,
            job.counts.skipped_duplicate,
        )
