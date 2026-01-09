from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.ingestion import (
    IngestItemState,
    IngestItemStatus,
    JobCounts,
    JobState,
    JobStats,
)
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.services.ingestion_ports import (
    DocumentRegistry,
    IngestPayloadStore,
    IngestionJobStore,
    IngestionQueue,
    IngestionPipeline,
)

log = get_logger(__name__)


class IngestionWorker:
    def __init__(
        self,
        *,
        queue: IngestionQueue,
        job_store: IngestionJobStore,
        document_registry: DocumentRegistry,
        payload_store: IngestPayloadStore,
        pipeline: IngestionPipeline,
    ) -> None:
        self._queue = queue
        self._job_store = job_store
        self._document_registry = document_registry
        self._payload_store = payload_store
        self._pipeline = pipeline

        log.info("IngestionWorker initialized, queue_max_size=%s", queue.max_size())

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    async def run_forever(self) -> None:
        log.info("IngestionWorker started run loop")
        try:
            while True:
                job_id = await self._queue.dequeue()
                log.debug("Dequeued job, job_id=%s, queue_size=%s", job_id, self._queue.size())
                await self._process_job(job_id)
        except asyncio.CancelledError:
            log.info("IngestionWorker cancelled, exiting run loop")
            raise
        except Exception:
            log.exception("IngestionWorker run loop crashed")
            raise

    async def _process_job(self, job_id: str) -> None:
        log.info("Processing job started, job_id=%s", job_id)

        job = await self._load_job_or_skip(job_id)
        if job is None:
            return

        original_items = list(job.items)
        doc_id_by_doi = self._build_doc_id_index(job_id=job_id, job_items=original_items)

        items = await self._load_payload_or_fail(job_id)
        if items is None:
            return

        await self._mark_job_running(job_id=job_id)
        log.info("Job marked running, job_id=%s", job_id)

        stats = JobStats()
        processed_statuses = await self._process_items(
            job_id=job_id,
            embedding_model_id=job.effective_embedding_model_id,
            items=items,
            doc_id_by_doi=doc_id_by_doi,
            stats=stats,
        )

        await self._finalize_job(
            job_id=job_id,
            original_items=original_items,
            processed_statuses=processed_statuses,
        )

        await self._safe_payload_delete(job_id)

        log.info(
            "Job processing completed, job_id=%s, final_state=%s",
            job_id,
            JobState.succeeded.value,
        )

    async def _load_job_or_skip(self, job_id: str):
        try:
            job = await self._job_store.get(job_id)
        except KeyError:
            log.warning("Job not found in store, skipping, job_id=%s", job_id)
            await self._safe_payload_delete(job_id)
            return None
        except Exception:
            log.exception("Job store get failed, job_id=%s", job_id)
            raise

        log.debug(
            "Job loaded, job_id=%s, state=%s, item_count=%s, embedding_model_id=%s",
            job_id,
            job.state.value,
            len(job.items),
            job.effective_embedding_model_id,
        )
        return job

    async def _load_payload_or_fail(self, job_id: str):
        try:
            items = await self._payload_store.get(job_id=job_id)
        except KeyError:
            log.warning("Payload not found for job, failing job, job_id=%s", job_id)
            await self._mark_job_failed_missing_payload(job_id=job_id)
            return None
        except Exception:
            log.exception("Payload store get failed, job_id=%s", job_id)
            raise

        log.debug("Payload loaded, job_id=%s, payload_item_count=%s", job_id, len(items))
        return items

    async def _process_items(
        self,
        *,
        job_id: str,
        embedding_model_id: str,
        items,
        doc_id_by_doi: dict[str, str],
        stats: JobStats,
    ) -> list[IngestItemStatus]:
        processed: list[IngestItemStatus] = []

        for item in items:
            doi = item.doi_original
            doc_id = doc_id_by_doi.get(doi, "")

            if not doc_id:
                stats.failed += 1
                log.warning(
                    "Missing doc_id mapping for item, job_id=%s, doi_original=%s",
                    job_id,
                    doi,
                )
                processed.append(
                    IngestItemStatus(
                        doi_original=doi,
                        doc_id="",
                        state=IngestItemState.failed,
                        message="missing_doc_id_mapping",
                    )
                )
                continue

            status = await self._process_single_item(
                job_id=job_id,
                embedding_model_id=embedding_model_id,
                doc_id=doc_id,
                item=item,
            )

            processed.append(status)

            if status.state == IngestItemState.succeeded:
                stats.succeeded += 1
            elif status.state == IngestItemState.failed:
                stats.failed += 1
            else:
                stats.skipped += 1

        return processed

    async def _process_single_item(
        self,
        *,
        job_id: str,
        embedding_model_id: str,
        doc_id: str,
        item,
    ) -> IngestItemStatus:
        doi = item.doi_original

        log.debug(
            "Item processing started, job_id=%s, doi_original=%s, doc_id=%s",
            job_id,
            doi,
            doc_id,
        )

        try:
            await self._pipeline.ingest_item(
                job_id=job_id,
                embedding_model_id=embedding_model_id,
                doc_id=doc_id,
                item=item,
            )
        except SystemError as exc:
            if exc.code == "duplicate_doc":
                await self._safe_commit(job_embedding_model_id=embedding_model_id, doc_id=doc_id)
                log.debug(
                    "Item skipped as duplicate, job_id=%s, doi_original=%s, doc_id=%s",
                    job_id,
                    doi,
                    doc_id,
                )
                return IngestItemStatus(
                    doi_original=doi,
                    doc_id=doc_id,
                    state=IngestItemState.skipped_duplicate,
                    message="already_indexed",
                )

            log.warning(
                "Item processing failed with SystemError, job_id=%s, doi_original=%s,"
                "doc_id=%s, code=%s",
                job_id,
                doi,
                doc_id,
                exc.code,
            )
            await self._safe_release(job_embedding_model_id=embedding_model_id, doc_id=doc_id)
            return IngestItemStatus(
                doi_original=doi,
                doc_id=doc_id,
                state=IngestItemState.failed,
                message=str(exc.code),
            )
        except Exception as exc:
            log.exception(
                "Item processing failed with unexpected exception, job_id=%s, doi_original=%s,"
                "doc_id=%s, exc_type=%s",
                job_id,
                doi,
                doc_id,
                type(exc).__name__,
            )
            await self._safe_release(job_embedding_model_id=embedding_model_id, doc_id=doc_id)
            msg = f"processing_failed, {type(exc).__name__}"
            return IngestItemStatus(
                doi_original=doi,
                doc_id=doc_id,
                state=IngestItemState.failed,
                message=msg,
            )

        await self._safe_commit(job_embedding_model_id=embedding_model_id, doc_id=doc_id)

        log.debug(
            "Item processing succeeded, job_id=%s, doi_original=%s, doc_id=%s",
            job_id,
            doi,
            doc_id,
        )

        return IngestItemStatus(
            doi_original=doi,
            doc_id=doc_id,
            state=IngestItemState.succeeded,
            message=None,
        )

    async def _finalize_job(
        self,
        *,
        job_id: str,
        original_items: list[IngestItemStatus],
        processed_statuses: list[IngestItemStatus],
    ) -> None:
        job = await self._job_store.get(job_id)
        now = self._now()

        processed_by_doi: dict[str, IngestItemStatus] = {}
        for st in processed_statuses:
            processed_by_doi[st.doi_original] = st

        original_dois = {st.doi_original for st in original_items}

        unexpected_dois = [doi for doi in processed_by_doi.keys() if doi not in original_dois]
        if unexpected_dois:
            log.warning(
                "Unexpected payload items, job_id=%s, unexpected_dois=%s",
                job_id,
                sorted(unexpected_dois),
            )

        merged: list[IngestItemStatus] = []
        for st in original_items:
            replacement = processed_by_doi.get(st.doi_original)
            if replacement is None:
                merged.append(st)
            else:
                merged.append(replacement)

        counts = self._compute_counts(merged)

        if unexpected_dois:
            counts = JobCounts(
                total=counts.total,
                succeeded=counts.succeeded,
                failed=counts.failed + len(unexpected_dois),
                skipped_duplicate=counts.skipped_duplicate,
            )

        final_state = self._compute_final_state(counts)

        job.items = merged
        job.counts = counts
        job.state = final_state
        job.updated_at = now

        await self._job_store.update(job)

        log.info(
            "Job updated, job_id=%s, state=%s, total=%s, succeeded=%s, failed=%s, skipped=%s",
            job_id,
            job.state.value,
            job.counts.total,
            job.counts.succeeded,
            job.counts.failed,
            job.counts.skipped_duplicate,
        )

    def _compute_counts(self, items: list[IngestItemStatus]) -> JobCounts:
        succeeded = 0
        failed = 0
        skipped = 0

        for st in items:
            if st.state == IngestItemState.succeeded:
                succeeded += 1
            elif st.state == IngestItemState.failed:
                failed += 1
            elif st.state == IngestItemState.skipped_duplicate:
                skipped += 1

        return JobCounts(
            total=len(items),
            succeeded=succeeded,
            failed=failed,
            skipped_duplicate=skipped,
        )

    def _compute_final_state(self, counts: JobCounts) -> JobState:
        if counts.failed > 0 and counts.succeeded > 0:
            return JobState.partial
        if counts.failed > 0 and counts.succeeded == 0:
            return JobState.failed
        return JobState.succeeded

    async def _mark_job_running(self, *, job_id: str) -> None:
        job = await self._job_store.get(job_id)
        now = self._now()

        updated: list[IngestItemStatus] = []
        for it in job.items:
            if it.state == IngestItemState.queued:
                updated.append(
                    IngestItemStatus(
                        doi_original=it.doi_original,
                        doc_id=it.doc_id,
                        state=IngestItemState.running,
                        message=it.message,
                    )
                )
            else:
                updated.append(it)

        job.state = JobState.running
        job.updated_at = now
        job.items = updated

        await self._job_store.update(job)
        log.debug("Job state set to running, job_id=%s", job_id)

    async def _mark_job_failed_missing_payload(self, *, job_id: str) -> None:
        try:
            job = await self._job_store.get(job_id)
        except KeyError:
            log.warning("Cannot mark job failed, job missing, job_id=%s", job_id)
            return
        except Exception:
            log.exception("Job store get failed while marking missing payload, job_id=%s", job_id)
            raise

        log.warning(
            "Marking job failed due to missing payload, job_id=%s, item_count=%s",
            job_id,
            len(job.items),
        )

        for st in job.items:
            await self._safe_release(
                job_embedding_model_id=job.effective_embedding_model_id,
                doc_id=st.doc_id,
            )

        now = self._now()
        failed_items: list[IngestItemStatus] = []
        for it in job.items:
            failed_items.append(
                IngestItemStatus(
                    doi_original=it.doi_original,
                    doc_id=it.doc_id,
                    state=IngestItemState.failed,
                    message="missing_payload",
                )
            )

        job.state = JobState.failed
        job.updated_at = now
        job.items = failed_items
        job.counts = JobCounts(
            total=len(failed_items),
            succeeded=0,
            failed=len(failed_items),
            skipped_duplicate=0,
        )

        await self._job_store.update(job)
        await self._safe_payload_delete(job_id)

        log.info("Job marked failed due to missing payload, job_id=%s", job_id)

    def _build_doc_id_index(
        self, *, job_id: str, job_items: list[IngestItemStatus]
    ) -> dict[str, str]:
        mapping: dict[str, str] = {}
        duplicates: set[str] = set()

        for st in job_items:
            if st.doi_original in mapping:
                duplicates.add(st.doi_original)
            mapping[st.doi_original] = st.doc_id or ""

        if duplicates:
            log.warning(
                "Duplicate doi_original values in job items, job_id=%s, duplicates=%s",
                job_id,
                sorted(duplicates),
            )

        log.debug(
            "Doc id index built, job_id=%s, unique_dois=%s, duplicates=%s",
            job_id,
            len(mapping),
            len(duplicates),
        )

        return mapping

    async def _safe_commit(self, *, job_embedding_model_id: str, doc_id: str) -> None:
        if not doc_id:
            return
        try:
            await self._document_registry.commit(
                embedding_model_id=job_embedding_model_id,
                doc_id=doc_id,
            )
        except Exception:
            log.exception(
                "Commit failed, embedding_model_id=%s, doc_id=%s",
                job_embedding_model_id,
                doc_id,
            )

    async def _safe_release(self, *, job_embedding_model_id: str, doc_id: str) -> None:
        if not doc_id:
            return
        try:
            await self._document_registry.release(
                embedding_model_id=job_embedding_model_id,
                doc_id=doc_id,
            )
        except Exception:
            log.exception(
                "Release failed, embedding_model_id=%s, doc_id=%s",
                job_embedding_model_id,
                doc_id,
            )

    async def _safe_payload_delete(self, job_id: str) -> None:
        try:
            await self._payload_store.delete(job_id=job_id)
        except Exception:
            log.exception("Payload delete failed, job_id=%s", job_id)
