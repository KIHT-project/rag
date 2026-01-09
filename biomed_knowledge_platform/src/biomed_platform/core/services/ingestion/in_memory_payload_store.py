from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.ingestion import IngestItem
from biomed_platform.core.services.ingestion_ports import IngestPayloadStore

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _PayloadRecord:
    items: list[IngestItem]
    created_at: datetime


class InMemoryIngestPayloadStore(IngestPayloadStore):
    def __init__(self, *, ttl_seconds: int = 86400, max_jobs: int = 10_000) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_jobs <= 0:
            raise ValueError("max_jobs must be positive")

        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_jobs = max_jobs
        self._lock = asyncio.Lock()
        self._by_job: dict[str, _PayloadRecord] = {}

        log.info(
            "InMemoryIngestPayloadStore initialized, ttl_seconds=%s, max_jobs=%s",
            ttl_seconds,
            max_jobs,
        )

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _cleanup_unlocked(self, now: datetime) -> None:
        initial_size = len(self._by_job)
        expired: list[str] = []

        for job_id, rec in self._by_job.items():
            created_at = rec.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)

            if (now - created_at) > self._ttl:
                expired.append(job_id)

        for job_id in expired:
            self._by_job.pop(job_id, None)

        if expired:
            log.debug(
                "Expired payloads cleaned, expired_count=%s",
                len(expired),
            )

        if len(self._by_job) <= self._max_jobs:
            if initial_size != len(self._by_job):
                log.debug(
                    "Cleanup completed, store_size=%s",
                    len(self._by_job),
                )
            return

        ordered = sorted(self._by_job.items(), key=lambda kv: kv[1].created_at)

        evicted = 0
        while len(self._by_job) > self._max_jobs and ordered:
            job_id, _ = ordered.pop(0)
            self._by_job.pop(job_id, None)
            evicted += 1

        if evicted:
            log.warning(
                "Payload store exceeded max_jobs, evicted_oldest=%s, store_size=%s",
                evicted,
                len(self._by_job),
            )

    async def put(self, *, job_id: str, items: list[IngestItem]) -> None:
        now = self._now()

        log.debug(
            "Storing payload, job_id=%s, item_count=%s",
            job_id,
            len(items),
        )

        async with self._lock:
            self._cleanup_unlocked(now)
            self._by_job[job_id] = _PayloadRecord(
                items=list(items),
                created_at=now,
            )

            log.info(
                "Payload stored, job_id=%s, store_size=%s",
                job_id,
                len(self._by_job),
            )

    async def get(self, *, job_id: str) -> list[IngestItem]:
        now = self._now()

        log.debug("Fetching payload, job_id=%s", job_id)

        async with self._lock:
            self._cleanup_unlocked(now)
            rec = self._by_job.get(job_id)

            if rec is None:
                log.warning(
                    "Payload not found, job_id=%s, store_size=%s",
                    job_id,
                    len(self._by_job),
                )
                raise KeyError(job_id)

            log.debug(
                "Payload retrieved, job_id=%s, item_count=%s",
                job_id,
                len(rec.items),
            )

            return list(rec.items)

    async def delete(self, *, job_id: str) -> None:
        async with self._lock:
            existed = job_id in self._by_job
            self._by_job.pop(job_id, None)

            if existed:
                log.info(
                    "Payload deleted, job_id=%s, store_size=%s",
                    job_id,
                    len(self._by_job),
                )
            else:
                log.debug(
                    "Delete requested for missing payload, job_id=%s",
                    job_id,
                )
