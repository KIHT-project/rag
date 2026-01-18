from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.ingestion import IngestionJob
from biomed_platform.core.ports.ingestion import IngestionJobStore

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _JobRecord:
    job: IngestionJob
    completed_at: datetime | None


class InMemoryIngestionJobStore(IngestionJobStore):
    def __init__(
        self, *, ttl_seconds_after_completion: int = 86400, max_jobs: int = 10_000
    ) -> None:
        if ttl_seconds_after_completion <= 0:
            raise ValueError("ttl_seconds_after_completion must be positive")
        if max_jobs <= 0:
            raise ValueError("max_jobs must be positive")

        self._ttl = timedelta(seconds=ttl_seconds_after_completion)
        self._max_jobs = max_jobs
        self._jobs: dict[str, _JobRecord] = {}

        log.debug(
            "Ingestion job store initialized, ttl_seconds_after_completion=%d, max_jobs=%d",
            ttl_seconds_after_completion,
            max_jobs,
        )

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _is_completed(self, job: IngestionJob) -> bool:
        return job.state.value in ("succeeded", "failed", "partial")

    def _normalize_utc(self, dt: datetime) -> datetime:
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

    def _expired_job_ids(self, *, now: datetime) -> list[str]:
        expired: list[str] = []
        for job_id, rec in self._jobs.items():
            if rec.completed_at is None:
                continue
            completed_at = self._normalize_utc(rec.completed_at)
            if (now - completed_at) > self._ttl:
                expired.append(job_id)
        return expired

    def _remove_jobs(self, job_ids: list[str]) -> None:
        for job_id in job_ids:
            self._jobs.pop(job_id, None)

    def _ttl_cleanup(self, *, now: datetime) -> None:
        expired = self._expired_job_ids(now=now)
        if not expired:
            return

        self._remove_jobs(expired)

        log.debug(
            "Job store TTL cleanup removed jobs, count=%d",
            len(expired),
        )

    def _completed_jobs_sorted_oldest_first(self) -> list[tuple[datetime, str]]:
        completed: list[tuple[datetime, str]] = []
        for job_id, rec in self._jobs.items():
            if rec.completed_at is None:
                continue
            completed.append((self._normalize_utc(rec.completed_at), job_id))
        completed.sort(key=lambda x: x[0])
        return completed

    def _capacity_cleanup(self) -> None:
        if len(self._jobs) <= self._max_jobs:
            return

        completed = self._completed_jobs_sorted_oldest_first()

        removed = 0
        while len(self._jobs) > self._max_jobs and completed:
            _, job_id = completed.pop(0)
            self._jobs.pop(job_id, None)
            removed += 1

        if removed:
            log.warning(
                "Job store capacity cleanup evicted completed jobs, count=%d, max_jobs=%d",
                removed,
                self._max_jobs,
            )

    def _cleanup(self, now: datetime) -> None:
        self._ttl_cleanup(now=now)
        self._capacity_cleanup()

    async def create(self, job: IngestionJob) -> None:
        now = self._now()
        self._cleanup(now)

        completed_at = now if self._is_completed(job) else None
        replaced = job.job_id in self._jobs

        self._jobs[job.job_id] = _JobRecord(
            job=job,
            completed_at=completed_at,
        )

        log.debug(
            "Job created, job_id=%s, state=%s, completed=%s, replaced_existing=%s",
            job.job_id,
            job.state.value,
            completed_at is not None,
            replaced,
        )

    async def get(self, job_id: str) -> IngestionJob:
        now = self._now()
        self._cleanup(now)

        rec = self._jobs.get(job_id)
        if rec is None:
            log.debug(
                "Job not found, job_id=%s",
                job_id,
            )
            raise KeyError(job_id)

        log.debug(
            "Job retrieved, job_id=%s, state=%s",
            job_id,
            rec.job.state.value,
        )
        return rec.job

    async def update(self, job: IngestionJob) -> None:
        now = self._now()
        self._cleanup(now)

        completed_at = now if self._is_completed(job) else None
        existed = job.job_id in self._jobs

        self._jobs[job.job_id] = _JobRecord(
            job=job,
            completed_at=completed_at,
        )

        log.debug(
            "Job updated, job_id=%s, state=%s, completed=%s, existed=%s",
            job.job_id,
            job.state.value,
            completed_at is not None,
            existed,
        )

    async def delete(self, job_id: str) -> None:
        removed = self._jobs.pop(job_id, None) is not None

        log.debug(
            "Job deleted, job_id=%s, existed=%s",
            job_id,
            removed,
        )
