from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class RunStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"


class TriggerType(StrEnum):
    SCHEDULED = "SCHEDULED"
    MANUAL = "MANUAL"


class DoiExecutionStatus(StrEnum):
    SKIPPED_EXISTS = "SKIPPED_EXISTS"
    ENQUEUED = "ENQUEUED"
    INGESTED = "INGESTED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class SchedulerRunCreated:
    run_id: UUID
    status: RunStatus
    started_at: datetime


@dataclass(frozen=True, slots=True)
class SchedulerRunRecord:
    run_id: UUID
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None


@dataclass(frozen=True, slots=True)
class SchedulerStatus:
    enabled: bool
    utc_schedule: list[str]
    next_run_at: datetime
    last_run_at: datetime | None
    last_run_status: RunStatus | None


@dataclass(frozen=True, slots=True)
class PubMedSearchResult:
    pmid: str
    doi: str | None
    published_at: datetime | None


@dataclass(frozen=True, slots=True)
class FetchBatchAccepted:
    job_id: str
    state: str


@dataclass(frozen=True, slots=True)
class IngestJobItemStatus:
    doi: str
    state: str
    message: str | None


@dataclass(frozen=True, slots=True)
class IngestJobStatus:
    job_id: str
    state: str
    items: list[IngestJobItemStatus]
