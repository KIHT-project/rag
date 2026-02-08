from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from scheduler_pubmed.src.core.domains.scheduler import (
    DoiExecutionStatus,
    RunStatus,
    SchedulerRunCreated,
    SchedulerRunRecord,
    TriggerType,
)


class SchedulerRepository(Protocol):
    async def create_run(self, *, trigger_type: TriggerType) -> SchedulerRunCreated:
        raise NotImplementedError

    async def complete_run(self, *, run_id: UUID, status: RunStatus) -> None:
        raise NotImplementedError

    async def get_last_run(self) -> SchedulerRunRecord | None:
        raise NotImplementedError

    async def create_query_execution(self, *, run_id: UUID, query_id: UUID) -> UUID:
        raise NotImplementedError

    async def complete_query_execution(
        self,
        *,
        query_execution_id: UUID,
        status: RunStatus,
        pubmed_result_count: int,
        doi_resolved_count: int,
        doi_skipped_exists_count: int,
        doi_enqueued_count: int,
        doi_failed_count: int,
        ingest_job_id: UUID | None,
        error_message: str | None,
    ) -> None:
        raise NotImplementedError

    async def upsert_doi_execution_result(
        self,
        *,
        query_execution_id: UUID,
        run_id: UUID,
        doi: str,
        status: DoiExecutionStatus,
        error_message: str | None = None,
    ) -> None:
        raise NotImplementedError

    async def set_query_last_successful_run_at(
        self,
        *,
        query_id: UUID,
        value: datetime,
    ) -> None:
        raise NotImplementedError
