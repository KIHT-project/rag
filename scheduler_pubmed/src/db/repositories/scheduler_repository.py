from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scheduler_pubmed.src.core.domains.scheduler import (
    DoiExecutionStatus,
    RunStatus,
    SchedulerRunCreated,
    SchedulerRunRecord,
    TriggerType,
)
from scheduler_pubmed.src.db.models.scheduler import (
    PubMedQuery as DbPubMedQuery,
    QueryExecution as DbQueryExecution,
    QueryExecutionDoi as DbQueryExecutionDoi,
    SchedulerRun as DbSchedulerRun,
)


class SqlAlchemySchedulerRepository:
    def __init__(self, *, session_maker: async_sessionmaker[AsyncSession]) -> None:
        self._session_maker = session_maker

    async def create_run(self, *, trigger_type: TriggerType) -> SchedulerRunCreated:
        async with self._session_maker() as session:
            model = DbSchedulerRun(
                trigger_type=trigger_type.value,
                status=RunStatus.RUNNING.value,
            )
            session.add(model)
            await session.flush()
            await session.refresh(model)
            await session.commit()
            return SchedulerRunCreated(
                run_id=model.id,
                status=RunStatus(model.status),
                started_at=model.started_at,
            )

    async def complete_run(self, *, run_id: UUID, status: RunStatus) -> None:
        async with self._session_maker() as session:
            model = await session.get(DbSchedulerRun, run_id)
            if model is None:
                return
            model.status = status.value
            model.finished_at = datetime.now(UTC)
            await session.commit()

    async def get_last_run(self) -> SchedulerRunRecord | None:
        async with self._session_maker() as session:
            stmt = select(DbSchedulerRun).order_by(desc(DbSchedulerRun.started_at)).limit(1)
            result = await session.execute(stmt)
            model = result.scalar_one_or_none()
            if model is None:
                return None
            return SchedulerRunRecord(
                run_id=model.id,
                status=RunStatus(model.status),
                started_at=model.started_at,
                finished_at=model.finished_at,
            )

    async def create_query_execution(self, *, run_id: UUID, query_id: UUID) -> UUID:
        async with self._session_maker() as session:
            model = DbQueryExecution(
                run_id=run_id,
                query_id=query_id,
                status=RunStatus.RUNNING.value,
            )
            session.add(model)
            await session.flush()
            await session.refresh(model)
            await session.commit()
            return model.id

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
        async with self._session_maker() as session:
            model = await session.get(DbQueryExecution, query_execution_id)
            if model is None:
                return

            model.status = status.value
            model.pubmed_result_count = pubmed_result_count
            model.doi_resolved_count = doi_resolved_count
            model.doi_skipped_exists_count = doi_skipped_exists_count
            model.doi_enqueued_count = doi_enqueued_count
            model.doi_failed_count = doi_failed_count
            model.ingest_job_id = ingest_job_id
            model.error_message = error_message
            model.finished_at = datetime.now(UTC)
            await session.commit()

    async def upsert_doi_execution_result(
        self,
        *,
        query_execution_id: UUID,
        run_id: UUID,
        doi: str,
        status: DoiExecutionStatus,
        error_message: str | None = None,
    ) -> None:
        async with self._session_maker() as session:
            stmt = select(DbQueryExecutionDoi).where(
                DbQueryExecutionDoi.query_execution_id == query_execution_id,
                DbQueryExecutionDoi.doi == doi,
            )
            result = await session.execute(stmt)
            model = result.scalar_one_or_none()
            if model is None:
                model = DbQueryExecutionDoi(
                    query_execution_id=query_execution_id,
                    run_id=run_id,
                    doi=doi,
                    status=status.value,
                    error_message=error_message,
                )
                session.add(model)
            else:
                model.status = status.value
                model.error_message = error_message

            await session.commit()

    async def set_query_last_successful_run_at(
        self,
        *,
        query_id: UUID,
        value: datetime,
    ) -> None:
        async with self._session_maker() as session:
            model = await session.get(DbPubMedQuery, query_id)
            if model is None:
                return
            model.last_successful_run_at = value
            model.updated_at = datetime.now(UTC)
            await session.commit()
