from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scheduler_pubmed.src.core.domains.scheduler import (
    DoiExecutionStatus,
    QueryExecution,
    RunDoiResult,
    RunStatus,
    SchedulerRun,
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

    @staticmethod
    def _to_query_execution(model: DbQueryExecution) -> QueryExecution:
        return QueryExecution(
            query_execution_id=model.id,
            query_id=model.query_id,
            status=RunStatus(model.status),
            pubmed_result_count=model.pubmed_result_count,
            doi_resolved_count=model.doi_resolved_count,
            doi_skipped_exists_count=model.doi_skipped_exists_count,
            doi_enqueued_count=model.doi_enqueued_count,
            doi_failed_count=model.doi_failed_count,
            ingest_job_id=model.ingest_job_id,
        )

    @staticmethod
    def _to_scheduler_run(
        *,
        run_model: DbSchedulerRun,
        queries: list[QueryExecution],
    ) -> SchedulerRun:
        return SchedulerRun(
            run_id=run_model.id,
            status=RunStatus(run_model.status),
            started_at=run_model.started_at,
            finished_at=run_model.finished_at,
            queries=queries,
        )

    async def _queries_by_run_id(
        self, *, session: AsyncSession, run_ids: list[UUID]
    ) -> dict[UUID, list[QueryExecution]]:
        if not run_ids:
            return {}

        stmt = select(DbQueryExecution).where(DbQueryExecution.run_id.in_(run_ids))
        stmt = stmt.order_by(DbQueryExecution.started_at.asc())
        result = await session.execute(stmt)
        models = result.scalars().all()

        by_run_id: dict[UUID, list[QueryExecution]] = {run_id: [] for run_id in run_ids}
        for model in models:
            by_run_id.setdefault(model.run_id, []).append(self._to_query_execution(model))
        return by_run_id

    async def list_runs(
        self,
        *,
        status: RunStatus | None,
        from_at: datetime | None,
        to_at: datetime | None,
    ) -> list[SchedulerRun]:
        async with self._session_maker() as session:
            stmt = select(DbSchedulerRun)
            if status is not None:
                stmt = stmt.where(DbSchedulerRun.status == status.value)
            if from_at is not None:
                stmt = stmt.where(DbSchedulerRun.started_at >= from_at)
            if to_at is not None:
                stmt = stmt.where(DbSchedulerRun.started_at <= to_at)
            stmt = stmt.order_by(desc(DbSchedulerRun.started_at))

            result = await session.execute(stmt)
            run_models = result.scalars().all()
            if not run_models:
                return []

            run_ids = [model.id for model in run_models]
            queries_by_run_id = await self._queries_by_run_id(session=session, run_ids=run_ids)
            return [
                self._to_scheduler_run(
                    run_model=model,
                    queries=queries_by_run_id.get(model.id, []),
                )
                for model in run_models
            ]

    async def get_run(self, *, run_id: UUID) -> SchedulerRun | None:
        async with self._session_maker() as session:
            run_model = await session.get(DbSchedulerRun, run_id)
            if run_model is None:
                return None
            queries_by_run_id = await self._queries_by_run_id(
                session=session,
                run_ids=[run_id],
            )
            return self._to_scheduler_run(
                run_model=run_model,
                queries=queries_by_run_id.get(run_id, []),
            )

    async def list_run_dois(self, *, run_id: UUID) -> list[RunDoiResult] | None:
        async with self._session_maker() as session:
            run_model = await session.get(DbSchedulerRun, run_id)
            if run_model is None:
                return None

            stmt = select(DbQueryExecutionDoi).where(DbQueryExecutionDoi.run_id == run_id)
            stmt = stmt.order_by(DbQueryExecutionDoi.created_at.asc())
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [
                RunDoiResult(
                    doi=model.doi,
                    status=DoiExecutionStatus(model.status),
                    error_message=model.error_message,
                )
                for model in models
            ]

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
