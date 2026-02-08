from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from scheduler_pubmed.src.core.domains.scheduler import DoiExecutionStatus, RunStatus, TriggerType
from scheduler_pubmed.src.db.models.scheduler import (
    PubMedQuery as DbPubMedQuery,
    QueryExecution as DbQueryExecution,
    QueryExecutionDoi as DbQueryExecutionDoi,
    SchedulerRun as DbSchedulerRun,
)
from scheduler_pubmed.src.db.repositories.scheduler_repository import SqlAlchemySchedulerRepository


class _FakeExecuteResult:
    def __init__(self, scalar_value):
        self._scalar_value = scalar_value

    def scalar_one_or_none(self):
        return self._scalar_value


class _FakeSession:
    def __init__(
        self,
        *,
        get_items: dict[tuple[type, UUID], object] | None = None,
        execute_scalar=None,
    ) -> None:
        self.get_items = get_items or {}
        self.execute_scalar = execute_scalar
        self.added: list[object] = []
        self.commits = 0
        self.flushes = 0
        self.refreshes = 0
        self.last_stmt = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def add(self, item: object) -> None:
        self.added.append(item)

    async def flush(self) -> None:
        self.flushes += 1

    async def refresh(self, item: object) -> None:
        self.refreshes += 1
        now = datetime.now(UTC)
        if getattr(item, "id", None) is None:
            item.id = uuid4()
        if getattr(item, "started_at", None) is None:
            item.started_at = now
        if getattr(item, "created_at", None) is None:
            item.created_at = now

    async def commit(self) -> None:
        self.commits += 1

    async def execute(self, stmt):
        self.last_stmt = stmt
        return _FakeExecuteResult(self.execute_scalar)

    async def get(self, model_cls, item_id: UUID):
        return self.get_items.get((model_cls, item_id))


class _FakeSessionMaker:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    def __call__(self):
        return self._session


@pytest.mark.asyncio
async def test_create_run_persists_running_record() -> None:
    session = _FakeSession()
    repository = SqlAlchemySchedulerRepository(session_maker=_FakeSessionMaker(session))  # type: ignore[arg-type]

    created = await repository.create_run(trigger_type=TriggerType.MANUAL)

    assert created.status == RunStatus.RUNNING
    assert len(session.added) == 1
    assert session.commits == 1
    model = session.added[0]
    assert isinstance(model, DbSchedulerRun)
    assert model.trigger_type == "MANUAL"


@pytest.mark.asyncio
async def test_complete_run_updates_existing_model() -> None:
    run_id = uuid4()
    model = DbSchedulerRun(
        id=run_id,
        trigger_type="MANUAL",
        status="RUNNING",
        started_at=datetime.now(UTC),
        finished_at=None,
        created_at=datetime.now(UTC),
    )
    session = _FakeSession(get_items={(DbSchedulerRun, run_id): model})
    repository = SqlAlchemySchedulerRepository(session_maker=_FakeSessionMaker(session))  # type: ignore[arg-type]

    await repository.complete_run(run_id=run_id, status=RunStatus.SUCCESS)

    assert model.status == "SUCCESS"
    assert model.finished_at is not None
    assert session.commits == 1


@pytest.mark.asyncio
async def test_complete_run_noop_when_missing() -> None:
    session = _FakeSession()
    repository = SqlAlchemySchedulerRepository(session_maker=_FakeSessionMaker(session))  # type: ignore[arg-type]

    await repository.complete_run(run_id=uuid4(), status=RunStatus.SUCCESS)

    assert session.commits == 0


@pytest.mark.asyncio
async def test_get_last_run_returns_none_when_empty() -> None:
    session = _FakeSession(execute_scalar=None)
    repository = SqlAlchemySchedulerRepository(session_maker=_FakeSessionMaker(session))  # type: ignore[arg-type]

    result = await repository.get_last_run()

    assert result is None


@pytest.mark.asyncio
async def test_get_last_run_maps_model() -> None:
    run_id = uuid4()
    model = DbSchedulerRun(
        id=run_id,
        trigger_type="SCHEDULED",
        status="PARTIAL_SUCCESS",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        created_at=datetime.now(UTC),
    )
    session = _FakeSession(execute_scalar=model)
    repository = SqlAlchemySchedulerRepository(session_maker=_FakeSessionMaker(session))  # type: ignore[arg-type]

    result = await repository.get_last_run()

    assert result is not None
    assert result.run_id == run_id
    assert result.status == RunStatus.PARTIAL_SUCCESS
    assert "ORDER BY" in str(session.last_stmt)


@pytest.mark.asyncio
async def test_create_query_execution_persists_running_record() -> None:
    session = _FakeSession()
    repository = SqlAlchemySchedulerRepository(session_maker=_FakeSessionMaker(session))  # type: ignore[arg-type]

    query_execution_id = await repository.create_query_execution(run_id=uuid4(), query_id=uuid4())

    assert isinstance(query_execution_id, UUID)
    assert len(session.added) == 1
    model = session.added[0]
    assert isinstance(model, DbQueryExecution)
    assert model.status == "RUNNING"
    assert session.commits == 1


@pytest.mark.asyncio
async def test_complete_query_execution_updates_existing_model() -> None:
    query_execution_id = uuid4()
    model = DbQueryExecution(
        id=query_execution_id,
        run_id=uuid4(),
        query_id=uuid4(),
        status="RUNNING",
        pubmed_result_count=0,
        doi_resolved_count=0,
        doi_skipped_exists_count=0,
        doi_enqueued_count=0,
        doi_failed_count=0,
        ingest_job_id=None,
        error_message=None,
        started_at=datetime.now(UTC),
        finished_at=None,
        created_at=datetime.now(UTC),
    )
    session = _FakeSession(get_items={(DbQueryExecution, query_execution_id): model})
    repository = SqlAlchemySchedulerRepository(session_maker=_FakeSessionMaker(session))  # type: ignore[arg-type]

    ingest_job_id = uuid4()
    await repository.complete_query_execution(
        query_execution_id=query_execution_id,
        status=RunStatus.PARTIAL_SUCCESS,
        pubmed_result_count=5,
        doi_resolved_count=4,
        doi_skipped_exists_count=1,
        doi_enqueued_count=2,
        doi_failed_count=1,
        ingest_job_id=ingest_job_id,
        error_message="warning",
    )

    assert model.status == "PARTIAL_SUCCESS"
    assert model.pubmed_result_count == 5
    assert model.doi_resolved_count == 4
    assert model.doi_skipped_exists_count == 1
    assert model.doi_enqueued_count == 2
    assert model.doi_failed_count == 1
    assert model.ingest_job_id == ingest_job_id
    assert model.error_message == "warning"
    assert model.finished_at is not None
    assert session.commits == 1


@pytest.mark.asyncio
async def test_complete_query_execution_noop_when_missing() -> None:
    session = _FakeSession()
    repository = SqlAlchemySchedulerRepository(session_maker=_FakeSessionMaker(session))  # type: ignore[arg-type]

    await repository.complete_query_execution(
        query_execution_id=uuid4(),
        status=RunStatus.SUCCESS,
        pubmed_result_count=0,
        doi_resolved_count=0,
        doi_skipped_exists_count=0,
        doi_enqueued_count=0,
        doi_failed_count=0,
        ingest_job_id=None,
        error_message=None,
    )

    assert session.commits == 0


@pytest.mark.asyncio
async def test_upsert_doi_execution_result_creates_when_missing() -> None:
    session = _FakeSession(execute_scalar=None)
    repository = SqlAlchemySchedulerRepository(session_maker=_FakeSessionMaker(session))  # type: ignore[arg-type]

    await repository.upsert_doi_execution_result(
        query_execution_id=uuid4(),
        run_id=uuid4(),
        doi="10.1000/a",
        status=DoiExecutionStatus.ENQUEUED,
        error_message=None,
    )

    assert len(session.added) == 1
    model = session.added[0]
    assert isinstance(model, DbQueryExecutionDoi)
    assert model.status == "ENQUEUED"
    assert session.commits == 1


@pytest.mark.asyncio
async def test_upsert_doi_execution_result_updates_existing() -> None:
    existing = DbQueryExecutionDoi(
        id=uuid4(),
        query_execution_id=uuid4(),
        run_id=uuid4(),
        doi="10.1000/a",
        status="ENQUEUED",
        error_message=None,
        created_at=datetime.now(UTC),
    )
    session = _FakeSession(execute_scalar=existing)
    repository = SqlAlchemySchedulerRepository(session_maker=_FakeSessionMaker(session))  # type: ignore[arg-type]

    await repository.upsert_doi_execution_result(
        query_execution_id=existing.query_execution_id,
        run_id=existing.run_id,
        doi="10.1000/a",
        status=DoiExecutionStatus.FAILED,
        error_message="boom",
    )

    assert existing.status == "FAILED"
    assert existing.error_message == "boom"
    assert len(session.added) == 0
    assert session.commits == 1


@pytest.mark.asyncio
async def test_set_query_last_successful_run_at_updates_existing() -> None:
    query_id = uuid4()
    model = DbPubMedQuery(
        id=query_id,
        pubmed_query="x",
        description="y",
        enabled=True,
        last_successful_run_at=None,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session = _FakeSession(get_items={(DbPubMedQuery, query_id): model})
    repository = SqlAlchemySchedulerRepository(session_maker=_FakeSessionMaker(session))  # type: ignore[arg-type]

    value = datetime.now(UTC)
    await repository.set_query_last_successful_run_at(query_id=query_id, value=value)

    assert model.last_successful_run_at == value
    assert model.updated_at is not None
    assert session.commits == 1


@pytest.mark.asyncio
async def test_set_query_last_successful_run_at_noop_when_missing() -> None:
    session = _FakeSession()
    repository = SqlAlchemySchedulerRepository(session_maker=_FakeSessionMaker(session))  # type: ignore[arg-type]

    await repository.set_query_last_successful_run_at(query_id=uuid4(), value=datetime.now(UTC))

    assert session.commits == 0
