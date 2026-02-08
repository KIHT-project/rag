from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from scheduler_pubmed.src.core.domains.pubmed_query import PubMedQuery
from scheduler_pubmed.src.core.domains.scheduler import (
    DoiExecutionStatus,
    FetchBatchAccepted,
    IngestJobItemStatus,
    IngestJobStatus,
    PubMedSearchResult,
    RunStatus,
    SchedulerRunCreated,
    SchedulerRunRecord,
    TriggerType,
)
from scheduler_pubmed.src.core.use_cases.scheduler import (
    SchedulerOrchestrationUseCase,
    _QueryExecutionState,
    _compute_next_run_at,
    _derive_query_status,
    _derive_run_status,
    _extract_unique_dois,
    _filter_incremental_results,
    _map_ingest_item_state,
    _parse_schedule_time,
    _to_uuid_or_none,
)


class _FakeQueryRepository:
    def __init__(self, queries: list[PubMedQuery]) -> None:
        self._queries = queries

    async def list(self, *, enabled: bool | None = None) -> list[PubMedQuery]:
        if enabled is None:
            return self._queries
        return [query for query in self._queries if query.enabled is enabled]


class _FakeSchedulerRepository:
    def __init__(self) -> None:
        self.last_created_run: SchedulerRunCreated | None = None
        self.completed_runs: list[tuple[UUID, RunStatus]] = []
        self.query_execution_ids: list[UUID] = []
        self.completed_query_executions: list[dict[str, object]] = []
        self.doi_results: dict[
            tuple[UUID, str], tuple[DoiExecutionStatus, str | None]
        ] = {}
        self.last_successful_updates: dict[UUID, datetime] = {}
        self.last_run_record: SchedulerRunRecord | None = None

    async def create_run(self, *, trigger_type: TriggerType) -> SchedulerRunCreated:
        run = SchedulerRunCreated(
            run_id=uuid4(),
            status=RunStatus.RUNNING,
            started_at=datetime.now(UTC),
        )
        self.last_created_run = run
        return run

    async def complete_run(self, *, run_id: UUID, status: RunStatus) -> None:
        self.completed_runs.append((run_id, status))
        self.last_run_record = SchedulerRunRecord(
            run_id=run_id,
            status=status,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )

    async def get_last_run(self) -> SchedulerRunRecord | None:
        return self.last_run_record

    async def create_query_execution(self, *, run_id: UUID, query_id: UUID) -> UUID:
        query_execution_id = uuid4()
        self.query_execution_ids.append(query_execution_id)
        return query_execution_id

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
        self.completed_query_executions.append(
            {
                "query_execution_id": query_execution_id,
                "status": status,
                "pubmed_result_count": pubmed_result_count,
                "doi_resolved_count": doi_resolved_count,
                "doi_skipped_exists_count": doi_skipped_exists_count,
                "doi_enqueued_count": doi_enqueued_count,
                "doi_failed_count": doi_failed_count,
                "ingest_job_id": ingest_job_id,
                "error_message": error_message,
            }
        )

    async def upsert_doi_execution_result(
        self,
        *,
        query_execution_id: UUID,
        run_id: UUID,
        doi: str,
        status: DoiExecutionStatus,
        error_message: str | None = None,
    ) -> None:
        self.doi_results[(query_execution_id, doi)] = (status, error_message)

    async def set_query_last_successful_run_at(
        self, *, query_id: UUID, value: datetime
    ) -> None:
        self.last_successful_updates[query_id] = value


class _FakePubMedClient:
    def __init__(self, *, results: list[PubMedSearchResult]) -> None:
        self._results = results
        self.received_queries: list[str] = []
        self.received_reldate_days: list[int | None] = []

    async def search(
        self,
        *,
        query: str,
        reldate_days: int | None = None,
    ) -> list[PubMedSearchResult]:
        self.received_queries.append(query)
        self.received_reldate_days.append(reldate_days)
        return self._results


class _FakeDocumentsClient:
    def __init__(
        self,
        *,
        existing_dois: set[str] | None = None,
        fail_on_exists_for: set[str] | None = None,
        ingest_state: str = "succeeded",
    ) -> None:
        self.existing_dois = {item.lower() for item in (existing_dois or set())}
        self.fail_on_exists_for = {
            item.lower() for item in (fail_on_exists_for or set())
        }
        self.ingest_state = ingest_state
        self.batches: list[list[str]] = []

    async def document_exists(self, *, doi: str) -> bool:
        key = doi.lower()
        if key in self.fail_on_exists_for:
            raise RuntimeError("exists check failed")
        return key in self.existing_dois

    async def fetch_batch(self, *, dois: list[str]) -> FetchBatchAccepted:
        self.batches.append(dois)
        return FetchBatchAccepted(job_id=str(uuid4()), state="queued")

    async def get_ingest_job_status(self, *, job_id: str) -> IngestJobStatus:
        return IngestJobStatus(
            job_id=job_id,
            state=self.ingest_state,
            items=[
                IngestJobItemStatus(doi=doi, state="succeeded", message=None)
                for doi in (self.batches[-1] if self.batches else [])
            ],
        )


def _make_query(*, last_successful_run_at: datetime | None = None) -> PubMedQuery:
    now = datetime.now(UTC)
    return PubMedQuery(
        id=uuid4(),
        pubmed_query="thrombosis",
        description="desc",
        enabled=True,
        last_successful_run_at=last_successful_run_at,
        created_at=now,
        updated_at=now,
    )


def test_parse_schedule_time_rejects_invalid_values() -> None:
    assert _parse_schedule_time("02:45") == (2, 45)
    with pytest.raises(ValueError):
        _parse_schedule_time("0245")
    with pytest.raises(ValueError):
        _parse_schedule_time("24:00")


def test_compute_next_run_at_empty_schedule_returns_now() -> None:
    now = datetime(2026, 2, 8, 10, 0, tzinfo=UTC)
    assert _compute_next_run_at(now=now, utc_schedule=[]) == now


def test_to_uuid_or_none_handles_blank_and_invalid() -> None:
    assert _to_uuid_or_none(None) is None
    assert _to_uuid_or_none("") is None
    assert _to_uuid_or_none("not-a-uuid") is None
    value = str(uuid4())
    assert _to_uuid_or_none(value) == UUID(value)


def test_derive_status_helpers_cover_empty_and_zero_resolved() -> None:
    assert _derive_query_status(resolved_count=0, failed_count=0) == RunStatus.SUCCESS
    assert _derive_run_status([]) == RunStatus.SUCCESS


def test_filter_incremental_results_and_unique_doi_extraction() -> None:
    since = datetime(2026, 2, 8, 10, 0, tzinfo=UTC)
    results = [
        PubMedSearchResult(
            pmid="1", doi=None, published_at=since + timedelta(minutes=1)
        ),
        PubMedSearchResult(
            pmid="2", doi="   ", published_at=since + timedelta(minutes=1)
        ),
        PubMedSearchResult(
            pmid="3", doi="10.1/A", published_at=since + timedelta(minutes=1)
        ),
        PubMedSearchResult(
            pmid="4", doi="10.1/a", published_at=since + timedelta(minutes=2)
        ),
        PubMedSearchResult(pmid="5", doi="10.1/B", published_at=None),
        PubMedSearchResult(
            pmid="6", doi="10.1/C", published_at=since - timedelta(minutes=1)
        ),
    ]

    filtered = _filter_incremental_results(results=results, since=since)
    dois = _extract_unique_dois(results=filtered)

    assert [item.pmid for item in filtered] == ["1", "2", "3", "4"]
    assert dois == ["10.1/A"]


def test_map_ingest_item_state_variants() -> None:
    failed, failed_error = _map_ingest_item_state(
        IngestJobItemStatus(doi="10.1/a", state="failed", message="fail")
    )
    skipped, skipped_error = _map_ingest_item_state(
        IngestJobItemStatus(doi="10.1/a", state="skipped_duplicate", message="exists")
    )
    fallback, fallback_error = _map_ingest_item_state(
        IngestJobItemStatus(doi="10.1/a", state="queued", message="queued")
    )

    assert (failed, failed_error) == (DoiExecutionStatus.FAILED, "fail")
    assert (skipped, skipped_error) == (DoiExecutionStatus.SKIPPED_EXISTS, "exists")
    assert (fallback, fallback_error) == (DoiExecutionStatus.ENQUEUED, "queued")


@pytest.mark.asyncio
async def test_execute_run_success_updates_counts_and_last_successful() -> None:
    query = _make_query()
    pubmed_results = [
        PubMedSearchResult(pmid="1", doi="10.1000/a", published_at=datetime.now(UTC))
    ]

    scheduler_repo = _FakeSchedulerRepository()
    use_case = SchedulerOrchestrationUseCase(
        query_repository=_FakeQueryRepository([query]),
        scheduler_repository=scheduler_repo,
        pubmed_client=_FakePubMedClient(results=pubmed_results),
        documents_client=_FakeDocumentsClient(existing_dois=set()),
        ingest_poll_max_attempts=1,
        ingest_poll_interval_seconds=0.0,
    )

    run_id = uuid4()
    await use_case.execute_run(run_id=run_id, reldate_days=14)

    assert scheduler_repo.completed_runs[-1] == (run_id, RunStatus.SUCCESS)
    summary = scheduler_repo.completed_query_executions[-1]
    assert summary["status"] == RunStatus.SUCCESS
    assert summary["pubmed_result_count"] == 1
    assert summary["doi_resolved_count"] == 1
    assert summary["doi_enqueued_count"] == 1
    assert summary["doi_failed_count"] == 0
    assert query.id in scheduler_repo.last_successful_updates
    assert use_case._pubmed_client.received_reldate_days == [14]  # type: ignore[attr-defined] # noqa: SLF001


@pytest.mark.asyncio
async def test_execute_run_partial_success_when_some_doi_fail() -> None:
    query = _make_query()
    pubmed_results = [
        PubMedSearchResult(pmid="1", doi="10.1000/a", published_at=datetime.now(UTC)),
        PubMedSearchResult(pmid="2", doi="10.1000/b", published_at=datetime.now(UTC)),
    ]

    scheduler_repo = _FakeSchedulerRepository()
    use_case = SchedulerOrchestrationUseCase(
        query_repository=_FakeQueryRepository([query]),
        scheduler_repository=scheduler_repo,
        pubmed_client=_FakePubMedClient(results=pubmed_results),
        documents_client=_FakeDocumentsClient(
            existing_dois={"10.1000/a"}, fail_on_exists_for={"10.1000/b"}
        ),
        ingest_poll_max_attempts=1,
        ingest_poll_interval_seconds=0.0,
    )

    run_id = uuid4()
    await use_case.execute_run(run_id=run_id, reldate_days=None)

    assert scheduler_repo.completed_runs[-1] == (run_id, RunStatus.PARTIAL_SUCCESS)
    summary = scheduler_repo.completed_query_executions[-1]
    assert summary["status"] == RunStatus.PARTIAL_SUCCESS
    assert summary["doi_skipped_exists_count"] == 1
    assert summary["doi_failed_count"] == 1
    assert query.id in scheduler_repo.last_successful_updates


@pytest.mark.asyncio
async def test_execute_run_failed_when_all_dois_fail() -> None:
    query = _make_query()
    pubmed_results = [
        PubMedSearchResult(pmid="1", doi="10.1000/a", published_at=datetime.now(UTC))
    ]

    scheduler_repo = _FakeSchedulerRepository()
    use_case = SchedulerOrchestrationUseCase(
        query_repository=_FakeQueryRepository([query]),
        scheduler_repository=scheduler_repo,
        pubmed_client=_FakePubMedClient(results=pubmed_results),
        documents_client=_FakeDocumentsClient(fail_on_exists_for={"10.1000/a"}),
        ingest_poll_max_attempts=1,
        ingest_poll_interval_seconds=0.0,
    )

    run_id = uuid4()
    await use_case.execute_run(run_id=run_id, reldate_days=7)

    assert scheduler_repo.completed_runs[-1] == (run_id, RunStatus.FAILED)
    summary = scheduler_repo.completed_query_executions[-1]
    assert summary["status"] == RunStatus.FAILED
    assert query.id not in scheduler_repo.last_successful_updates


@pytest.mark.asyncio
async def test_get_status_computes_next_run_and_last_run() -> None:
    scheduler_repo = _FakeSchedulerRepository()
    scheduler_repo.last_run_record = SchedulerRunRecord(
        run_id=uuid4(),
        status=RunStatus.SUCCESS,
        started_at=datetime.now(UTC) - timedelta(hours=2),
        finished_at=datetime.now(UTC) - timedelta(hours=1),
    )

    use_case = SchedulerOrchestrationUseCase(
        query_repository=_FakeQueryRepository([]),
        scheduler_repository=scheduler_repo,
        pubmed_client=_FakePubMedClient(results=[]),
        documents_client=_FakeDocumentsClient(),
    )

    status = await use_case.get_status(enabled=True, utc_schedule=["02:00", "14:00"])

    assert status.enabled is True
    assert status.utc_schedule == ["02:00", "14:00"]
    assert status.last_run_status == RunStatus.SUCCESS
    assert isinstance(status.next_run_at, datetime)


@pytest.mark.asyncio
async def test_trigger_run_delegates_to_repository() -> None:
    scheduler_repo = _FakeSchedulerRepository()
    use_case = SchedulerOrchestrationUseCase(
        query_repository=_FakeQueryRepository([]),
        scheduler_repository=scheduler_repo,
        pubmed_client=_FakePubMedClient(results=[]),
        documents_client=_FakeDocumentsClient(),
    )

    run = await use_case.trigger_run(trigger_type=TriggerType.MANUAL)

    assert run.status == RunStatus.RUNNING
    assert scheduler_repo.last_created_run is not None


@pytest.mark.asyncio
async def test_execute_run_marks_failed_when_query_listing_raises() -> None:
    class _FailingQueryRepository:
        async def list(self, *, enabled: bool | None = None):
            raise RuntimeError("query list failed")

    scheduler_repo = _FakeSchedulerRepository()
    use_case = SchedulerOrchestrationUseCase(
        query_repository=_FailingQueryRepository(),  # type: ignore[arg-type]
        scheduler_repository=scheduler_repo,
        pubmed_client=_FakePubMedClient(results=[]),
        documents_client=_FakeDocumentsClient(),
    )

    run_id = uuid4()
    await use_case.execute_run(run_id=run_id, reldate_days=None)

    assert scheduler_repo.completed_runs[-1] == (run_id, RunStatus.FAILED)


@pytest.mark.asyncio
async def test_poll_ingest_job_returns_none_when_status_call_fails() -> None:
    class _FailingDocumentsClient(_FakeDocumentsClient):
        async def get_ingest_job_status(self, *, job_id: str) -> IngestJobStatus:
            raise RuntimeError("unavailable")

    use_case = SchedulerOrchestrationUseCase(
        query_repository=_FakeQueryRepository([]),
        scheduler_repository=_FakeSchedulerRepository(),
        pubmed_client=_FakePubMedClient(results=[]),
        documents_client=_FailingDocumentsClient(),
        ingest_poll_max_attempts=2,
        ingest_poll_interval_seconds=0.0,
    )

    result = await use_case._poll_ingest_job(job_id=str(uuid4()))  # noqa: SLF001
    assert result is None


@pytest.mark.asyncio
async def test_poll_ingest_job_returns_latest_after_max_attempts() -> None:
    class _RunningDocumentsClient(_FakeDocumentsClient):
        async def get_ingest_job_status(self, *, job_id: str) -> IngestJobStatus:
            return IngestJobStatus(job_id=job_id, state="running", items=[])

    use_case = SchedulerOrchestrationUseCase(
        query_repository=_FakeQueryRepository([]),
        scheduler_repository=_FakeSchedulerRepository(),
        pubmed_client=_FakePubMedClient(results=[]),
        documents_client=_RunningDocumentsClient(),
        ingest_poll_max_attempts=2,
        ingest_poll_interval_seconds=0.0,
    )

    result = await use_case._poll_ingest_job(job_id=str(uuid4()))  # noqa: SLF001
    assert result is not None
    assert result.state == "running"


@pytest.mark.asyncio
async def test_apply_ingest_updates_skips_unknown_doi_items() -> None:
    scheduler_repo = _FakeSchedulerRepository()
    use_case = SchedulerOrchestrationUseCase(
        query_repository=_FakeQueryRepository([]),
        scheduler_repository=scheduler_repo,
        pubmed_client=_FakePubMedClient(results=[]),
        documents_client=_FakeDocumentsClient(),
    )

    state = _QueryExecutionState(
        filtered_results=[],
        resolved_dois=["10.1/a"],
        doi_statuses={},
        enqueued_doi_count=1,
        ingest_job_id=None,
        query_error_message=None,
    )

    await use_case._apply_ingest_job_updates(  # noqa: SLF001
        run_id=uuid4(),
        query_execution_id=uuid4(),
        to_enqueue=["10.1/a"],
        ingest_job=IngestJobStatus(
            job_id=str(uuid4()),
            state="succeeded",
            items=[IngestJobItemStatus(doi="10.1/b", state="succeeded", message=None)],
        ),
        state=state,
    )

    assert state.doi_statuses == {}


@pytest.mark.asyncio
async def test_enqueue_new_documents_keeps_enqueued_when_poll_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler_repo = _FakeSchedulerRepository()
    use_case = SchedulerOrchestrationUseCase(
        query_repository=_FakeQueryRepository([]),
        scheduler_repository=scheduler_repo,
        pubmed_client=_FakePubMedClient(results=[]),
        documents_client=_FakeDocumentsClient(),
    )

    async def _poll_none(*, job_id: str):  # noqa: ARG001
        return None

    monkeypatch.setattr(use_case, "_poll_ingest_job", _poll_none)

    state = _QueryExecutionState(
        filtered_results=[],
        resolved_dois=["10.1/a"],
        doi_statuses={},
        enqueued_doi_count=0,
        ingest_job_id=None,
        query_error_message=None,
    )
    query_execution_id = uuid4()
    await use_case._enqueue_new_documents(  # noqa: SLF001
        run_id=uuid4(),
        query_execution_id=query_execution_id,
        to_enqueue=["10.1/a"],
        state=state,
    )

    assert state.doi_statuses[("10.1/a")] == (DoiExecutionStatus.ENQUEUED, None)
    assert scheduler_repo.doi_results[(query_execution_id, "10.1/a")] == (
        DoiExecutionStatus.ENQUEUED,
        None,
    )


@pytest.mark.asyncio
async def test_execute_query_marks_to_enqueue_failed_when_enqueue_errors() -> None:
    class _FailingEnqueueDocumentsClient(_FakeDocumentsClient):
        async def fetch_batch(self, *, dois: list[str]) -> FetchBatchAccepted:
            raise RuntimeError("enqueue failed")

    query = _make_query()
    scheduler_repo = _FakeSchedulerRepository()
    use_case = SchedulerOrchestrationUseCase(
        query_repository=_FakeQueryRepository([query]),
        scheduler_repository=scheduler_repo,
        pubmed_client=_FakePubMedClient(
            results=[
                PubMedSearchResult(
                    pmid="1", doi="10.1000/a", published_at=datetime.now(UTC)
                )
            ]
        ),
        documents_client=_FailingEnqueueDocumentsClient(),
        ingest_poll_max_attempts=1,
        ingest_poll_interval_seconds=0.0,
    )

    status = await use_case._execute_query(  # noqa: SLF001
        run_id=uuid4(),
        query=query,
        reldate_days=30,
    )

    assert status == RunStatus.FAILED
    summary = scheduler_repo.completed_query_executions[-1]
    assert summary["doi_failed_count"] == 1
    assert summary["error_message"] is None


@pytest.mark.asyncio
async def test_execute_query_sets_query_error_when_pubmed_search_fails() -> None:
    class _FailingPubMedClient(_FakePubMedClient):
        async def search(
            self,
            *,
            query: str,  # noqa: ARG002
            reldate_days: int | None = None,  # noqa: ARG002
        ) -> list[PubMedSearchResult]:  # noqa: ARG002
            raise RuntimeError("pubmed failed")

    query = _make_query()
    scheduler_repo = _FakeSchedulerRepository()
    use_case = SchedulerOrchestrationUseCase(
        query_repository=_FakeQueryRepository([query]),
        scheduler_repository=scheduler_repo,
        pubmed_client=_FailingPubMedClient(results=[]),
        documents_client=_FakeDocumentsClient(),
        ingest_poll_max_attempts=1,
        ingest_poll_interval_seconds=0.0,
    )

    status = await use_case._execute_query(  # noqa: SLF001
        run_id=uuid4(),
        query=query,
        reldate_days=None,
    )

    assert status == RunStatus.FAILED
    summary = scheduler_repo.completed_query_executions[-1]
    assert summary["doi_resolved_count"] == 0
    assert summary["doi_failed_count"] == 0
    assert summary["error_message"] == "pubmed failed"
