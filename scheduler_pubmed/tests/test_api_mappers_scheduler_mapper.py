from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from scheduler_pubmed.src.api.mappers.scheduler_mapper import (
    to_api_run_doi_result,
    to_api_scheduler_run,
    to_api_scheduler_run_created,
    to_api_scheduler_run_list,
    to_api_scheduler_status,
)
from scheduler_pubmed.src.core.domains.scheduler import (
    DoiExecutionStatus,
    QueryExecution,
    RunDoiResult,
    RunStatus,
    SchedulerRun,
    SchedulerRunCreated,
    SchedulerStatus,
)


def test_to_api_scheduler_run_created_maps_fields() -> None:
    model = SchedulerRunCreated(
        run_id=uuid4(),
        status=RunStatus.RUNNING,
        started_at=datetime.now(UTC),
    )

    response = to_api_scheduler_run_created(model)

    assert response.run_id == model.run_id
    assert response.status == "RUNNING"
    assert response.started_at == model.started_at


def test_to_api_scheduler_status_maps_fields() -> None:
    model = SchedulerStatus(
        enabled=True,
        utc_schedule=["02:00", "14:00"],
        next_run_at=datetime.now(UTC),
        last_run_at=datetime.now(UTC),
        last_run_status=RunStatus.SUCCESS,
    )

    response = to_api_scheduler_status(model)

    assert response.enabled is True
    assert response.utc_schedule == ["02:00", "14:00"]
    assert response.last_run_status == "SUCCESS"


def test_to_api_scheduler_run_and_list_map_fields() -> None:
    model = SchedulerRun(
        run_id=uuid4(),
        status=RunStatus.PARTIAL_SUCCESS,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        queries=[
            QueryExecution(
                query_execution_id=uuid4(),
                query_id=uuid4(),
                status=RunStatus.FAILED,
                pubmed_result_count=10,
                doi_resolved_count=5,
                doi_skipped_exists_count=2,
                doi_enqueued_count=3,
                doi_failed_count=1,
                ingest_job_id=None,
            )
        ],
    )

    single = to_api_scheduler_run(model)
    listing = to_api_scheduler_run_list([model])

    assert single.run_id == model.run_id
    assert single.status == "PARTIAL_SUCCESS"
    assert single.queries[0].status == "FAILED"
    assert len(listing) == 1
    assert listing[0].run_id == model.run_id


def test_to_api_run_doi_result_maps_fields() -> None:
    model = RunDoiResult(
        doi="10.1000/a",
        status=DoiExecutionStatus.FAILED,
        error_message="boom",
    )

    response = to_api_run_doi_result(model)

    assert response.doi == "10.1000/a"
    assert response.status == "FAILED"
    assert response.error_message == "boom"
