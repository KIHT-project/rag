from __future__ import annotations

from scheduler_pubmed.src.api.models import schemas
from scheduler_pubmed.src.core.domains.scheduler import (
    QueryExecution,
    RunDoiResult,
    SchedulerRun,
    SchedulerRunCreated,
    SchedulerStatus,
)


def to_api_scheduler_run_created(model: SchedulerRunCreated) -> schemas.SchedulerRunCreated:
    return schemas.SchedulerRunCreated(
        run_id=model.run_id,
        status=model.status.value,
        started_at=model.started_at,
    )


def to_api_scheduler_status(model: SchedulerStatus) -> schemas.SchedulerStatus:
    return schemas.SchedulerStatus(
        enabled=model.enabled,
        utc_schedule=model.utc_schedule,
        next_run_at=model.next_run_at,
        last_run_at=model.last_run_at,
        last_run_status=model.last_run_status.value if model.last_run_status else None,
    )


def to_api_query_execution(model: QueryExecution) -> schemas.QueryExecution:
    return schemas.QueryExecution(
        query_execution_id=model.query_execution_id,
        query_id=model.query_id,
        status=model.status.value,
        pubmed_result_count=model.pubmed_result_count,
        doi_resolved_count=model.doi_resolved_count,
        doi_skipped_exists_count=model.doi_skipped_exists_count,
        doi_enqueued_count=model.doi_enqueued_count,
        doi_failed_count=model.doi_failed_count,
        ingest_job_id=model.ingest_job_id,
    )


def to_api_scheduler_run(model: SchedulerRun) -> schemas.SchedulerRun:
    return schemas.SchedulerRun(
        run_id=model.run_id,
        status=model.status.value,
        started_at=model.started_at,
        finished_at=model.finished_at,
        queries=[to_api_query_execution(item) for item in model.queries],
    )


def to_api_scheduler_run_list(models: list[SchedulerRun]) -> list[schemas.SchedulerRun]:
    return [to_api_scheduler_run(model) for model in models]


def to_api_run_doi_result(model: RunDoiResult) -> schemas.RunDoiResult:
    return schemas.RunDoiResult(
        doi=model.doi,
        status=model.status.value,
        error_message=model.error_message,
    )


def to_api_run_doi_result_list(
    models: list[RunDoiResult],
) -> list[schemas.RunDoiResult]:
    return [to_api_run_doi_result(model) for model in models]
