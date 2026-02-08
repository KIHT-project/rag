from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Path, Query, Request

from scheduler_pubmed.src.api.contracts import contracts
from scheduler_pubmed.src.api.mappers.scheduler_mapper import (
    to_api_run_doi_result_list,
    to_api_scheduler_run,
    to_api_scheduler_run_list,
)
from scheduler_pubmed.src.api.models import schemas
from scheduler_pubmed.src.core.domains.scheduler import RunStatus
from scheduler_pubmed.src.core.errors.errors import SystemError
from scheduler_pubmed.src.core.services.scheduler_runtime import SchedulerRuntimeService

router = APIRouter(prefix="/v1/pubmed/runs", tags=["Runs"])


def _get_runtime(request: Request) -> SchedulerRuntimeService:
    runtime = getattr(request.app.state, "scheduler_runtime", None)
    if runtime is None:
        raise SystemError(
            code="service_not_configured",
            message="Scheduler runtime service is not configured",
            details=None,
            retryable=False,
        )
    return runtime


@router.get(
    "",
    response_model=list[schemas.SchedulerRun],
    operation_id="listSchedulerRuns",
    responses={
        400: {"model": schemas.ErrorResponse},
        422: {"model": schemas.ErrorResponse},
        500: {"model": schemas.ErrorResponse},
    },
    summary="List scheduler runs",
)
async def list_scheduler_runs(
    request: Request,
    status: RunStatus | None = Query(default=None),
    from_at: datetime | None = Query(default=None, alias="from"),
    to_at: datetime | None = Query(default=None, alias="to"),
) -> contracts.ListSchedulerRunsResponse:
    runtime = _get_runtime(request)
    runs = await runtime.list_runs(status=status, from_at=from_at, to_at=to_at)
    return to_api_scheduler_run_list(runs)


@router.get(
    "/{runId}",
    response_model=schemas.SchedulerRun,
    operation_id="getSchedulerRun",
    responses={
        404: {"model": schemas.ErrorResponse},
        422: {"model": schemas.ErrorResponse},
        500: {"model": schemas.ErrorResponse},
    },
    summary="Get scheduler run by id",
)
async def get_scheduler_run(
    request: Request,
    run_id: UUID = Path(alias="runId"),
) -> contracts.GetSchedulerRunResponse:
    runtime = _get_runtime(request)
    run = await runtime.get_run(run_id=run_id)
    return to_api_scheduler_run(run)


@router.get(
    "/{runId}/dois",
    response_model=list[schemas.RunDoiResult],
    operation_id="listRunDois",
    responses={
        404: {"model": schemas.ErrorResponse},
        422: {"model": schemas.ErrorResponse},
        500: {"model": schemas.ErrorResponse},
    },
    summary="DOI results for a run",
)
async def list_run_dois(
    request: Request,
    run_id: UUID = Path(alias="runId"),
) -> contracts.ListRunDoisResponse:
    runtime = _get_runtime(request)
    doi_results = await runtime.list_run_dois(run_id=run_id)
    return to_api_run_doi_result_list(doi_results)
