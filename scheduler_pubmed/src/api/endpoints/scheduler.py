from __future__ import annotations

from fastapi import APIRouter, Request, status

from scheduler_pubmed.src.api.contracts import contracts
from scheduler_pubmed.src.api.mappers.scheduler_mapper import (
    to_api_scheduler_run_created,
    to_api_scheduler_status,
)
from scheduler_pubmed.src.api.models import schemas
from scheduler_pubmed.src.core.errors.errors import SystemError
from scheduler_pubmed.src.core.services.scheduler_runtime import SchedulerRuntimeService

router = APIRouter(prefix="/v1/pubmed/scheduler", tags=["Scheduler"])


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


@router.post(
    "/run",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=schemas.SchedulerRunCreated,
    operation_id="triggerSchedulerRun",
    responses={
        500: {"model": schemas.ErrorResponse},
    },
    summary="Manually trigger scheduler",
)
async def trigger_scheduler_run(
    request: Request,
    body: contracts.RunSchedulerRequest | None = None,
) -> contracts.RunSchedulerResponse:
    runtime = _get_runtime(request)
    reldate_days = body.reldate_days if body is not None else None
    run = await runtime.trigger_manual_run(reldate_days=reldate_days)
    return to_api_scheduler_run_created(run)


@router.get(
    "/status",
    response_model=schemas.SchedulerStatus,
    operation_id="getSchedulerStatus",
    responses={
        500: {"model": schemas.ErrorResponse},
    },
    summary="Get scheduler status",
)
async def get_scheduler_status(
    request: Request,
) -> contracts.GetSchedulerStatusResponse:
    runtime = _get_runtime(request)
    status_model = await runtime.get_status()
    return to_api_scheduler_status(status_model)
