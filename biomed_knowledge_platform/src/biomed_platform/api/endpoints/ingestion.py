from __future__ import annotations

from fastapi import APIRouter, Header, Request, Response, status

from biomed_platform.api.mappers.ingestion_mapper import (
    to_ingest_batch_command,
    to_ingest_job_accepted_response,
    to_ingest_job_status_response,
)
from biomed_platform.api.models.generated import schemas
from biomed_platform.common.middleware.trace import get_request_id
from biomed_platform.core.errors.errors import AppError
from biomed_platform.common.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/v1/ingest", tags=["Ingestion"])


def _resolve_effective_embedding_model_id(
    *,
    request: Request,
    body: schemas.IngestBatchRequest,
) -> str:
    cfg = getattr(request.app.state, "settings", None)
    default_model_id = None

    if cfg is not None:
        rag_cfg = cfg.require_rag()
        default_model_id = rag_cfg.get("default_embedding_model_id")

    effective = body.embedding_model_id or str(default_model_id or "").strip()

    log.debug(
        "Resolved embedding model id, "
        "provided_model_id=%r, "
        "default_model_id=%r, "
        "effective_model_id=%r",
        body.embedding_model_id,
        default_model_id,
        effective,
    )

    return effective


def _status_for_error_code(code: str) -> int:
    if code == "validation_error":
        return status.HTTP_400_BAD_REQUEST
    if code == "invalid_model_id":
        return status.HTTP_400_BAD_REQUEST
    if code == "duplicate_doi":
        return status.HTTP_409_CONFLICT
    if code == "not_found":
        return status.HTTP_404_NOT_FOUND
    if code == "too_many_requests":
        return status.HTTP_429_TOO_MANY_REQUESTS
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _to_error_response(*, request_id: str, err: AppError) -> schemas.ErrorResponse:
    return schemas.ErrorResponse(
        request_id=request_id,
        error=schemas.Error(err.code),
        message=err.message,
        details=err.details,
    )


@router.post(
    "/items",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"model": schemas.IngestJobAcceptedResponse},
        400: {"model": schemas.ErrorResponse},
        409: {"model": schemas.ErrorResponse},
        429: {"model": schemas.ErrorResponse},
    },
    summary="Ingest documents (abstract now, full text later), DOI only",
)
async def ingest_items(
    request: Request,
    response: Response,
    body: schemas.IngestBatchRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> schemas.IngestJobAcceptedResponse | schemas.ErrorResponse:
    request_id = get_request_id()

    log.info(
        "Ingest batch request received, items_count=%d, idempotency_key_present=%s",
        len(body.items),
        idempotency_key is not None,
    )

    try:
        effective_embedding_model_id = _resolve_effective_embedding_model_id(
            request=request,
            body=body,
        )

        service = getattr(request.app.state, "ingestion_service", None)
        if service is None:
            log.error("Ingestion service not configured")
            raise AppError(
                code="validation_error",
                message="Ingestion service not configured",
                details=None,
                retryable=False,
            )

        cmd = to_ingest_batch_command(
            request=body,
            effective_embedding_model_id=effective_embedding_model_id,
            idempotency_key=idempotency_key,
        )

        log.debug(
            "Dispatching ingest batch command, effective_embedding_model_id=%s",
            effective_embedding_model_id,
        )

        accepted = await service.ingest_batch(cmd)

        log.info(
            "Ingest batch accepted, job_id=%s",
            accepted.job_id,
        )

        return to_ingest_job_accepted_response(accepted)

    except AppError as err:
        http_status = _status_for_error_code(err.code)
        response.status_code = http_status

        if err.code == "too_many_requests":
            details = err.details or {}
            retry_after = details.get("retry_after_seconds")
            if isinstance(retry_after, int):
                response.headers["Retry-After"] = str(retry_after)

            log.warning(
                "Ingest request rate limited, retry_after_seconds=%r",
                retry_after,
            )
        else:
            log.warning(
                "Ingest request failed, error_code=%s, message=%s",
                err.code,
                err.message,
            )

        return _to_error_response(request_id=request_id, err=err)


@router.get(
    "/jobs/{job_id}",
    responses={
        200: {"model": schemas.IngestJobStatusResponse},
        404: {"model": schemas.ErrorResponse},
    },
    summary="Get ingestion job status",
)
async def get_job_status(
    request: Request,
    response: Response,
    job_id: str,
) -> schemas.IngestJobStatusResponse | schemas.ErrorResponse:
    request_id = get_request_id()

    log.info(
        "Job status request received,job_id=%s",
        job_id,
    )

    try:
        service = getattr(request.app.state, "ingestion_service", None)
        if service is None:
            log.error("Ingestion service not configured")
            raise AppError(
                code="validation_error",
                message="Ingestion service not configured",
                details=None,
                retryable=False,
            )

        job = await service.get_job_status(job_id=job_id)

        log.debug(
            "Job status retrieved, job_id=%s, job_state=%s",
            job_id,
            job.state,
        )

        return to_ingest_job_status_response(job)

    except AppError as err:
        response.status_code = _status_for_error_code(err.code)

        log.warning(
            "Job status request failed, job_id=%s, error_code=%s, message=%s",
            job_id,
            err.code,
            err.message,
        )

        return _to_error_response(request_id=request_id, err=err)
