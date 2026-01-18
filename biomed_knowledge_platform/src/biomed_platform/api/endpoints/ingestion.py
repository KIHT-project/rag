# src/biomed_platform/api/endpoints/ingestion.py
from __future__ import annotations

from fastapi import APIRouter, Header, Request, status

from biomed_platform.api.mappers.ingestion_mapper import (
    to_ingest_batch_command,
    to_ingest_job_accepted_response,
    to_ingest_job_status_response,
)
from biomed_platform.api.models.generated import schemas
from biomed_platform.common.logging import get_logger
from biomed_platform.common.middleware.trace import get_request_id
from biomed_platform.core.errors.errors import SystemError

log = get_logger(__name__)

router = APIRouter(prefix="/v1/ingest", tags=["Ingestion"])


def _resolve_effective_embedding_model_id(
    *, request: Request, body: schemas.IngestBatchRequest
) -> str:
    cfg = getattr(request.app.state, "settings", None)
    default_model_id: str | None = None

    if cfg is not None:
        rag_cfg = cfg.require_rag()
        emb_cfg = rag_cfg.get("embedding", {})
        if isinstance(emb_cfg, dict):
            default_model_id = emb_cfg.get("provider")

    requested = getattr(body, "embedding_model_id", None)
    effective = (requested or default_model_id or "").strip()
    if not effective:
        raise SystemError(
            code="missing_embedding_model_id",
            message="Missing embedding model id, set rag.embedding.provider"
            "or pass embedding_model_id",
            details=None,
            retryable=False,
        )

    log.debug(
        "Resolved embedding model id, provided_model_id=%r,"
        "default_model_id=%r, effective_model_id=%r",
        requested,
        default_model_id,
        effective,
    )
    return effective


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
    body: schemas.IngestBatchRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> schemas.IngestJobAcceptedResponse | schemas.ErrorResponse:
    request_id = get_request_id()

    log.info(
        "Ingest batch request received, items_count=%d, idempotency_key_present=%s",
        len(body.items),
        idempotency_key is not None,
    )

    effective_embedding_model_id = _resolve_effective_embedding_model_id(request=request, body=body)

    service = getattr(request.app.state, "ingestion_service", None)
    if service is None:
        raise SystemError(
            code="service_not_configured",
            message="Ingestion service not configured",
            details=None,
            retryable=False,
        )

    cmd = to_ingest_batch_command(
        request=body,
        effective_embedding_model_id=effective_embedding_model_id,
        idempotency_key=idempotency_key,
        correlation_id=request_id,
    )

    accepted = await service.ingest_batch(cmd)
    return to_ingest_job_accepted_response(accepted)


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
    job_id: str,
) -> schemas.IngestJobStatusResponse | schemas.ErrorResponse:

    service = getattr(request.app.state, "ingestion_service", None)
    if service is None:
        raise SystemError(
            code="service_not_configured",
            message="Ingestion service not configured",
            details=None,
            retryable=False,
        )

    job = await service.get_job_status(job_id=job_id)
    return to_ingest_job_status_response(job)
