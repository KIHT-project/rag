from __future__ import annotations

from fastapi import APIRouter, Request, status

from biomed_platform.api.models.generated import schemas
from biomed_platform.common.logging import get_logger
from biomed_platform.common.middleware.trace import get_request_id
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.use_cases.search import SearchUseCase

log = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Retrieval"])


def _resolve_effective_embedding_model_id(*, request: Request) -> str:
    cfg = getattr(request.app.state, "settings", None)
    default_model_id: str | None = None

    if cfg is not None:
        rag_cfg = cfg.require_rag()
        emb_cfg = rag_cfg.get("embedding", {})
        if isinstance(emb_cfg, dict):
            default_model_id = emb_cfg.get("provider")

    effective = (default_model_id or "").strip()
    if not effective:
        raise SystemError(
            code="missing_embedding_model_id",
            message="Missing embedding model id, set rag.embedding.provider",
            details=None,
            retryable=False,
        )

    return effective


@router.post(
    "/search",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": schemas.SearchResponse},
        400: {"model": schemas.ErrorResponse},
        404: {"model": schemas.ErrorResponse},
        429: {"model": schemas.ErrorResponse},
        500: {"model": schemas.ErrorResponse},
    },
    summary="Vector search across ingested documents",
)
async def search(request: Request, body: schemas.SearchRequest) -> schemas.SearchResponse:
    request_id = get_request_id()
    embedding_model_id = _resolve_effective_embedding_model_id(request=request)

    use_case: SearchUseCase | None = getattr(request.app.state, "search_use_case", None)
    if use_case is None:
        raise SystemError(
            code="service_not_configured",
            message="Search use case not configured",
            details=None,
            retryable=False,
        )

    return await use_case.execute(
        request_id=request_id,
        embedding_model_id=embedding_model_id,
        req=body,
    )
