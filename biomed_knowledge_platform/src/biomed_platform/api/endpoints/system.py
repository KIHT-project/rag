from __future__ import annotations

import httpx
from fastapi import APIRouter, Request, Response, status

from biomed_platform.api.mappers.readiness_mapper import to_api_readiness_response
from biomed_platform.api.models.generated.schemas import ReadinessResponse
from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.readiness import (
    ReadinessStatus as DomainReadinessStatus,
)
from biomed_platform.core.services.readiness import (
    compute_readiness,
    normalize_ollama_base_url,
)

router = APIRouter(tags=["System"])
log = get_logger(__name__)


@router.get("/health", summary="Health check")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness check", response_model=ReadinessResponse)
async def readiness_check(request: Request, response: Response) -> ReadinessResponse:
    settings = getattr(request.app.state, "settings", None)

    qdrant_url = ""
    ollama_url = ""

    if settings is not None:
        qdrant_cfg = settings.require_qdrant()
        llm_cfg = settings.require_llm()
        qdrant_url = str(qdrant_cfg.get("url", "")).rstrip("/")
        ollama_url = normalize_ollama_base_url(str(llm_cfg.get("ollama_base_url", "")).rstrip("/"))

    timeout = httpx.Timeout(connect=2.0, read=2.0, write=2.0, pool=2.0)

    domain_result = await compute_readiness(
        qdrant_url=qdrant_url,
        ollama_url=ollama_url,
        timeout=timeout,
    )

    is_ready = domain_result.status == DomainReadinessStatus.ready
    response.status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    (log.debug if is_ready else log.warning)(
        "Readiness result, status=%s, qdrant=%s, llm=%s, errors=%s",
        domain_result.status.value,
        domain_result.checks.qdrant.value,
        domain_result.checks.llm.value,
        domain_result.errors,
    )

    return to_api_readiness_response(domain_result)
