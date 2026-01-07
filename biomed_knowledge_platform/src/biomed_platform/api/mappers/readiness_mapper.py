from __future__ import annotations

from biomed_platform.api.models.generated.schemas import (
    CheckStatus as ApiCheckStatus,
    ReadinessChecks as ApiReadinessChecks,
    ReadinessResponse as ApiReadinessResponse,
    ReadinessStatus as ApiReadinessStatus,
)
from biomed_platform.core.domains.readiness import (
    CheckStatus as DomainCheckStatus,
    ReadinessResult as DomainReadinessResult,
)


def to_api_check_status(status: DomainCheckStatus) -> ApiCheckStatus:
    return ApiCheckStatus(status.value)


def to_api_readiness_response(result: DomainReadinessResult) -> ApiReadinessResponse:
    return ApiReadinessResponse(
        status=ApiReadinessStatus(result.status.value),
        checks=ApiReadinessChecks(
            qdrant=to_api_check_status(result.checks.qdrant),
            llm=to_api_check_status(result.checks.llm),
        ),
        errors=result.errors,
    )
