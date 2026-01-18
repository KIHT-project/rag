from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette import status

from biomed_platform.api.models.generated import schemas
from biomed_platform.common.logging import get_logger
from biomed_platform.common.middleware.trace import request_id_ctx
from biomed_platform.core.errors.errors import AppError

log = get_logger(__name__)


def _rid() -> str:
    return request_id_ctx.get() or "none"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _http_status_for_error(code: str) -> int:
    if code == "validation_error":
        return status.HTTP_400_BAD_REQUEST
    if code == "invalid_model_id":
        return status.HTTP_400_BAD_REQUEST
    if code == "missing_embedding_model_id":
        return status.HTTP_400_BAD_REQUEST
    if code == "duplicate_doi":
        return status.HTTP_409_CONFLICT
    if code == "not_found":
        return status.HTTP_404_NOT_FOUND
    if code == "too_many_requests":
        return status.HTTP_429_TOO_MANY_REQUESTS
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _to_error_response(*, request_id: str, http_status: int, exc: AppError) -> dict:
    try:
        err = schemas.Error(exc.code)
    except Exception:
        err = schemas.Error.system_error

    model = schemas.ErrorResponse(
        request_id=request_id,
        error=err,
        message=exc.message,
        details=exc.details,
        status_code=int(http_status),
        timestamp=_utc_now_iso(),
    )
    return model.model_dump(mode="json")


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        request_id = _rid()
        http_status = _http_status_for_error(exc.code)

        if http_status >= 500:
            log.warning("App error, code=%s, request_id=%s", exc.code, request_id)
        else:
            log.info("App error, code=%s, request_id=%s", exc.code, request_id)

        headers: dict[str, str] = {}
        if exc.code == "too_many_requests":
            retry_after = None
            if isinstance(exc.details, dict):
                retry_after = exc.details.get("retry_after_seconds")
            if isinstance(retry_after, int) and retry_after > 0:
                headers["Retry-After"] = str(retry_after)

        return JSONResponse(
            status_code=http_status,
            content=_to_error_response(request_id=request_id, http_status=http_status, exc=exc),
            headers=headers,
        )
