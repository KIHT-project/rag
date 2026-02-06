from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette import status

from biomed_platform.api.models.generated import schemas
from biomed_platform.common.logging import get_logger
from biomed_platform.common.middleware.trace import request_id_ctx
from biomed_platform.core.errors.errors import AppError, SystemError

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


def _to_validation_error(*, details: dict[str, object] | None) -> AppError:
    return AppError(
        code="validation_error",
        message="Request validation failed",
        details=details,
        retryable=False,
    )


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

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = _rid()
        err = _to_validation_error(details={"errors": exc.errors()})
        log.info("Request validation error, request_id=%s", request_id)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=_to_error_response(
                request_id=request_id,
                http_status=status.HTTP_422_UNPROCESSABLE_CONTENT,
                exc=err,
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = _rid()
        status_code = int(getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR))
        if status_code == status.HTTP_404_NOT_FOUND:
            err = AppError(code="not_found", message="Resource not found", details=None)
        elif status_code in {
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_405_METHOD_NOT_ALLOWED,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        }:
            err = _to_validation_error(details={"detail": exc.detail})
        else:
            err = SystemError(
                code="system_error",
                message="Internal server error",
                details={"detail": str(exc.detail)},
                retryable=False,
            )
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        return JSONResponse(
            status_code=status_code,
            content=_to_error_response(request_id=request_id, http_status=status_code, exc=err),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        request_id = _rid()
        log.exception("Unhandled exception, request_id=%s, type=%s", request_id, type(exc).__name__)
        err = SystemError(
            code="system_error",
            message="Internal server error",
            details={"exception": type(exc).__name__},
            retryable=False,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_to_error_response(
                request_id=request_id,
                http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                exc=err,
            ),
        )
