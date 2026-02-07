from __future__ import annotations

from datetime import datetime, timezone
import traceback
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from biomed_platform.audit.service import PostgresAuditService


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


async def _audit_exception(
    *,
    request: Request,
    request_id: str,
    exc: Exception,
    error_code: str | None,
    phase: str,
) -> None:
    service: PostgresAuditService | None = getattr(request.app.state, "audit_service", None)
    if service is None:
        return

    stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    event_id = await service.create_event(
        request_id=request_id,
        run_id=None,
        event_type="EXCEPTION_RAISED",
        component="API",
        status="ERROR",
        message=str(exc),
        phase=phase,
        payload={"exception": type(exc).__name__},
        stacktrace_raw=stack,
    )
    error_id = await service.create_error(
        request_id=request_id,
        run_id=None,
        event_id=event_id,
        exc=exc,
        component="API",
        phase=phase,
        error_code=error_code,
        is_fatal=True,
    )
    await service.mark_request_error(request_id=request_id, error_id=error_id)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        request_id = _rid()
        http_status = _http_status_for_error(exc.code)
        await _audit_exception(
            request=request,
            request_id=request_id,
            exc=exc,
            error_code=exc.code,
            phase="APP_ERROR",
        )

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
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = _rid()
        err = _to_validation_error(details={"errors": exc.errors()})
        await _audit_exception(
            request=request,
            request_id=request_id,
            exc=exc,
            error_code=err.code,
            phase="VALIDATION",
        )
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
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
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

        await _audit_exception(
            request=request,
            request_id=request_id,
            exc=exc,
            error_code=err.code,
            phase="HTTP_EXCEPTION",
        )
        return JSONResponse(
            status_code=status_code,
            content=_to_error_response(request_id=request_id, http_status=status_code, exc=err),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = _rid()
        log.exception("Unhandled exception, request_id=%s, type=%s", request_id, type(exc).__name__)
        await _audit_exception(
            request=request,
            request_id=request_id,
            exc=exc,
            error_code="system_error",
            phase="UNHANDLED",
        )
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
