from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status
from starlette.exceptions import HTTPException as StarletteHTTPException

from scheduler_pubmed.src.api.models import schemas
from scheduler_pubmed.src.common.logging import get_logger
from scheduler_pubmed.src.core.errors.errors import AppError, SystemError

log = get_logger(__name__)


def _http_status_for_error(code: str) -> int:
    if code == "validation_error":
        return status.HTTP_400_BAD_REQUEST
    if code == "not_found":
        return status.HTTP_404_NOT_FOUND
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _to_error_response(*, exc: AppError) -> dict[str, object]:
    return schemas.ErrorResponse(
        error=exc.code,
        message=exc.message,
        details=exc.details,
    ).model_dump(mode="json")


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        http_status = _http_status_for_error(exc.code)
        if http_status >= 500:
            log.warning("App error, code=%s", exc.code)
        else:
            log.info("App error, code=%s", exc.code)
        return JSONResponse(status_code=http_status, content=_to_error_response(exc=exc))

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        _: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        err = AppError(
            code="validation_error",
            message="Request validation failed",
            details={"errors": exc.errors()},
            retryable=False,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=_to_error_response(exc=err),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        status_code = int(getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR))
        if status_code == status.HTTP_404_NOT_FOUND:
            err = AppError(code="not_found", message="Resource not found", details=None)
            return JSONResponse(status_code=status_code, content=_to_error_response(exc=err))

        if status_code in {
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_405_METHOD_NOT_ALLOWED,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        }:
            err = AppError(
                code="validation_error",
                message="Request validation failed",
                details={"detail": exc.detail},
            )
            return JSONResponse(status_code=status_code, content=_to_error_response(exc=err))

        err = SystemError(
            code="system_error",
            message="Internal server error",
            details={"detail": str(exc.detail)},
            retryable=False,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_to_error_response(exc=err),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled exception, type=%s", type(exc).__name__)
        err = SystemError(
            code="system_error",
            message="Internal server error",
            details={"exception": type(exc).__name__},
            retryable=False,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_to_error_response(exc=err),
        )
