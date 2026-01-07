from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette import status

from biomed_platform.common.middleware.trace import request_id_ctx
from biomed_platform.common.logging import get_logger
from biomed_platform.core.errors.errors import BusinessError, SystemError, AppError

log = get_logger(__name__)


def _rid() -> str:
    return request_id_ctx.get() or "none"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(BusinessError)
    async def business_error_handler(_: Request, exc: BusinessError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "type": "business",
                "error": exc.code,
                "message": exc.message,
                "details": exc.details or {},
                "retryable": exc.retryable,
                "request_id": _rid(),
            },
        )

    @app.exception_handler(SystemError)
    async def system_error_handler(_: Request, exc: SystemError) -> JSONResponse:
        log.warning("System error, code=%s, request_id=%s", exc.code, _rid())
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "type": "system",
                "error": exc.code,
                "message": exc.message,
                "details": exc.details or {},
                "retryable": exc.retryable,
                "request_id": _rid(),
            },
        )

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        log.warning("App error, code=%s, request_id=%s", exc.code, _rid())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "type": "system",
                "error": exc.code,
                "message": exc.message,
                "details": exc.details or {},
                "retryable": exc.retryable,
                "request_id": _rid(),
            },
        )
