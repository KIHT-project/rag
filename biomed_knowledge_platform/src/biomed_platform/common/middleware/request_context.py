from __future__ import annotations

import uuid
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from biomed_platform.common.logging import get_logger
from biomed_platform.common.middleware.trace import request_id_ctx

log = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        token = request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)

        response.headers["X-Request-Id"] = request_id
        return response

class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        log.info(
            "HTTP request, method=%s, path=%s, status=%s, duration_ms=%.2f, request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id_ctx.get() or "none",
        )
        return response