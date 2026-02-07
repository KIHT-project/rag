from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from biomed_platform.common.logging import get_logger
from biomed_platform.common.middleware.trace import request_id_ctx

log = get_logger(__name__)

if TYPE_CHECKING:
    from biomed_platform.audit.service import PostgresAuditService


def _decode_body(body: bytes) -> str | None:
    if not body:
        return None
    return body.decode("utf-8", errors="replace")


async def _capture_response_body(response: Response) -> tuple[bytes, Response]:
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else bytes(chunk))
    body = b"".join(chunks)
    cloned = Response(
        content=body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
        background=response.background,
    )
    return body, cloned


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        service: PostgresAuditService | None = getattr(request.app.state, "audit_service", None)
        request_id = request_id_ctx.get() or request.headers.get("X-Request-Id") or "none"

        body_bytes = await request.body()

        async def _receive() -> dict[str, Any]:
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        request._receive = _receive  # type: ignore[attr-defined]

        if service is not None:
            payload_received = {
                "path": request.url.path,
                "query_string": request.url.query,
                "path_params": dict(getattr(request, "path_params", {}) or {}),
            }
            if body_bytes:
                payload_received["body_raw"] = _decode_body(body_bytes)
            await service.create_request(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                query_string_raw=request.url.query,
                headers=dict(request.headers),
                body_raw=_decode_body(body_bytes),
                client_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                api_key_id=request.headers.get("x-api-key-id"),
                session_id=request.headers.get("x-session-id"),
                user_id=request.headers.get("x-user-id"),
            )
            await service.create_event(
                request_id=request_id,
                run_id=None,
                event_type="API_REQUEST_RECEIVED",
                component="API",
                status="STARTED",
                message=f"{request.method} {request.url.path}",
                payload=payload_received,
            )
            await service.create_event(
                request_id=request_id,
                run_id=None,
                event_type="API_HANDLER_STARTED",
                component="API",
                status="STARTED",
                message="handler started",
                payload=None,
            )

        try:
            response = await call_next(request)
        except Exception as exc:
            if service is not None:
                stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                event_id = await service.create_event(
                    request_id=request_id,
                    run_id=None,
                    event_type="EXCEPTION_RAISED",
                    component="API",
                    status="ERROR",
                    message=str(exc),
                    payload={"exception": type(exc).__name__},
                    stacktrace_raw=stack,
                )
                error_id = await service.create_error(
                    request_id=request_id,
                    run_id=None,
                    event_id=event_id,
                    exc=exc,
                    component="API",
                    phase="REQUEST",
                    error_code=None,
                    is_fatal=True,
                )
                await service.mark_request_error(request_id=request_id, error_id=error_id)
            raise

        response_body, response = await _capture_response_body(response)
        completed_at = datetime.now(timezone.utc)

        if service is not None:
            await service.complete_request(
                request_id=request_id,
                response_status_code=response.status_code,
                response_headers=dict(response.headers),
                response_body_raw=_decode_body(response_body),
                completed_at=completed_at,
            )
            status_label = "ERROR" if int(response.status_code) >= 400 else "COMPLETED"
            await service.create_event(
                request_id=request_id,
                run_id=None,
                event_type="API_HANDLER_COMPLETED",
                component="API",
                status=status_label,
                message="handler completed",
                payload={"status_code": int(response.status_code)},
            )
            await service.create_event(
                request_id=request_id,
                run_id=None,
                event_type="API_RESPONSE_SENT",
                component="API",
                status=status_label,
                message=f"{request.method} {request.url.path}",
                payload={"status_code": int(response.status_code)},
            )
        return response
