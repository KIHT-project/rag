from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.testclient import TestClient

from biomed_platform.api.error_handlers import _http_status_for_error, install_error_handlers
from biomed_platform.common.middleware.trace import request_id_ctx
from biomed_platform.core.errors.errors import AppError


class _FakeAuditService:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.errors: list[dict] = []
        self.marked: list[dict] = []

    async def create_event(self, **kwargs):
        self.events.append(kwargs)
        return "event-1"

    async def create_error(self, **kwargs):
        self.errors.append(kwargs)
        return "error-1"

    async def mark_request_error(self, **kwargs):
        self.marked.append(kwargs)


def test_http_status_for_error_mapping() -> None:
    # Given
    assert _http_status_for_error("validation_error") == 400
    assert _http_status_for_error("duplicate_doi") == 409
    assert _http_status_for_error("not_found") == 404
    assert _http_status_for_error("too_many_requests") == 429


def test_app_error_handler_sets_retry_after_and_request_id() -> None:
    # Given, an app with error handlers installed
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    async def boom():
        raise AppError(
            code="too_many_requests",
            message="slow down",
            details={"retry_after_seconds": 7},
            retryable=True,
        )

    token = request_id_ctx.set("rid")
    try:
        client = TestClient(app)

        # When
        r = client.get("/boom")

        # Then
        assert r.status_code == 429
        assert r.headers.get("Retry-After") == "7"
        body = r.json()
        assert body["request_id"] == "rid"
        assert body["error"] == "too_many_requests"
    finally:
        request_id_ctx.reset(token)


def test_request_validation_error_is_normalized_with_422() -> None:
    app = FastAPI()
    install_error_handlers(app)

    class _Payload(BaseModel):
        value: int

    @app.post("/validate")
    async def validate(payload: _Payload):
        return {"value": payload.value}

    client = TestClient(app)
    r = client.post("/validate", json={"value": "not-int"})

    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_error"
    assert body["request_id"] == "none"
    assert isinstance(body.get("details"), dict)
    assert isinstance(body["details"].get("errors"), list)


def test_not_found_http_exception_is_normalized() -> None:
    app = FastAPI()
    install_error_handlers(app)
    client = TestClient(app)

    r = client.get("/missing-route")
    assert r.status_code == 404
    body = r.json()
    assert body["error"] == "not_found"
    assert body["message"] == "Resource not found"


def test_unhandled_exception_returns_standard_500_error_envelope() -> None:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/explode")
    async def explode():
        raise ValueError("boom")

    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/explode")
    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "system_error"
    assert body["message"] == "Internal server error"
    assert body["status_code"] == 500


def test_error_handlers_emit_audit_rows_when_service_present() -> None:
    app = FastAPI()
    install_error_handlers(app)
    app.state.audit_service = _FakeAuditService()

    @app.get("/explode")
    async def explode():
        raise ValueError("boom")

    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/explode")

    assert r.status_code == 500
    svc = app.state.audit_service
    assert len(svc.events) == 1
    assert svc.events[0]["event_type"] == "EXCEPTION_RAISED"
    assert len(svc.errors) == 1
    assert isinstance(svc.errors[0]["exc"], ValueError)
    assert len(svc.marked) == 1
