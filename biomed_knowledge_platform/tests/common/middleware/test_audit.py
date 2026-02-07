from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from biomed_platform.common.middleware.audit import AuditMiddleware
from biomed_platform.common.middleware.request_context import RequestContextMiddleware


@dataclass
class _FakeAuditService:
    requests_created: list[dict[str, Any]] = field(default_factory=list)
    requests_completed: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    marked_errors: list[dict[str, Any]] = field(default_factory=list)

    async def create_request(self, **kwargs):
        self.requests_created.append(kwargs)

    async def complete_request(self, **kwargs):
        self.requests_completed.append(kwargs)

    async def create_event(self, **kwargs):
        self.events.append(kwargs)
        return f"event-{len(self.events)}"

    async def create_error(self, **kwargs):
        self.errors.append(kwargs)
        return f"error-{len(self.errors)}"

    async def mark_request_error(self, **kwargs):
        self.marked_errors.append(kwargs)


def _build_app(service: _FakeAuditService) -> FastAPI:
    app = FastAPI()
    app.state.audit_service = service
    app.add_middleware(AuditMiddleware)
    app.add_middleware(RequestContextMiddleware)
    return app


def test_audit_middleware_logs_successful_request() -> None:
    service = _FakeAuditService()
    app = _build_app(service)

    @app.get("/ok")
    async def ok():
        return {"status": "ok"}

    with TestClient(app) as client:
        res = client.get("/ok", headers={"X-Request-Id": "rid-ok"})
        assert res.status_code == 200

    assert len(service.requests_created) == 1
    assert service.requests_created[0]["request_id"] == "rid-ok"
    assert len(service.requests_completed) == 1
    assert service.requests_completed[0]["request_id"] == "rid-ok"
    assert service.requests_completed[0]["response_status_code"] == 200
    event_types = [e["event_type"] for e in service.events]
    assert "API_REQUEST_RECEIVED" in event_types
    assert "API_RESPONSE_SENT" in event_types


def test_audit_middleware_logs_unhandled_exception_path() -> None:
    service = _FakeAuditService()
    app = _build_app(service)

    @app.get("/boom")
    async def boom():
        raise RuntimeError("boom")

    with TestClient(app, raise_server_exceptions=False) as client:
        res = client.get("/boom", headers={"X-Request-Id": "rid-err"})
        assert res.status_code == 500

    assert len(service.requests_created) == 1
    assert len(service.errors) == 1
    assert service.errors[0]["request_id"] == "rid-err"
    assert isinstance(service.errors[0]["exc"], RuntimeError)
    assert len(service.marked_errors) == 1


def test_audit_middleware_allows_duplicate_request_ids() -> None:
    service = _FakeAuditService()
    app = _build_app(service)

    @app.get("/dupe")
    async def dupe():
        return {"ts": datetime.now(timezone.utc).isoformat()}

    with TestClient(app) as client:
        r1 = client.get("/dupe", headers={"X-Request-Id": "rid-dup"})
        r2 = client.get("/dupe", headers={"X-Request-Id": "rid-dup"})
        assert r1.status_code == 200
        assert r2.status_code == 200

    created_ids = [r["request_id"] for r in service.requests_created]
    assert created_ids == ["rid-dup", "rid-dup"]
    assert len(service.requests_completed) == 2
