from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import pytest
from fastapi import FastAPI
from fastapi import Response as FastAPIResponse
from starlette.requests import Request

from biomed_platform.core.errors.errors import AppError

import biomed_platform.api.endpoints.ingestion as ingestion_mod
from biomed_platform.common.middleware.trace import request_id_ctx
from conftest import clear_request_id_ctx

pytestmark = pytest.mark.asyncio


def _make_app(*, settings: Any | None = None, ingestion_service: Any | None = None) -> FastAPI:
    app = FastAPI()
    if settings is not None:
        app.state.settings = settings
    if ingestion_service is not None:
        app.state.ingestion_service = ingestion_service
    return app


async def _noop_receive() -> dict[str, Any]:
    return {"type": "http.request", "body": b"", "more_body": False}


def _make_request(app: FastAPI, *, method: str = "POST", path: str = "/v1/ingest/items") -> Request:
    scope: dict[str, Any] = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "headers": [],
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "scheme": "http",
        "root_path": "",
        "app": app,
    }
    return Request(scope, receive=_noop_receive)


class _Settings:
    def __init__(self, *, default_embedding_model_id: str | None) -> None:
        self._default_embedding_model_id = default_embedding_model_id

    def require_rag(self) -> dict[str, Any]:
        return {"default_embedding_model_id": self._default_embedding_model_id}


class _FakeIngestionService:
    def __init__(
        self,
        *,
        ingest_batch_impl: Callable[[Any], Awaitable[Any]] | None = None,
        get_job_status_impl: Callable[[str], Awaitable[Any]] | None = None,
    ) -> None:
        self._ingest_batch_impl = ingest_batch_impl
        self._get_job_status_impl = get_job_status_impl
        self.calls: list[tuple[str, Any]] = []

    async def ingest_batch(self, cmd: Any) -> Any:
        self.calls.append(("ingest_batch", cmd))
        if self._ingest_batch_impl is None:
            raise AssertionError("ingest_batch_impl not configured")
        return await self._ingest_batch_impl(cmd)

    async def get_job_status(self, *, job_id: str) -> Any:
        self.calls.append(("get_job_status", job_id))
        if self._get_job_status_impl is None:
            raise AssertionError("get_job_status_impl not configured")
        return await self._get_job_status_impl(job_id)


def _make_ingest_body(*, embedding_model_id: str | None, items_count: int = 1) -> Any:
    return SimpleNamespace(
        embedding_model_id=embedding_model_id,
        items=[SimpleNamespace() for _ in range(items_count)],
    )


class TestIngestionEndpoints:
    async def test_resolve_effective_embedding_model_id_uses_body_when_present(self) -> None:
        app = _make_app(settings=_Settings(default_embedding_model_id="default_one"))
        request = _make_request(app)
        body = _make_ingest_body(embedding_model_id="body_one", items_count=1)

        got = ingestion_mod._resolve_effective_embedding_model_id(request=request, body=body)

        assert got == "body_one"

    async def test_resolve_effective_embedding_model_id_uses_default_from_settings_and_strips(self) -> None:
        app = _make_app(settings=_Settings(default_embedding_model_id="  default_one  "))
        request = _make_request(app)
        body = _make_ingest_body(embedding_model_id=None, items_count=1)

        got = ingestion_mod._resolve_effective_embedding_model_id(request=request, body=body)

        assert got == "default_one"

    async def test_resolve_effective_embedding_model_id_returns_empty_when_no_settings_and_no_body(self) -> None:
        app = _make_app(settings=None)
        request = _make_request(app)
        body = _make_ingest_body(embedding_model_id=None, items_count=1)

        got = ingestion_mod._resolve_effective_embedding_model_id(request=request, body=body)

        assert got == ""

    @pytest.mark.parametrize(
        ("code", "expected_status"),
        [
            ("validation_error", 400),
            ("invalid_model_id", 400),
            ("duplicate_doi", 409),
            ("not_found", 404),
            ("too_many_requests", 429),
            ("something_else", 500),
        ],
    )
    async def test_status_for_error_code_maps_correctly(self, code: str, expected_status: int) -> None:
        assert ingestion_mod._status_for_error_code(code) == expected_status

    async def test_ingest_items_returns_202_when_service_accepts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        async def ingest_batch_impl(cmd: Any) -> Any:
            return SimpleNamespace(job_id="j1", state="queued")

        service = _FakeIngestionService(ingest_batch_impl=ingest_batch_impl)
        app = _make_app(settings=_Settings(default_embedding_model_id="def_model"), ingestion_service=service)

        request = _make_request(app)
        response = FastAPIResponse()

        def fake_to_cmd(*, request: Any, effective_embedding_model_id: str, idempotency_key: str | None) -> Any:
            captured["effective_embedding_model_id"] = effective_embedding_model_id
            captured["idempotency_key"] = idempotency_key
            captured["body"] = request
            return SimpleNamespace(cmd="ok", effective_embedding_model_id=effective_embedding_model_id)

        def fake_to_accepted(accepted: Any) -> Any:
            return {"job_id": accepted.job_id, "state": "queued"}

        monkeypatch.setattr(ingestion_mod, "to_ingest_batch_command", fake_to_cmd)
        monkeypatch.setattr(ingestion_mod, "to_ingest_job_accepted_response", fake_to_accepted)

        body = _make_ingest_body(embedding_model_id=None, items_count=3)

        token = request_id_ctx.set("req1")
        try:
            got = await ingestion_mod.ingest_items(
                request=request,
                response=response,
                body=body,
                idempotency_key="idem1",
            )
        finally:
            request_id_ctx.reset(token)

        assert response.status_code in (200, 202)
        assert got == {"job_id": "j1", "state": "queued"}
        assert captured["effective_embedding_model_id"] == "def_model"
        assert captured["idempotency_key"] == "idem1"

    async def test_ingest_items_works_when_context_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def ingest_batch_impl(cmd: Any) -> Any:
            return SimpleNamespace(job_id="j1", state="queued")

        service = _FakeIngestionService(ingest_batch_impl=ingest_batch_impl)
        app = _make_app(settings=None, ingestion_service=service)

        request = _make_request(app)
        response = FastAPIResponse()

        monkeypatch.setattr(
            ingestion_mod,
            "to_ingest_batch_command",
            lambda *, request, effective_embedding_model_id, idempotency_key: SimpleNamespace(cmd="ok"),
        )
        monkeypatch.setattr(
            ingestion_mod,
            "to_ingest_job_accepted_response",
            lambda accepted: {"job_id": accepted.job_id, "state": "queued"},
        )

        body = _make_ingest_body(embedding_model_id=None, items_count=1)

        got = await ingestion_mod.ingest_items(
            request=request,
            response=response,
            body=body,
            idempotency_key=None,
        )

        assert response.status_code in (200, 202)
        assert got == {"job_id": "j1", "state": "queued"}

    async def test_ingest_items_returns_400_when_service_missing(self) -> None:
        app = _make_app(settings=None, ingestion_service=None)
        request = _make_request(app)
        response = FastAPIResponse()

        body = _make_ingest_body(embedding_model_id=None, items_count=1)

        token = request_id_ctx.set("req1")
        try:
            got = await ingestion_mod.ingest_items(
                request=request,
                response=response,
                body=body,
                idempotency_key=None,
            )
        finally:
            request_id_ctx.reset(token)

        assert response.status_code == 400
        assert got.request_id == "req1"
        assert got.error.value == "validation_error"

    async def test_ingest_items_sets_retry_after_header_on_too_many_requests(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def ingest_batch_impl(cmd: Any) -> Any:
            raise AppError(
                code="too_many_requests",
                message="busy",
                details={"retry_after_seconds": 5},
                retryable=True,
            )

        service = _FakeIngestionService(ingest_batch_impl=ingest_batch_impl)
        app = _make_app(settings=None, ingestion_service=service)

        request = _make_request(app)
        response = FastAPIResponse()

        monkeypatch.setattr(
            ingestion_mod,
            "to_ingest_batch_command",
            lambda *, request, effective_embedding_model_id, idempotency_key: SimpleNamespace(cmd="ok"),
        )

        body = _make_ingest_body(embedding_model_id=None, items_count=1)

        token = request_id_ctx.set("req1")
        try:
            got = await ingestion_mod.ingest_items(
                request=request,
                response=response,
                body=body,
                idempotency_key=None,
            )
        finally:
            request_id_ctx.reset(token)

        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "5"
        assert got.request_id == "req1"
        assert got.error.value == "too_many_requests"

    async def test_get_job_status_returns_200_when_service_returns_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def get_job_status_impl(job_id: str) -> Any:
            return SimpleNamespace(job_id=job_id, state="succeeded")

        service = _FakeIngestionService(get_job_status_impl=get_job_status_impl)
        app = _make_app(settings=None, ingestion_service=service)

        request = _make_request(app, method="GET", path="/v1/ingest/jobs/j1")
        response = FastAPIResponse()

        monkeypatch.setattr(
            ingestion_mod,
            "to_ingest_job_status_response",
            lambda job: {"job_id": job.job_id, "state": job.state},
        )

        token = request_id_ctx.set("req1")
        try:
            got = await ingestion_mod.get_job_status(
                request=request,
                response=response,
                job_id="j1",
            )
        finally:
            request_id_ctx.reset(token)

        assert response.status_code == 200
        assert got == {"job_id": "j1", "state": "succeeded"}

    async def test_get_job_status_returns_400_when_service_missing(self) -> None:
        app = _make_app(settings=None, ingestion_service=None)
        request = _make_request(app, method="GET", path="/v1/ingest/jobs/j1")
        response = FastAPIResponse()

        token = request_id_ctx.set("req1")
        try:
            got = await ingestion_mod.get_job_status(
                request=request,
                response=response,
                job_id="j1",
            )
        finally:
            request_id_ctx.reset(token)

        assert response.status_code == 400
        assert got.request_id == "req1"
        assert got.error.value == "validation_error"

    async def test_get_job_status_returns_404_when_service_raises_not_found(self) -> None:
        async def get_job_status_impl(job_id: str) -> Any:
            raise AppError(code="not_found", message="missing", details={"job_id": job_id}, retryable=False)

        service = _FakeIngestionService(get_job_status_impl=get_job_status_impl)
        app = _make_app(settings=None, ingestion_service=service)

        request = _make_request(app, method="GET", path="/v1/ingest/jobs/j404")
        response = FastAPIResponse()

        got = await ingestion_mod.get_job_status(
            request=request,
            response=response,
            job_id="j404",
        )

        assert response.status_code == 404
        assert got.request_id == "none"
        assert got.error.value == "not_found"
        assert got.details == {"job_id": "j404"}
