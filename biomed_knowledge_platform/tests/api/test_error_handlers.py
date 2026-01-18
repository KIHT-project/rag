from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from biomed_platform.api.error_handlers import _http_status_for_error, install_error_handlers
from biomed_platform.common.middleware.trace import request_id_ctx
from biomed_platform.core.errors.errors import AppError


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
