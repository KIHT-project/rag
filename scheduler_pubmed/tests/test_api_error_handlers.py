from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from scheduler_pubmed.src.api.error_handlers import install_error_handlers
from scheduler_pubmed.src.core.errors.errors import AppError


def _build_app() -> FastAPI:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/raise-app")
    async def raise_app() -> dict[str, str]:
        raise AppError(code="validation_error", message="bad payload", details={"field": "x"})

    @app.get("/raise-http")
    async def raise_http() -> dict[str, str]:
        raise HTTPException(status_code=404, detail="not found")

    @app.post("/validate")
    async def validate(body: dict[str, int]) -> dict[str, int]:
        return body

    @app.get("/raise-unhandled")
    async def raise_unhandled() -> dict[str, str]:
        raise RuntimeError("boom")

    return app


def test_app_error_returns_json_error_response() -> None:
    with TestClient(_build_app()) as client:
        response = client.get("/raise-app")

    assert response.status_code == 400
    assert response.json()["error"] == "validation_error"
    assert response.json()["message"] == "bad payload"


def test_http_exception_not_found_is_normalized() -> None:
    with TestClient(_build_app()) as client:
        response = client.get("/raise-http")

    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_request_validation_error_is_normalized() -> None:
    with TestClient(_build_app()) as client:
        response = client.post("/validate", json={"x": "not-an-int"})

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    assert response.json()["message"] == "Request validation failed"


def test_unhandled_exception_is_normalized() -> None:
    with TestClient(_build_app(), raise_server_exceptions=False) as client:
        response = client.get("/raise-unhandled")

    assert response.status_code == 500
    assert response.json()["error"] == "system_error"
    assert response.json()["message"] == "Internal server error"
