from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from biomed_platform.api.endpoints import system as system_mod


class _FakeAsyncClient:
    def __init__(self, get_impl: Callable[[str], Any], *, timeout: Any = None) -> None:
        self._get_impl = get_impl
        self._timeout = timeout

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str) -> httpx.Response:
        result = self._get_impl(url)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, httpx.Response):
            return result
        raise TypeError(f"Fake client get_impl must return httpx.Response or Exception, got {type(result)}")


def _make_app(settings: Any | None = None) -> FastAPI:
    app = FastAPI()
    if settings is not None:
        app.state.settings = settings
    app.include_router(system_mod.router)
    return app


class TestSystemEndpointsBDD:
    def test_health_check_returns_ok(self) -> None:
        # Given
        app = _make_app()
        client = TestClient(app)

        # When
        r = client.get("/health")

        # Then
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_readiness_returns_ready_when_all_dependencies_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given
        settings = SimpleNamespace(
            require_qdrant=lambda: {"url": "http://qdrant:6333"},
            require_llm=lambda: {"ollama_base_url": "http://ollama:11434"},
        )
        app = _make_app(settings=settings)

        def fake_get(url: str) -> httpx.Response:
            if url == "http://qdrant:6333/collections":
                return httpx.Response(status_code=200, json={"collections": []})
            if url == "http://ollama:11434/api/version":
                return httpx.Response(status_code=200, json={"version": "x"})
            return httpx.Response(status_code=404)

        monkeypatch.setattr(
            system_mod.httpx,
            "AsyncClient",
            lambda timeout: _FakeAsyncClient(fake_get, timeout=timeout),
        )

        client = TestClient(app)

        # When
        r = client.get("/ready")

        # Then
        assert r.status_code == 200
        assert r.json() == {"status": "ready", "checks": {"qdrant": "ok", "llm": "ok"}}

    def test_readiness_returns_503_when_any_dependency_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given
        settings = SimpleNamespace(
            require_qdrant=lambda: {"url": "http://qdrant:6333"},
            require_llm=lambda: {"ollama_base_url": "http://ollama:11434"},
        )
        app = _make_app(settings=settings)

        def fake_get(url: str) -> httpx.Response:
            if url == "http://qdrant:6333/collections":
                return httpx.Response(status_code=500)
            if url == "http://ollama:11434/api/version":
                return httpx.Response(status_code=200, json={"version": "x"})
            return httpx.Response(status_code=404)

        monkeypatch.setattr(
            system_mod.httpx,
            "AsyncClient",
            lambda timeout: _FakeAsyncClient(fake_get, timeout=timeout),
        )

        client = TestClient(app)

        # When
        r = client.get("/ready")

        # Then
        assert r.status_code == 503
        assert r.json() == {"status": "not_ready", "checks": {"qdrant": "http_500", "llm": "ok"}}

    def test_readiness_returns_missing_config_when_settings_not_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given
        app = _make_app(settings=None)

        def fake_get(url: str) -> httpx.Response:
            return httpx.Response(status_code=200)

        monkeypatch.setattr(
            system_mod.httpx,
            "AsyncClient",
            lambda timeout: _FakeAsyncClient(fake_get, timeout=timeout),
        )

        client = TestClient(app)

        # When
        r = client.get("/ready")

        # Then
        assert r.status_code == 503
        assert r.json() == {
            "status": "not_ready",
            "checks": {"qdrant": "missing_config", "llm": "missing_config"},
        }
