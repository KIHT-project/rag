from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from biomed_platform.api.endpoints import system as system_mod
from biomed_platform.core.domains.readiness import (
    CheckStatus,
    ReadinessChecks,
    ReadinessResult,
    ReadinessStatus,
)


def _make_app(settings: Any | None = None) -> FastAPI:
    app = FastAPI()
    if settings is not None:
        app.state.settings = settings
    app.include_router(system_mod.router)
    return app


class _Settings:
    def require_qdrant(self) -> dict[str, str]:
        return {"url": "http://qdrant:6333"}

    def require_llm(self) -> dict[str, str]:
        return {"ollama_base_url": "http://ollama:11434"}


class TestSystemEndpoints:
    def test_health_check_returns_ok(self) -> None:
        # Given
        app = _make_app()
        client = TestClient(app)

        # When
        r = client.get("/health")

        # Then
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_readiness_returns_ready_when_domain_result_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        app = _make_app(settings=_Settings())

        async def fake_compute_readiness(*, qdrant_url: str, ollama_url: str, timeout) -> ReadinessResult:
            return ReadinessResult(
                status=ReadinessStatus.ready,
                checks=ReadinessChecks(qdrant=CheckStatus.ok, llm=CheckStatus.ok),
                errors=None,
            )

        monkeypatch.setattr(system_mod, "compute_readiness", fake_compute_readiness)
        client = TestClient(app)

        # When
        r = client.get("/ready")

        # Then
        assert r.status_code == 200
        assert r.json() == {
            "status": "ready",
            "checks": {"qdrant": "ok", "llm": "ok"},
            "errors": None,
        }

    def test_readiness_returns_503_when_domain_result_not_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        app = _make_app(settings=_Settings())

        async def fake_compute_readiness(*, qdrant_url: str, ollama_url: str, timeout) -> ReadinessResult:
            return ReadinessResult(
                status=ReadinessStatus.not_ready,
                checks=ReadinessChecks(qdrant=CheckStatus.unhealthy, llm=CheckStatus.ok),
                errors={"qdrant": {"reason": "http_5xx", "status_code": 500}},
            )

        monkeypatch.setattr(system_mod, "compute_readiness", fake_compute_readiness)
        client = TestClient(app)

        # When
        r = client.get("/ready")

        # Then
        assert r.status_code == 503
        assert r.json() == {
            "status": "not_ready",
            "checks": {"qdrant": "unhealthy", "llm": "ok"},
            "errors": {"qdrant": {"reason": "http_5xx", "status_code": 500}},
        }

    def test_readiness_returns_missing_config_when_settings_not_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        app = _make_app(settings=None)

        async def fake_compute_readiness(*, qdrant_url: str, ollama_url: str, timeout) -> ReadinessResult:
            assert qdrant_url == ""
            assert ollama_url == ""
            return ReadinessResult(
                status=ReadinessStatus.not_ready,
                checks=ReadinessChecks(
                    qdrant=CheckStatus.missing_config, llm=CheckStatus.missing_config
                ),
                errors={
                    "qdrant": {"reason": "missing_config"},
                    "ollama": {"reason": "missing_config"},
                },
            )

        monkeypatch.setattr(system_mod, "compute_readiness", fake_compute_readiness)
        client = TestClient(app)

        # When
        r = client.get("/ready")

        # Then
        assert r.status_code == 503
        assert r.json() == {
            "status": "not_ready",
            "checks": {"qdrant": "missing_config", "llm": "missing_config"},
            "errors": {
                "qdrant": {"reason": "missing_config"},
                "ollama": {"reason": "missing_config"},
            },
        }

    def test_readiness_normalizes_ollama_base_url_before_calling_service(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        class _SettingsWithVersion:
            def require_qdrant(self) -> dict[str, str]:
                return {"url": "http://qdrant:6333"}

            def require_llm(self) -> dict[str, str]:
                return {"ollama_base_url": "http://ollama:11434/version"}

        app = _make_app(settings=_SettingsWithVersion())
        captured: dict[str, str] = {}

        async def fake_compute_readiness(*, qdrant_url: str, ollama_url: str, timeout) -> ReadinessResult:
            captured["qdrant_url"] = qdrant_url
            captured["ollama_url"] = ollama_url
            return ReadinessResult(
                status=ReadinessStatus.ready,
                checks=ReadinessChecks(qdrant=CheckStatus.ok, llm=CheckStatus.ok),
                errors=None,
            )

        monkeypatch.setattr(system_mod, "compute_readiness", fake_compute_readiness)
        client = TestClient(app)

        # When
        r = client.get("/ready")

        # Then
        assert r.status_code == 200
        assert r.json() == {
            "status": "ready",
            "checks": {"qdrant": "ok", "llm": "ok"},
            "errors": None,
        }
        assert captured["qdrant_url"] == "http://qdrant:6333"
        assert captured["ollama_url"] == "http://ollama:11434"
