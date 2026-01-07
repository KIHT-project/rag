from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI

import biomed_platform.api.app as app_mod


class TestApiAppBDD:
    def test_given_settings_when_create_app_then_configures_logging_and_sets_app_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        configured: dict[str, Any] = {}

        class _Settings:
            logging_path = "/tmp/logging.yaml"

            def require_api(self) -> dict[str, str]:
                return {
                    "title": "Biomedical Knowledge Platform",
                    "description": "RAG based biomedical knowledge system",
                    "version": "0.1.0",
                }

        def fake_load_settings() -> _Settings:
            return _Settings()

        def fake_configure_logging(path) -> None:
            configured["logging_path"] = path

        class _Logger:
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple[Any, ...]]] = []

            def info(self, msg: str, *args: Any) -> None:
                self.calls.append((msg, args))

        logger = _Logger()

        def fake_get_logger(_: str):
            return logger

        monkeypatch.setattr(app_mod, "load_settings", fake_load_settings)
        monkeypatch.setattr(app_mod, "configure_logging", fake_configure_logging)
        monkeypatch.setattr(app_mod, "get_logger", fake_get_logger)

        # When
        app = app_mod.create_app()

        # Then
        assert isinstance(app, FastAPI)

        assert configured["logging_path"] == "/tmp/logging.yaml"

        assert app.title == "Biomedical Knowledge Platform"
        assert app.description == "RAG based biomedical knowledge system"
        assert app.version == "0.1.0"

        assert getattr(app.state, "settings") is not None

        assert any("API %s | version=%s | Description: %s" in c[0] for c in logger.calls)
        assert any(
            c[1] == ("Biomedical Knowledge Platform", "0.1.0", "RAG based biomedical knowledge system")
            for c in logger.calls
        )

    def test_given_create_app_when_called_then_registers_expected_routers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        class _Settings:
            logging_path = "/tmp/logging.yaml"

            def require_api(self) -> dict[str, str]:
                return {"title": "t", "description": "d", "version": "v"}

        monkeypatch.setattr(app_mod, "load_settings", lambda: _Settings())
        monkeypatch.setattr(app_mod, "configure_logging", lambda _: None)
        monkeypatch.setattr(app_mod, "get_logger", lambda _: type("L", (), {"info": lambda *a, **k: None})())

        # When
        app = app_mod.create_app()

        # Then
        paths = {getattr(r, "path", None) for r in app.router.routes}
        # system router defines /health and /ready
        assert "/health" in paths
        assert "/ready" in paths

    def test_given_create_app_when_called_then_adds_middlewares_in_expected_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        class _Settings:
            logging_path = "/tmp/logging.yaml"

            def require_api(self) -> dict[str, str]:
                return {"title": "t", "description": "d", "version": "v"}

        monkeypatch.setattr(app_mod, "load_settings", lambda: _Settings())
        monkeypatch.setattr(app_mod, "configure_logging", lambda _: None)
        monkeypatch.setattr(app_mod, "get_logger", lambda _: type("L", (), {"info": lambda *a, **k: None})())

        # When
        app = app_mod.create_app()

        # Then
        middleware_classes = [m.cls for m in app.user_middleware]
        assert app_mod.AccessLogMiddleware in middleware_classes
        assert app_mod.RequestContextMiddleware in middleware_classes

        # Starlette executes middlewares in reverse registration order.
        # You add AccessLogMiddleware first, then RequestContextMiddleware,
        # so RequestContextMiddleware must be earlier in the chain.
        assert middleware_classes.index(app_mod.RequestContextMiddleware) < middleware_classes.index(
            app_mod.AccessLogMiddleware
        )

    def test_given_load_settings_raises_when_create_app_then_propagates_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        monkeypatch.setattr(app_mod, "load_settings", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        # When, Then
        with pytest.raises(RuntimeError, match="boom"):
            app_mod.create_app()
