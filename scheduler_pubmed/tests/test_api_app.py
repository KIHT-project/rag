from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scheduler_pubmed.src.api import app as app_mod


def test_migrations_enabled_flag_defaults_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(app_mod.MIGRATIONS_ON_STARTUP_ENV, raising=False)
    assert app_mod._migrations_enabled_on_startup() is True


@pytest.mark.parametrize("value", ["0", "false", "False", "no"])
def test_migrations_enabled_flag_false_values(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(app_mod.MIGRATIONS_ON_STARTUP_ENV, value)
    assert app_mod._migrations_enabled_on_startup() is False


def test_create_app_runs_migrations_and_disposes_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {"migrations": 0, "disposed": 0}

    class FakeSettings:
        logging_path = "unused"

        @staticmethod
        def require_api() -> dict[str, str]:
            return {"title": "t", "description": "d", "version": "v"}

        @staticmethod
        def require_postgres() -> dict[str, object]:
            return {"postgres_user": "u", "postgres_password": "p", "postgres_db": "d"}

    class FakeEngine:
        async def dispose(self) -> None:
            calls["disposed"] = int(calls["disposed"]) + 1

    def fake_create_engine_and_sessionmaker(*, pg_cfg: dict[str, object]) -> tuple[FakeEngine, object]:
        calls["pg_cfg"] = pg_cfg
        return FakeEngine(), object()

    async def fake_run_migrations(*, pg_cfg: dict[str, object]) -> None:
        calls["migrations"] = int(calls["migrations"]) + 1
        calls["migrations_pg_cfg"] = pg_cfg

    monkeypatch.setattr(app_mod, "load_settings", lambda: FakeSettings())
    monkeypatch.setattr(app_mod, "configure_logging", lambda _: None)
    monkeypatch.setattr(app_mod, "create_engine_and_sessionmaker", fake_create_engine_and_sessionmaker)
    monkeypatch.setattr(app_mod, "run_migrations", fake_run_migrations)
    monkeypatch.setenv(app_mod.MIGRATIONS_ON_STARTUP_ENV, "true")

    app = app_mod.create_app()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert calls["migrations"] == 1
    assert calls["disposed"] == 1


def test_create_app_skips_migrations_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"migrations": 0}

    class FakeSettings:
        logging_path = "unused"

        @staticmethod
        def require_api() -> dict[str, str]:
            return {"title": "t", "description": "d", "version": "v"}

        @staticmethod
        def require_postgres() -> dict[str, object]:
            return {"postgres_user": "u", "postgres_password": "p", "postgres_db": "d"}

    class FakeEngine:
        async def dispose(self) -> None:
            return None

    def fake_create_engine_and_sessionmaker(*, pg_cfg: dict[str, object]) -> tuple[FakeEngine, object]:
        return FakeEngine(), object()

    async def fake_run_migrations(*, pg_cfg: dict[str, object]) -> None:
        calls["migrations"] += 1

    monkeypatch.setattr(app_mod, "load_settings", lambda: FakeSettings())
    monkeypatch.setattr(app_mod, "configure_logging", lambda _: None)
    monkeypatch.setattr(app_mod, "create_engine_and_sessionmaker", fake_create_engine_and_sessionmaker)
    monkeypatch.setattr(app_mod, "run_migrations", fake_run_migrations)
    monkeypatch.setenv(app_mod.MIGRATIONS_ON_STARTUP_ENV, "false")

    app = app_mod.create_app()

    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200

    assert calls["migrations"] == 0


def test_health_routes_are_tagged_health() -> None:
    openapi_app = app_mod.create_app()
    schema = openapi_app.openapi()

    assert schema["paths"]["/health"]["get"]["tags"] == ["Health"]
    assert schema["paths"]["/health/live"]["get"]["tags"] == ["Health"]
    assert schema["paths"]["/health/ready"]["get"]["tags"] == ["Health"]
