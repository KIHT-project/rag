from __future__ import annotations

import pytest

from scheduler_pubmed.src.db import engine as engine_mod


def test_build_async_database_url() -> None:
    url = engine_mod.build_async_database_url(
        {
            "postgres_user": "user",
            "postgres_password": "password",
            "postgres_db": "pubmed_scheduler",
            "host": "db",
            "postgres_port": 5434,
        }
    )

    assert url == "postgresql+asyncpg://user:password@db:5434/pubmed_scheduler"


def test_build_async_database_url_requires_credentials() -> None:
    with pytest.raises(ValueError):
        engine_mod.build_async_database_url(
            {
                "postgres_user": "",
                "postgres_password": "password",
                "postgres_db": "pubmed_scheduler",
            }
        )


def test_build_async_connect_args_empty_schema() -> None:
    assert engine_mod.build_async_connect_args({"postgres_schema": " "}) == {}


def test_build_async_connect_args_non_empty_schema() -> None:
    assert engine_mod.build_async_connect_args({"postgres_schema": "pubmed_scheduler"}) == {
        "server_settings": {"search_path": "pubmed_scheduler"}
    }


def test_create_engine_and_sessionmaker_calls_sqlalchemy_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeEngine:
        pass

    fake_engine = FakeEngine()

    def fake_create_async_engine(url: str, **kwargs: object) -> FakeEngine:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return fake_engine

    def fake_async_sessionmaker(engine: object, expire_on_commit: bool) -> str:
        captured["session_engine"] = engine
        captured["expire_on_commit"] = expire_on_commit
        return "session-maker"

    monkeypatch.setattr(engine_mod, "create_async_engine", fake_create_async_engine)
    monkeypatch.setattr(engine_mod, "async_sessionmaker", fake_async_sessionmaker)

    engine, session_maker = engine_mod.create_engine_and_sessionmaker(
        pg_cfg={
            "postgres_user": "user",
            "postgres_password": "password",
            "postgres_db": "pubmed_scheduler",
            "host": "db",
            "postgres_port": 5432,
            "postgres_schema": "pubmed_scheduler",
        }
    )

    assert engine is fake_engine
    assert session_maker == "session-maker"
    assert captured["url"] == "postgresql+asyncpg://user:password@db:5432/pubmed_scheduler"
    assert captured["session_engine"] is fake_engine
    assert captured["expire_on_commit"] is False
