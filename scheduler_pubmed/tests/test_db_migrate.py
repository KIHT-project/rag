from __future__ import annotations

from pathlib import Path

import pytest

from scheduler_pubmed.src.db import migrate as migrate_mod


def test_asyncpg_dsn_conversion() -> None:
    assert (
        migrate_mod._asyncpg_dsn("postgresql+asyncpg://u:p@h:5432/db")
        == "postgresql://u:p@h:5432/db"
    )


@pytest.mark.asyncio
async def test_wait_for_postgres_success_first_try(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"connect": 0, "closed": 0}

    class FakeConn:
        async def close(self) -> None:
            calls["closed"] += 1

    async def fake_connect(_: str) -> FakeConn:
        calls["connect"] += 1
        return FakeConn()

    monkeypatch.setattr(migrate_mod.asyncpg, "connect", fake_connect)

    await migrate_mod.wait_for_postgres(
        pg_cfg={
            "postgres_user": "u",
            "postgres_password": "p",
            "postgres_db": "d",
            "host": "h",
            "postgres_port": 5432,
        },
        migration_cfg=migrate_mod.MigrationConfig(max_attempts=2, delay_seconds=0),
    )

    assert calls == {"connect": 1, "closed": 1}


@pytest.mark.asyncio
async def test_wait_for_postgres_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"connect": 0, "sleep": 0}

    class FakeConn:
        async def close(self) -> None:
            return None

    async def fake_connect(_: str) -> FakeConn:
        calls["connect"] += 1
        if calls["connect"] == 1:
            raise RuntimeError("temporary")
        return FakeConn()

    async def fake_sleep(_: float) -> None:
        calls["sleep"] += 1

    monkeypatch.setattr(migrate_mod.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(migrate_mod.asyncio, "sleep", fake_sleep)

    await migrate_mod.wait_for_postgres(
        pg_cfg={
            "postgres_user": "u",
            "postgres_password": "p",
            "postgres_db": "d",
            "host": "h",
            "postgres_port": 5432,
        },
        migration_cfg=migrate_mod.MigrationConfig(max_attempts=3, delay_seconds=0),
    )

    assert calls["connect"] == 2
    assert calls["sleep"] == 1


@pytest.mark.asyncio
async def test_wait_for_postgres_raises_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    async def always_fail(_: str) -> None:
        raise RuntimeError("down")

    async def no_wait(_: float) -> None:
        return None

    monkeypatch.setattr(migrate_mod.asyncpg, "connect", always_fail)
    monkeypatch.setattr(migrate_mod.asyncio, "sleep", no_wait)

    with pytest.raises(RuntimeError, match="Postgres not reachable"):
        await migrate_mod.wait_for_postgres(
            pg_cfg={
                "postgres_user": "u",
                "postgres_password": "p",
                "postgres_db": "d",
                "host": "h",
                "postgres_port": 5432,
            },
            migration_cfg=migrate_mod.MigrationConfig(max_attempts=2, delay_seconds=0),
        )


def test_run_alembic_upgrade_invokes_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setenv("PYTHONPATH", "existing_path")

    def fake_run(cmd: list[str], cwd: str, env: dict[str, str], check: bool) -> None:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        captured["check"] = check

    monkeypatch.setattr(migrate_mod.subprocess, "run", fake_run)

    migrate_mod._run_alembic_upgrade(cwd="/tmp/scheduler_pubmed")

    assert captured["cmd"] == [
        migrate_mod.sys.executable,
        "-m",
        "alembic",
        "-c",
        "alembic.ini",
        "upgrade",
        "head",
    ]
    assert captured["cwd"] == "/tmp/scheduler_pubmed"
    assert captured["check"] is True
    assert str(captured["env"]["PYTHONPATH"]).startswith("/tmp/scheduler_pubmed/src")


@pytest.mark.asyncio
async def test_run_migrations_calls_wait_and_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    async def fake_wait_for_postgres(*, pg_cfg: dict[str, object], migration_cfg: object) -> None:
        calls["wait_pg_cfg"] = pg_cfg
        calls["wait_cfg"] = migration_cfg

    async def fake_to_thread(fn: object, **kwargs: object) -> None:
        calls["to_thread_fn"] = fn
        calls["to_thread_kwargs"] = kwargs

    monkeypatch.setattr(migrate_mod, "wait_for_postgres", fake_wait_for_postgres)
    monkeypatch.setattr(migrate_mod.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(migrate_mod, "project_root", lambda: Path("/tmp/project"))

    pg_cfg = {
        "postgres_user": "u",
        "postgres_password": "p",
        "postgres_db": "d",
        "host": "h",
        "postgres_port": 5432,
    }

    await migrate_mod.run_migrations(pg_cfg=pg_cfg)

    assert calls["wait_pg_cfg"] == pg_cfg
    assert calls["to_thread_fn"] is migrate_mod._run_alembic_upgrade
    assert calls["to_thread_kwargs"] == {"cwd": "/tmp/project"}
