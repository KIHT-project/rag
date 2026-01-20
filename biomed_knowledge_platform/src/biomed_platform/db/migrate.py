from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg

from biomed_platform.common.logging import get_logger
from biomed_platform.common.settings import project_root
from biomed_platform.db.engine import build_async_database_url

log = get_logger(__name__)


@dataclass(frozen=True)
class MigrationConfig:
    max_attempts: int = 30
    delay_seconds: float = 1.0


def _asyncpg_dsn(async_sqlalchemy_url: str) -> str:
    return async_sqlalchemy_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def wait_for_postgres(
    *,
    pg_cfg: dict[str, Any],
    migration_cfg: MigrationConfig,
) -> None:
    url = build_async_database_url(pg_cfg)
    dsn = _asyncpg_dsn(url)

    last_err: Exception | None = None

    for attempt in range(1, migration_cfg.max_attempts + 1):
        try:
            conn = await asyncpg.connect(dsn)
            await conn.close()
            log.info("Postgres is ready")
            return
        except Exception as e:
            last_err = e
            log.info(
                "Waiting for Postgres, attempt=%s/%s, error=%s",
                attempt,
                migration_cfg.max_attempts,
                type(e).__name__,
            )
            await asyncio.sleep(migration_cfg.delay_seconds)

    raise RuntimeError("Postgres not reachable") from last_err


def _run_alembic_upgrade(*, cwd: str) -> None:
    env = os.environ.copy()

    # Ensure local source is importable when alembic/env.py imports biomed_platform
    src_path = str(Path(cwd) / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing}" if existing else src_path

    cmd = [
        sys.executable,
        "-m",
        "alembic",
        "-c",
        "alembic.ini",
        "upgrade",
        "head",
    ]

    subprocess.run(cmd, cwd=cwd, env=env, check=True)


async def run_migrations(*, pg_cfg: dict[str, Any]) -> None:
    cfg = MigrationConfig()
    await wait_for_postgres(pg_cfg=pg_cfg, migration_cfg=cfg)

    cwd = str(project_root())
    await asyncio.to_thread(_run_alembic_upgrade, cwd=cwd)
    log.info("Database migrations applied")
