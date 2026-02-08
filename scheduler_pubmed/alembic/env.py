from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from common.settings import load_settings  # noqa: E402
from db.base import Base  # noqa: E402
from db.engine import build_async_connect_args, build_async_database_url  # noqa: E402

from db import models  # noqa: F401,E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    settings = load_settings()
    pg_cfg = settings.require_postgres()
    schema = str(pg_cfg.get("postgres_schema", "")).strip() or None

    context.configure(
        url=build_async_database_url(pg_cfg),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_schemas=True,
        version_table_schema=schema,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    settings = load_settings()
    pg_cfg = settings.require_postgres()
    schema = str(pg_cfg.get("postgres_schema", "")).strip() or None

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_schemas=True,
        version_table_schema=schema,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    settings = load_settings()
    pg_cfg = settings.require_postgres()

    connectable: AsyncEngine = create_async_engine(
        build_async_database_url(pg_cfg),
        poolclass=pool.NullPool,
        connect_args=build_async_connect_args(pg_cfg),
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
