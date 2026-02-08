from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def build_async_database_url(pg_cfg: dict[str, Any]) -> str:
    user = str(pg_cfg.get("postgres_user", "postgres")).strip()
    password = str(pg_cfg.get("postgres_password", "postgres")).strip()
    db = str(pg_cfg.get("postgres_db", "postgres")).strip()

    host = str(pg_cfg.get("host", "localhost")).strip()
    port = str(pg_cfg.get("postgres_port", "5432")).strip()

    if not user or not password or not db:
        raise ValueError("postgres.yaml must include postgres_user, postgres_password, postgres_db")

    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


def build_async_connect_args(pg_cfg: dict[str, Any]) -> dict[str, Any]:
    schema = str(pg_cfg.get("postgres_schema", "")).strip()
    if not schema:
        return {}
    return {"server_settings": {"search_path": schema}}


def create_engine_and_sessionmaker(
    *,
    pg_cfg: dict[str, Any],
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout_seconds: int = 30,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    url = build_async_database_url(pg_cfg)
    connect_args = build_async_connect_args(pg_cfg)

    engine = create_async_engine(
        url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout_seconds,
        pool_pre_ping=True,
        connect_args=connect_args,
    )

    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    return engine, session_maker
