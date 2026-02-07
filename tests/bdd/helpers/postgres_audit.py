from __future__ import annotations

import asyncio
import os
from typing import Any

import asyncpg


async def _fetch_rows(*, sql: str, args: tuple[Any, ...]) -> list[dict[str, Any]]:
    dsn = os.getenv("BDD_POSTGRES_DSN", "")
    if not dsn:
        raise RuntimeError("BDD_POSTGRES_DSN is not set")

    conn = await asyncpg.connect(dsn, ssl=False)
    try:
        rows = await conn.fetch(sql, *args)
        return [dict(r) for r in rows]
    finally:
        await conn.close()


def fetch_rows(*, sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return asyncio.run(_fetch_rows(sql=sql, args=args))


def fetch_one(*, sql: str, args: tuple[Any, ...] = ()) -> dict[str, Any]:
    rows = fetch_rows(sql=sql, args=args)
    if not rows:
        raise AssertionError(f"No rows for sql={sql!r} args={args!r}")
    return rows[0]
