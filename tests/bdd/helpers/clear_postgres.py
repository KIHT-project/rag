from __future__ import annotations

import os
import sys

import asyncpg


async def _truncate_public_schema(*, dsn: str) -> None:
    try:
        conn = await asyncpg.connect(dsn, ssl=False)
    except Exception as err:
        print(f"[clear_postgres] skipping cleanup: {err}", file=sys.stderr)
        return

    try:
        rows = await conn.fetch(
            """
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename <> 'alembic_version'
            """
        )
        tables = [r["tablename"] for r in rows]
        if not tables:
            return

        joined = ", ".join(f'"{t}"' for t in tables)
        await conn.execute(f"TRUNCATE TABLE {joined} RESTART IDENTITY CASCADE;")
    finally:
        await conn.close()



def main() -> int:
    dsn = os.getenv("BDD_POSTGRES_DSN")
    if not dsn:
        print("BDD_POSTGRES_DSN is not set", file=sys.stderr)
        return 2

    import asyncio

    asyncio.run(_truncate_public_schema(dsn=dsn))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
