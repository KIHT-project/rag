from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from scheduler_pubmed.src.api.endpoints.health import router as health_router
from scheduler_pubmed.src.common.logging import configure_logging
from scheduler_pubmed.src.common.settings import load_settings
from scheduler_pubmed.src.db.engine import create_engine_and_sessionmaker
from scheduler_pubmed.src.db.migrate import run_migrations

MIGRATIONS_ON_STARTUP_ENV = "PUBMED_SCHEDULER_RUN_MIGRATIONS_ON_STARTUP"


def _migrations_enabled_on_startup() -> bool:
    raw = os.getenv(MIGRATIONS_ON_STARTUP_ENV, "true").strip().lower()
    return raw not in {"0", "false", "no"}


def create_app() -> FastAPI:
    settings = load_settings()
    configure_logging(settings.logging_path)
    api_cfg = settings.require_api()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pg_cfg = settings.require_postgres()
        db_engine, db_sessionmaker = create_engine_and_sessionmaker(pg_cfg=pg_cfg)
        app.state.db_engine = db_engine
        app.state.db_sessionmaker = db_sessionmaker

        if _migrations_enabled_on_startup():
            await run_migrations(pg_cfg=pg_cfg)

        try:
            yield
        finally:
            await db_engine.dispose()

    app = FastAPI(
        title=str(api_cfg.get("title", "PubMed Scheduler API")),
        description=str(
            api_cfg.get(
                "description",
                "Dummy application exposing only healthcheck endpoints.",
            )
        ),
        version=str(api_cfg.get("version", "1.0.0")),
        lifespan=lifespan,
    )
    app.include_router(health_router)

    return app


app = create_app()
