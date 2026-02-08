from __future__ import annotations

import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from scheduler_pubmed.src.adapters.pubmed.query_client import PubMedQueryClient
from scheduler_pubmed.src.adapters.rag.documents_client import RagDocumentsClient
from scheduler_pubmed.src.api.endpoints.pubmed_queries import router as pubmed_queries_router
from scheduler_pubmed.src.api.endpoints.scheduler import router as scheduler_router
from scheduler_pubmed.src.api.error_handlers import install_error_handlers
from scheduler_pubmed.src.api.endpoints.health import router as health_router
from scheduler_pubmed.src.common.logging import configure_logging
from scheduler_pubmed.src.common.settings import load_settings
from scheduler_pubmed.src.core.services.scheduler_runtime import SchedulerRuntimeService
from scheduler_pubmed.src.core.use_cases.scheduler import SchedulerOrchestrationUseCase
from scheduler_pubmed.src.db.engine import create_engine_and_sessionmaker
from scheduler_pubmed.src.db.migrate import run_migrations
from scheduler_pubmed.src.db.repositories.pubmed_query_repository import (
    SqlAlchemyPubMedQueryRepository,
)
from scheduler_pubmed.src.db.repositories.scheduler_repository import (
    SqlAlchemySchedulerRepository,
)

MIGRATIONS_ON_STARTUP_ENV = "PUBMED_SCHEDULER_RUN_MIGRATIONS_ON_STARTUP"
AUTO_RUNS_ON_STARTUP_ENV = "PUBMED_SCHEDULER_ENABLE_AUTOMATIC_RUNS"


def _migrations_enabled_on_startup() -> bool:
    raw = os.getenv(MIGRATIONS_ON_STARTUP_ENV, "true").strip().lower()
    return raw not in {"0", "false", "no"}


def _automatic_runs_enabled_on_startup() -> bool:
    raw = os.getenv(AUTO_RUNS_ON_STARTUP_ENV, "true").strip().lower()
    return raw not in {"0", "false", "no"}


def create_app() -> FastAPI:
    settings = load_settings()
    configure_logging(settings.logging_path)
    api_cfg = settings.require_api()
    scheduler_cfg = settings.require_scheduler()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pg_cfg = settings.require_postgres()
        db_engine, db_sessionmaker = create_engine_and_sessionmaker(pg_cfg=pg_cfg)
        app.state.db_engine = db_engine
        app.state.db_sessionmaker = db_sessionmaker

        pubmed_http_client = httpx.AsyncClient(timeout=httpx.Timeout(20.0))
        documents_http_client = httpx.AsyncClient(timeout=httpx.Timeout(20.0))
        app.state.pubmed_http_client = pubmed_http_client
        app.state.documents_http_client = documents_http_client

        scheduler_api_cfg = scheduler_cfg.get("api", {})
        documents_client = RagDocumentsClient(
            client=documents_http_client,
            base_url=str(scheduler_api_cfg.get("base_url", "http://localhost:8000")),
            documents_get_path=str(scheduler_api_cfg.get("documents_get", "/v1/documents/")),
            documents_post_batch_path=str(
                scheduler_api_cfg.get("documents_post_batch", "/v1/documents/fetch/batch")
            ),
            ingest_jobs_get_path=str(scheduler_api_cfg.get("ingest_jobs_get", "/v1/ingest/jobs/")),
        )
        pubmed_client = PubMedQueryClient(client=pubmed_http_client)

        scheduler_use_case = SchedulerOrchestrationUseCase(
            query_repository=SqlAlchemyPubMedQueryRepository(session_maker=db_sessionmaker),
            scheduler_repository=SqlAlchemySchedulerRepository(session_maker=db_sessionmaker),
            pubmed_client=pubmed_client,
            documents_client=documents_client,
        )
        scheduler_runtime = SchedulerRuntimeService(
            use_case=scheduler_use_case,
            enabled=bool(scheduler_cfg.get("enabled", False)),
            utc_schedule=[
                str(item) for item in scheduler_cfg.get("schedule", {}).get("utc_times", [])
            ],
            automatic_schedule_enabled=_automatic_runs_enabled_on_startup(),
        )
        app.state.scheduler_runtime = scheduler_runtime

        if _migrations_enabled_on_startup():
            await run_migrations(pg_cfg=pg_cfg)

        await scheduler_runtime.start()

        try:
            yield
        finally:
            await scheduler_runtime.stop()
            await pubmed_http_client.aclose()
            await documents_http_client.aclose()
            await db_engine.dispose()

    app = FastAPI(
        title=str(api_cfg.get("title", "PubMed Scheduler API")),
        description=str(
            api_cfg.get(
                "description",
                "PubMed Scheduler API service.",
            )
        ),
        version=str(api_cfg.get("version", "1.0.0")),
        lifespan=lifespan,
    )
    install_error_handlers(app)
    app.include_router(pubmed_queries_router)
    app.include_router(scheduler_router)
    app.include_router(health_router)

    return app


app = create_app()
