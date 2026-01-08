from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from biomed_platform.api.endpoints.system import router as system_router
from biomed_platform.api.endpoints.ingestion import router as ingestion_router
from biomed_platform.api.error_handlers import install_error_handlers
from biomed_platform.common.middleware.request_context import (
    RequestContextMiddleware,
    AccessLogMiddleware,
)
from biomed_platform.api.router import router as v1_router
from biomed_platform.common.logging import configure_logging, get_logger
from biomed_platform.common.settings import load_settings
from biomed_platform.core.services.ingestion import (
    DefaultIngestionService,
    InMemoryIdempotencyStore,
    InMemoryIngestionJobStore,
    InMemoryIngestionQueue,
    IngestionWorker,
    SimpleBackpressurePolicy,
    InMemoryDocumentRegistry,
)

WORKER_COUNT = 1
QUEUE_MAX_SIZE = 3
JOB_TTL_SECONDS = 86400
IDEMPOTENCY_TTL_SECONDS = 86400
MAX_JOBS = 10_000


def create_app() -> FastAPI:
    settings = load_settings()
    configure_logging(settings.logging_path)
    log = get_logger(__name__)

    api_cfg = settings.require_api()

    # Build runtime state inside create_app
    queue = InMemoryIngestionQueue(max_size=QUEUE_MAX_SIZE)
    job_store = InMemoryIngestionJobStore(
        ttl_seconds_after_completion=JOB_TTL_SECONDS,
        max_jobs=MAX_JOBS,
    )
    idempotency = InMemoryIdempotencyStore(ttl_seconds=IDEMPOTENCY_TTL_SECONDS)
    document_registry = InMemoryDocumentRegistry()
    backpressure = SimpleBackpressurePolicy(worker_count=WORKER_COUNT)

    service = DefaultIngestionService(
        queue=queue,
        job_store=job_store,
        idempotency_store=idempotency,
        document_registry=document_registry,
        backpressure=backpressure,
        worker_count=WORKER_COUNT,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # startup
        worker_tasks: list[asyncio.Task] = []

        for _ in range(WORKER_COUNT):
            worker = IngestionWorker(
                queue=queue,
                job_store=job_store,
                document_registry=document_registry,
            )
            task = asyncio.create_task(worker.run_forever())
            worker_tasks.append(task)

        log.info(
            "API %s | version=%s | Description: %s",
            api_cfg.get("title"),
            api_cfg.get("version"),
            api_cfg.get("description"),
        )

        yield

        # shutdown
        for task in worker_tasks:
            task.cancel()

        await asyncio.gather(*worker_tasks, return_exceptions=True)

    app = FastAPI(
        title=api_cfg.get("title"),
        description=api_cfg.get("description"),
        version=api_cfg.get("version"),
        lifespan=lifespan,
    )

    install_error_handlers(app)

    app.state.settings = settings
    app.state.ingestion_service = service

    app.include_router(system_router)
    app.include_router(v1_router)
    app.include_router(ingestion_router)

    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestContextMiddleware)

    return app


app = create_app()
