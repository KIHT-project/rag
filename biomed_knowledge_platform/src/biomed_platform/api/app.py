# src/biomed_platform/api/app.py
from __future__ import annotations
from biomed_platform.api.endpoints.ask import router as ask_router
from biomed_platform.api.endpoints.documents import router as documents_router
from biomed_platform.api.endpoints.retrieval import router as retrieval_router

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI
from qdrant_client import QdrantClient

from biomed_platform.api.endpoints.ingestion import router as ingestion_router
from biomed_platform.api.endpoints.system import router as system_router
from biomed_platform.api.error_handlers import install_error_handlers
from biomed_platform.api.router import router as v1_router
from biomed_platform.common.logging import configure_logging, get_logger
from biomed_platform.common.middleware.request_context import (
    AccessLogMiddleware,
    RequestContextMiddleware,
)
from biomed_platform.common.middleware.audit import AuditMiddleware
from biomed_platform.common.settings import load_settings
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.services.ingestion.backpressure import SimpleBackpressurePolicy
from biomed_platform.core.services.ingestion.in_memory_idempotency import InMemoryIdempotencyStore
from biomed_platform.core.services.ingestion.in_memory_job_store import InMemoryIngestionJobStore
from biomed_platform.core.services.ingestion.in_memory_queue import InMemoryIngestionQueue
from biomed_platform.core.services.ingestion.vector_index_document_registry import (
    VectorIndexDocumentRegistry,
)
from biomed_platform.core.use_cases.ingestion import IngestionUseCase
from biomed_platform.core.services.ingestion.chunking import SectionAwareChunker
from biomed_platform.core.services.ingestion.in_memory_payload_store import (
    InMemoryIngestPayloadStore,
)

from biomed_platform.core.services.ingestion.pipeline import DefaultIngestionPipeline
from biomed_platform.adapters.qdrant.vector_index import QdrantVectorIndex, parse_distance
from biomed_platform.adapters.ollama.ollama_client import OllamaLlmClient
from biomed_platform.adapters.pubmed import PubMedClientAdapter
from biomed_platform.core.use_cases.search import SearchUseCase
from biomed_platform.core.services.ingestion.sentence_transformers_embedder import (
    SentenceTransformersEmbeddingProvider,
)
from biomed_platform.core.services.ingestion.worker import IngestionWorker
from biomed_platform.db.engine import create_engine_and_sessionmaker
from biomed_platform.db.migrate import run_migrations

WORKER_COUNT = 1
QUEUE_MAX_SIZE = 3

JOB_TTL_SECONDS = 86400
IDEMPOTENCY_TTL_SECONDS = 86400
PAYLOAD_TTL_SECONDS = 86400

MAX_JOBS = 10_000

log = get_logger(__name__)


def _require_dict(value: Any, *, code: str, message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemError(
            code=code, message=message, details={"type": str(type(value))}, retryable=False
        )
    return value


def create_app() -> FastAPI:
    settings = load_settings()
    configure_logging(settings.logging_path)
    api_cfg = settings.require_api()

    queue = InMemoryIngestionQueue(max_size=QUEUE_MAX_SIZE)
    job_store = InMemoryIngestionJobStore(
        ttl_seconds_after_completion=JOB_TTL_SECONDS, max_jobs=MAX_JOBS
    )
    idempotency = InMemoryIdempotencyStore(ttl_seconds=IDEMPOTENCY_TTL_SECONDS)
    payload_store = InMemoryIngestPayloadStore(ttl_seconds=PAYLOAD_TTL_SECONDS, max_jobs=MAX_JOBS)
    backpressure = SimpleBackpressurePolicy(worker_count=WORKER_COUNT)

    rag_cfg = settings.require_rag()
    qdrant_cfg = settings.require_qdrant()

    emb_cfg = _require_dict(
        rag_cfg.get("embedding", {}),
        code="invalid_rag_config",
        message="rag.embedding must be a mapping",
    )
    chunk_cfg = _require_dict(
        rag_cfg.get("chunking", {}),
        code="invalid_rag_config",
        message="rag.chunking must be a mapping",
    )
    collection_cfg = _require_dict(
        qdrant_cfg.get("collection", {}),
        code="invalid_qdrant_config",
        message="qdrant.collection must be a mapping",
    )

    qdrant_url = str(qdrant_cfg.get("url", "")).strip()
    if not qdrant_url:
        raise SystemError(
            code="missing_qdrant_url",
            message="Missing qdrant.url in qdrant.yaml",
            details=None,
            retryable=False,
        )

    qdrant_api_key = qdrant_cfg.get("api_key")
    timeout_seconds = float(qdrant_cfg.get("timeout_seconds", 10.0))

    client = QdrantClient(
        url=qdrant_url,
        api_key=str(qdrant_api_key).strip() if qdrant_api_key else None,
        timeout=timeout_seconds,
    )

    chunker = SectionAwareChunker(
        chunk_size=int(chunk_cfg.get("chunk_size", 1200)),
        overlap=int(chunk_cfg.get("overlap", 150)),
    )

    embedder = SentenceTransformersEmbeddingProvider(
        device=str(emb_cfg.get("device", "cpu")).strip() or "cpu",
        normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", False)),
        batch_size=int(emb_cfg.get("batch_size", 32)),
    )

    index = QdrantVectorIndex(
        client=client,
        collection_name_prefix=str(collection_cfg.get("name_prefix", "docs")).strip() or "docs",
        distance=parse_distance(str(collection_cfg.get("distance", "COSINE"))),
    )

    document_registry = VectorIndexDocumentRegistry(vector_index=index)

    service = IngestionUseCase(
        queue=queue,
        job_store=job_store,
        idempotency_store=idempotency,
        document_registry=document_registry,
        payload_store=payload_store,
        backpressure=backpressure,
        worker_count=WORKER_COUNT,
    )

    pipeline = DefaultIngestionPipeline(
        chunker=chunker,
        embedder=embedder,
        index=index,
    )

    search_use_case = SearchUseCase(
        embedder=embedder,
        searcher=index,
        chunks=index,
    )

    llm_cfg = settings.require_llm()
    ollama_base_url = str(llm_cfg.get("ollama_base_url", "")).strip()
    if not ollama_base_url:
        raise SystemError(
            code="missing_ollama_base_url",
            message="Missing ollama_base_url in llm.yaml",
            details=None,
            retryable=False,
        )
    llm_timeout_seconds = float(llm_cfg.get("timeout_seconds", 30.0))
    llm_max_concurrency = int(llm_cfg.get("llm_max_concurrency", 3))
    ask_llm_max_retries = int(llm_cfg.get("ask_llm_max_retries", 1))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pg_cfg = settings.require_postgres()
        db_engine, db_sessionmaker = create_engine_and_sessionmaker(pg_cfg=pg_cfg)
        app.state.db_engine = db_engine
        app.state.db_sessionmaker = db_sessionmaker
        from biomed_platform.audit.service import PostgresAuditService

        app.state.audit_service = PostgresAuditService(session_maker=db_sessionmaker)

        await run_migrations(pg_cfg=pg_cfg)

        llm_http_client = httpx.AsyncClient(timeout=httpx.Timeout(llm_timeout_seconds))
        pubmed_http_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
        llm_semaphore = asyncio.Semaphore(max(1, llm_max_concurrency))
        llm_client = OllamaLlmClient(
            base_url=ollama_base_url,
            client=llm_http_client,
            max_retries=ask_llm_max_retries,
            semaphore=llm_semaphore,
        )
        app.state.llm_http_client = llm_http_client
        app.state.llm_client = llm_client
        app.state.pubmed_http_client = pubmed_http_client
        app.state.pubmed_client = PubMedClientAdapter(client=pubmed_http_client, max_retries=2)

        worker_tasks: list[asyncio.Task[None]] = []

        for _ in range(WORKER_COUNT):
            worker = IngestionWorker(
                queue=queue,
                job_store=job_store,
                document_registry=document_registry,
                payload_store=payload_store,
                pipeline=pipeline,
            )
            worker_tasks.append(asyncio.create_task(worker.run_forever()))

        log.info(
            "API %s | version=%s | Description: %s",
            api_cfg.get("title"),
            api_cfg.get("version"),
            api_cfg.get("description"),
        )
        log.info(
            "RAG runtime configured, qdrant_url=%s, embedding_provider=%s",
            qdrant_url,
            str(emb_cfg.get("provider", "")).strip(),
        )

        try:
            yield
        finally:
            for task in worker_tasks:
                task.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)

            await llm_http_client.aclose()
            await pubmed_http_client.aclose()

            await db_engine.dispose()

    app = FastAPI(
        title=api_cfg.get("title"),
        description=api_cfg.get("description"),
        version=api_cfg.get("version"),
        lifespan=lifespan,
    )

    install_error_handlers(app)

    app.state.settings = settings
    app.state.ingestion_service = service
    app.state.document_registry = document_registry
    app.state.search_use_case = search_use_case
    app.state.embedding_provider = embedder
    app.state.vector_index = index

    app.include_router(system_router)
    app.include_router(v1_router)
    app.include_router(ingestion_router)
    app.include_router(documents_router)
    app.include_router(retrieval_router)
    app.include_router(ask_router)

    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(AuditMiddleware)
    app.add_middleware(RequestContextMiddleware)

    return app


app = create_app()
