from __future__ import annotations

from datetime import datetime
from typing import Protocol

from biomed_platform.core.domains.ingestion import (
    IngestBatchAccepted,
    IngestBatchCommand,
    IngestionJob,
    RetryAfterHint,
)


class IngestionQueue(Protocol):
    def max_size(self) -> int: ...

    def size(self) -> int: ...

    async def enqueue(self, job_id: str) -> None: ...

    async def dequeue(self) -> str: ...


class IngestionJobStore(Protocol):
    async def create(self, job: IngestionJob) -> None: ...

    async def get(self, job_id: str) -> IngestionJob: ...

    async def update(self, job: IngestionJob) -> None: ...

    async def delete(self, job_id: str) -> None: ...


class IdempotencyStore(Protocol):
    async def get_job_id(self, *, key: str, body_hash: str) -> str | None: ...

    async def put(
        self,
        *,
        key: str,
        body_hash: str,
        job_id: str,
        created_at: datetime,
    ) -> None: ...


class DocumentExistenceChecker(Protocol):
    async def exists(self, *, embedding_model_id: str, doc_id: str) -> bool: ...


class IngestionService(Protocol):
    async def ingest_batch(self, cmd: IngestBatchCommand) -> IngestBatchAccepted: ...

    async def get_job_status(self, *, job_id: str) -> IngestionJob: ...


class BackpressurePolicy(Protocol):
    def retry_after(
        self,
        *,
        queue_depth: int,
        queue_max_size: int,
        worker_count: int,
    ) -> RetryAfterHint: ...


class DocumentRegistry(Protocol):
    async def reserve(self, *, embedding_model_id: str, doc_id: str) -> None: ...

    async def commit(self, *, embedding_model_id: str, doc_id: str) -> None: ...

    async def release(self, *, embedding_model_id: str, doc_id: str) -> None: ...
