from __future__ import annotations

from typing import Protocol

from scheduler_pubmed.src.core.domains.scheduler import FetchBatchAccepted, IngestJobStatus


class DocumentsClient(Protocol):
    async def document_exists(self, *, doi: str) -> bool:
        raise NotImplementedError

    async def fetch_batch(self, *, dois: list[str]) -> FetchBatchAccepted:
        raise NotImplementedError

    async def get_ingest_job_status(self, *, job_id: str) -> IngestJobStatus:
        raise NotImplementedError
