from __future__ import annotations

import asyncio

from biomed_platform.common.logging import get_logger
from biomed_platform.core.services.ingestion_ports import IngestionQueue

log = get_logger(__name__)


class InMemoryIngestionQueue(IngestionQueue):
    def __init__(self, *, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        self._q: asyncio.Queue[str] = asyncio.Queue(maxsize=max_size)
        self._max_size = max_size

        log.debug(
            "Ingestion queue initialized, max_size=%d",
            max_size,
        )

    def max_size(self) -> int:
        return self._max_size

    def size(self) -> int:
        return self._q.qsize()

    async def enqueue(self, job_id: str) -> None:
        size_before = self._q.qsize()

        self._q.put_nowait(job_id)

        log.debug(
            "Job enqueued, job_id=%s, size_before=%d, size_after=%d, max_size=%d",
            job_id,
            size_before,
            size_before + 1,
            self._max_size,
        )

    async def dequeue(self) -> str:
        size_before = self._q.qsize()

        job_id = await self._q.get()

        log.debug(
            "Job dequeued, job_id=%s, size_before=%d, size_after=%d",
            job_id,
            size_before,
            max(size_before - 1, 0),
        )

        return job_id
