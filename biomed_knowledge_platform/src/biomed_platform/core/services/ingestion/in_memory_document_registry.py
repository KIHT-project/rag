from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from biomed_platform.common.logging import get_logger
from biomed_platform.core.ports.ingestion import DocumentRegistry

log = get_logger(__name__)


@dataclass(slots=True)
class _ModelSpace:
    reserved: set[str] = field(default_factory=set)
    committed: set[str] = field(default_factory=set)


class InMemoryDocumentRegistry(DocumentRegistry):
    """
    In process DOI to doc_id registry.

    reserved: doc_ids in flight, queued or running
    committed: doc_ids completed successfully
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._spaces: dict[str, _ModelSpace] = {}

    def _space(self, embedding_model_id: str) -> _ModelSpace:
        return self._spaces.setdefault(embedding_model_id, _ModelSpace())

    async def reserve(self, *, embedding_model_id: str, doc_id: str) -> None:
        async with self._lock:
            space = self._space(embedding_model_id)

            if doc_id in space.committed:
                log.warning(
                    "Attempt to reserve already committed doc_id, "
                    "embedding_model_id=%s, doc_id=%s",
                    embedding_model_id,
                    doc_id,
                )
                raise KeyError(doc_id)

            if doc_id in space.reserved:
                log.debug(
                    "Attempt to reserve already reserved doc_id, "
                    "embedding_model_id=%s, doc_id=%s",
                    embedding_model_id,
                    doc_id,
                )
                raise KeyError(doc_id)

            space.reserved.add(doc_id)

            log.debug(
                "Doc_id reserved, embedding_model_id=%s, doc_id=%s, "
                "reserved_count=%d, committed_count=%d",
                embedding_model_id,
                doc_id,
                len(space.reserved),
                len(space.committed),
            )

    async def commit(self, *, embedding_model_id: str, doc_id: str) -> None:
        async with self._lock:
            space = self._space(embedding_model_id)

            was_reserved = doc_id in space.reserved
            space.reserved.discard(doc_id)
            space.committed.add(doc_id)

            log.debug(
                "Doc_id committed, embedding_model_id=%s, doc_id=%s, "
                "was_reserved=%s, reserved_count=%d, committed_count=%d",
                embedding_model_id,
                doc_id,
                was_reserved,
                len(space.reserved),
                len(space.committed),
            )

    async def release(self, *, embedding_model_id: str, doc_id: str) -> None:
        async with self._lock:
            space = self._space(embedding_model_id)

            existed = doc_id in space.reserved
            space.reserved.discard(doc_id)

            log.debug(
                "Doc_id released, embedding_model_id=%s, doc_id=%s, "
                "was_reserved=%s, reserved_count=%d",
                embedding_model_id,
                doc_id,
                existed,
                len(space.reserved),
            )
