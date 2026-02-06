from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from biomed_platform.common.logging import get_logger
from biomed_platform.core.ports.ingestion import DocumentRegistry, VectorWriter

log = get_logger(__name__)


@dataclass(slots=True)
class _ModelSpace:
    reserved: set[str] = field(default_factory=set)


class VectorIndexDocumentRegistry(DocumentRegistry):
    """
    In-process doc_id registry backed by the vector index for "committed" state.

    The in-memory registry tracks only in-flight ("reserved") doc_ids, while the
    underlying vector index is the source of truth for whether a document exists.
    This avoids stale "already exists" behavior across multi-worker / multi-process
    API deployments.
    """

    def __init__(self, *, vector_index: VectorWriter) -> None:
        self._vector_index = vector_index
        self._lock = asyncio.Lock()
        self._spaces: dict[str, _ModelSpace] = {}

    def _space(self, embedding_model_id: str) -> _ModelSpace:
        return self._spaces.setdefault(embedding_model_id, _ModelSpace())

    async def reserve(self, *, embedding_model_id: str, doc_id: str) -> None:
        async with self._lock:
            space = self._space(embedding_model_id)

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
                "Doc_id reserved, embedding_model_id=%s, doc_id=%s, reserved_count=%d",
                embedding_model_id,
                doc_id,
                len(space.reserved),
            )

        try:
            exists = await self._vector_index.exists(
                embedding_model_id=embedding_model_id,
                doc_id=doc_id,
            )
        except Exception:
            async with self._lock:
                self._space(embedding_model_id).reserved.discard(doc_id)
            raise

        if not exists:
            return

        async with self._lock:
            self._space(embedding_model_id).reserved.discard(doc_id)

        log.warning(
            "Attempt to reserve already existing doc_id (vector index), "
            "embedding_model_id=%s, doc_id=%s",
            embedding_model_id,
            doc_id,
        )
        raise KeyError(doc_id)

    async def commit(self, *, embedding_model_id: str, doc_id: str) -> None:
        async with self._lock:
            space = self._space(embedding_model_id)
            existed = doc_id in space.reserved
            space.reserved.discard(doc_id)

            log.debug(
                "Doc_id committed (released), embedding_model_id=%s, doc_id=%s, "
                "was_reserved=%s, reserved_count=%d",
                embedding_model_id,
                doc_id,
                existed,
                len(space.reserved),
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

    async def is_reserved(self, *, embedding_model_id: str, doc_id: str) -> bool:
        async with self._lock:
            space = self._space(embedding_model_id)
            return doc_id in space.reserved

    async def is_committed(self, *, embedding_model_id: str, doc_id: str) -> bool:
        return await self._vector_index.exists(
            embedding_model_id=embedding_model_id,
            doc_id=doc_id,
        )

    async def delete(self, *, embedding_model_id: str, doc_id: str) -> None:
        async with self._lock:
            space = self._space(embedding_model_id)
            was_reserved = doc_id in space.reserved
            space.reserved.discard(doc_id)

            log.debug(
                "Doc_id deleted (released), embedding_model_id=%s, doc_id=%s, "
                "was_reserved=%s, reserved_count=%d",
                embedding_model_id,
                doc_id,
                was_reserved,
                len(space.reserved),
            )
