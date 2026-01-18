from __future__ import annotations

from typing import Protocol, Sequence

from biomed_platform.core.domains.retrieval import VectorSearchHit


class VectorSearcher(Protocol):
    async def search(
        self,
        *,
        embedding_model_id: str,
        query_vector: Sequence[float],
        top_k: int,
        qfilter: object | None,
    ) -> list[VectorSearchHit]: ...


class ChunkStore(Protocol):
    async def fetch_by_doc_ids(
        self,
        *,
        embedding_model_id: str,
        doc_ids: Sequence[str],
        base_filter: object | None,
        limit: int,
    ) -> list[VectorSearchHit]: ...
