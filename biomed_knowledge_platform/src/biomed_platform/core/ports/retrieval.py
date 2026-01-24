from __future__ import annotations

from typing import Protocol, Sequence

from biomed_platform.core.domains.retrieval import ChunkCandidate, VectorSearchHit


class VectorSearcher(Protocol):
    async def search(
        self,
        *,
        embedding_model_id: str,
        query_vector: Sequence[float],
        top_k: int,
        qfilter: object | None,
    ) -> list[VectorSearchHit]: ...

    async def search_chunks(
        self,
        *,
        embedding_model_id: str,
        query_vector: Sequence[float],
        top_k: int,
        qfilter: object | None,
    ) -> list[ChunkCandidate]: ...


class ChunkStore(Protocol):
    async def fetch_by_doc_ids(
        self,
        *,
        embedding_model_id: str,
        doc_ids: Sequence[str],
        base_filter: object | None,
        limit: int,
    ) -> list[VectorSearchHit]: ...

    async def fetch_chunks_by_ids(
        self,
        *,
        embedding_model_id: str,
        chunk_ids: Sequence[str],
    ) -> list[ChunkCandidate]: ...
