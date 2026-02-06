from __future__ import annotations

import pytest

from biomed_platform.core.services.ingestion.vector_index_document_registry import (
    VectorIndexDocumentRegistry,
)


class _FakeVectorIndex:
    def __init__(self) -> None:
        self._exists: dict[tuple[str, str], bool] = {}
        self.raise_on_exists: Exception | None = None

    def set_exists(self, *, embedding_model_id: str, doc_id: str, exists: bool) -> None:
        self._exists[(embedding_model_id, doc_id)] = exists

    async def exists(self, *, embedding_model_id: str, doc_id: str) -> bool:
        if self.raise_on_exists is not None:
            raise self.raise_on_exists
        return self._exists.get((embedding_model_id, doc_id), False)


@pytest.mark.asyncio
async def test_registry_reserve_blocks_when_doc_exists_in_vector_index() -> None:
    index = _FakeVectorIndex()
    index.set_exists(embedding_model_id="m", doc_id="d", exists=True)
    reg = VectorIndexDocumentRegistry(vector_index=index)

    with pytest.raises(KeyError):
        await reg.reserve(embedding_model_id="m", doc_id="d")


@pytest.mark.asyncio
async def test_registry_release_allows_re_reserve_when_not_in_vector_index() -> None:
    index = _FakeVectorIndex()
    reg = VectorIndexDocumentRegistry(vector_index=index)

    await reg.reserve(embedding_model_id="m", doc_id="d")
    await reg.release(embedding_model_id="m", doc_id="d")

    await reg.reserve(embedding_model_id="m", doc_id="d")


@pytest.mark.asyncio
async def test_registry_reserve_rolls_back_reservation_on_exists_error() -> None:
    index = _FakeVectorIndex()
    index.raise_on_exists = RuntimeError("qdrant down")
    reg = VectorIndexDocumentRegistry(vector_index=index)

    with pytest.raises(RuntimeError):
        await reg.reserve(embedding_model_id="m", doc_id="d")

    index.raise_on_exists = None
    await reg.reserve(embedding_model_id="m", doc_id="d")

