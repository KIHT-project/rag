from __future__ import annotations

import pytest

from biomed_platform.core.services.ingestion.in_memory_document_registry import InMemoryDocumentRegistry


@pytest.mark.asyncio
async def test_registry_reserve_commit_blocks_duplicates() -> None:
    # Given a document registry
    reg = InMemoryDocumentRegistry()

    # When reserving and committing a doc id
    await reg.reserve(embedding_model_id="m", doc_id="d")
    await reg.commit(embedding_model_id="m", doc_id="d")

    # Then reserving again raises KeyError
    with pytest.raises(KeyError):
        await reg.reserve(embedding_model_id="m", doc_id="d")


@pytest.mark.asyncio
async def test_registry_release_allows_re_reserve() -> None:
    # Given a document registry with a reserved doc id
    reg = InMemoryDocumentRegistry()
    await reg.reserve(embedding_model_id="m", doc_id="d")

    # When releasing it
    await reg.release(embedding_model_id="m", doc_id="d")

    # Then it can be reserved again
    await reg.reserve(embedding_model_id="m", doc_id="d")
