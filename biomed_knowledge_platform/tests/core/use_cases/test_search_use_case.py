from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from biomed_platform.api.models.generated import schemas
from biomed_platform.core.domains.retrieval import ChunkPart, VectorSearchHit
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.use_cases.search import (
    SearchUseCase,
    _assemble_full_text,
    _parse_authors,
    _to_filter_payload,
)


@dataclass(slots=True)
class _FakeHit:
    point_id: str
    score: float
    payload: dict


@pytest.mark.anyio
async def test_execute_rejects_empty_query() -> None:
    # Given, empty query
    uc = SearchUseCase(embedder=AsyncMock(), searcher=AsyncMock(), chunks=AsyncMock())

    # When, Then
    with pytest.raises(SystemError) as exc:
        await uc.execute(
            request_id="r",
            embedding_model_id="m",
            req=schemas.SearchRequest(query="   ", top_k=1, filters=None),
        )

    assert exc.value.code == "validation_error"


@pytest.mark.anyio
async def test_execute_embedding_failed_when_no_vectors() -> None:
    # Given, embedder returns no vectors
    embedder = AsyncMock()
    embedder.embed_texts = AsyncMock(return_value=[])

    uc = SearchUseCase(embedder=embedder, searcher=AsyncMock(), chunks=AsyncMock())

    # When, Then
    with pytest.raises(SystemError) as exc:
        await uc.execute(
            request_id="r",
            embedding_model_id="m",
            req=schemas.SearchRequest(query="q", top_k=1, filters=None),
        )

    assert exc.value.code == "embedding_failed"


@pytest.mark.anyio
async def test_execute_no_doc_ids_returns_empty_hits_and_skips_chunk_fetch() -> None:
    # Given, search results without doc_id
    embedder = AsyncMock()
    embedder.embed_texts = AsyncMock(return_value=[[0.1]])

    searcher = AsyncMock()
    searcher.search = AsyncMock(
        return_value=[
            VectorSearchHit(point_id="p1", score=0.5, payload={"doc_id": ""}),
            VectorSearchHit(point_id="p2", score=0.4, payload={}),
        ]
    )

    chunks = AsyncMock()

    uc = SearchUseCase(embedder=embedder, searcher=searcher, chunks=chunks)

    # When
    res = await uc.execute(
        request_id="r",
        embedding_model_id="m",
        req=schemas.SearchRequest(query="q", top_k=3, filters=None),
    )

    # Then
    assert res.hits == []
    chunks.fetch_by_doc_ids.assert_not_called()


@pytest.mark.anyio
async def test_execute_parses_invalid_source_type_and_truncates_and_uses_chunk_index_sort() -> None:
    # Given, one best doc plus one doc with invalid source_type
    embedder = AsyncMock()
    embedder.embed_texts = AsyncMock(return_value=[[0.1]])

    searcher = AsyncMock()
    searcher.search = AsyncMock(
        return_value=[
            VectorSearchHit(
                point_id="p_bad",
                score=0.2,
                payload={
                    "doc_id": "d_bad",
                    "doi_original": "10.1/bad",
                    "title": "tb",
                    "year": 2020,
                    "source_type": "not_a_real_enum",
                    "authors": [{"root": "a"}],
                    "journal": 123,  # non string branch
                },
            ),
            VectorSearchHit(
                point_id="p_best",
                score=0.9,
                payload={
                    "doc_id": "d_best",
                    "doi": "10.1/best",  # doi fallback
                    "title": "t",
                    "year": 2021,
                    "source_type": schemas.SourceType.pubmed_abstract.value,
                    "authors": [" b "],
                    "journal": "j",
                },
            ),
            VectorSearchHit(
                point_id="p_best2",
                score=0.1,
                payload={"doc_id": "d_best"},
            ),
        ]
    )

    # Chunks arrive out of order, should be sorted by chunk_index for chunk_ids,
    # and chunk_end=-1 triggers the "end < 0" branch.
    long_text = "x" * 3000
    chunks = AsyncMock()
    chunks.fetch_by_doc_ids = AsyncMock(
        return_value=[
            VectorSearchHit(
                point_id="c2",
                score=0.0,
                payload={
                    "doc_id": "d_best",
                    "chunk_index": 1,
                    "chunk_start": 5,
                    "chunk_end": -1,
                    "text": long_text,
                },
            ),
            VectorSearchHit(
                point_id="c1",
                score=0.0,
                payload={
                    "doc_id": "d_best",
                    "chunk_index": 0,
                    "chunk_start": 0,
                    "chunk_end": 5,
                    "text": "hello",
                },
            ),
        ]
    )

    uc = SearchUseCase(embedder=embedder, searcher=searcher, chunks=chunks)

    req = schemas.SearchRequest(
        query="q",
        top_k=1,
        filters=schemas.SearchFilters(
            disease=schemas.Disease.thrombosis,
            source_type=schemas.SourceType.pubmed_abstract,
            year_min=2000,
            year_max=2024,
        ),
    )

    # When
    res = await uc.execute(request_id="r", embedding_model_id="m", req=req)

    # Then
    assert len(res.hits) == 1
    hit = res.hits[0]
    assert hit.doc_id == "d_best"
    assert hit.doi == "10.1/best"
    assert hit.authors == ["b"]
    assert hit.source_type == schemas.SourceType.pubmed_abstract

    assert hit.chunk_ids == ["c1", "c2"]  # sorted by chunk_index
    assert hit.content_text is not None
    assert len(hit.content_text) == 2000  # truncation branch


def test_helpers_cover_remaining_branches() -> None:
    # Given, parse authors variants
    assert _parse_authors(None) is None
    assert _parse_authors("   ") is None
    assert _parse_authors("x") == ["x"]
    assert _parse_authors(SimpleNamespace(root=" y ")) == ["y"]
    assert _parse_authors(
        [" a ", {"root": "b"}, SimpleNamespace(root="c"), {"root": "   "}, 1]
    ) == ["a", "b", "c"]

    # Given, assemble empty
    assert _assemble_full_text([]) == ""

    # Given, overlapping, empty text, and a fully consumed segment
    parts = [
        ChunkPart(chunk_id="1", chunk_index=0, start=0, end=5, text="hello"),
        ChunkPart(chunk_id="2", chunk_index=1, start=0, end=0, text=""),  # empty, skipped
        ChunkPart(chunk_id="3", chunk_index=2, start=3, end=8, text="lo wo"),  # overlap, cuts to " wo"
        ChunkPart(chunk_id="4", chunk_index=3, start=1, end=2, text="xx"),  # fully consumed, skipped
    ]

    # When
    full = _assemble_full_text(parts)

    # Then
    assert full == "hello wo"

    # Given, filters none and empty
    assert _to_filter_payload(None) is None
    assert _to_filter_payload(schemas.SearchFilters()) is None

    # Given, filters present
    payload = _to_filter_payload(
        schemas.SearchFilters(
            disease=schemas.Disease.thrombosis,
            source_type=schemas.SourceType.full_text,
            year_min=2000,
            year_max=2024,
        )
    )
    assert payload == {
        "disease": "thrombosis",
        "source_type": "full_text",
        "year_min": 2000,
        "year_max": 2024,
    }
