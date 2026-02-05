from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from biomed_platform.core.domains.ingestion import IngestItem
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.services.ingestion.pipeline import DefaultIngestionPipeline, _point_id


@dataclass(frozen=True, slots=True)
class _Chunk:
    index: int
    start: int
    end: int
    text: str
    section: str | None = None
    subsection: str | None = None


@pytest.mark.anyio
async def test_ingest_item_fails_when_chunker_returns_no_chunks() -> None:
    # Given, a pipeline where chunker returns no chunks
    chunker = MagicMock()
    chunker.chunk = MagicMock(return_value=[])

    pipe = DefaultIngestionPipeline(chunker=chunker, embedder=AsyncMock(), index=AsyncMock())

    item = IngestItem(
        doi_original="10.1/x",
        doi_normalized="10.1/x",
        title=None,
        journal=None,
        year=None,
        authors=[],
        disease=None,
        source_type=None,
        content_text="hello",
    )

    # When
    with pytest.raises(SystemError) as exc:
        await pipe.ingest_item(job_id="j", embedding_model_id="m", doc_id="d", item=item)

    # Then
    assert exc.value.code == "no_chunks"


@pytest.mark.anyio
async def test_ingest_item_fails_on_embedding_count_mismatch_and_invalid_vector_size() -> None:
    # Given, two chunks and one vector
    chunker = MagicMock()
    chunker.chunk = MagicMock(return_value=[_Chunk(0, 0, 1, "a"), _Chunk(1, 1, 2, "b")])

    embedder = AsyncMock()
    embedder.embed_texts = AsyncMock(return_value=[[0.1]])

    index = AsyncMock()
    index.ensure_collection = AsyncMock()
    index.exists = AsyncMock(return_value=False)

    pipe = DefaultIngestionPipeline(chunker=chunker, embedder=embedder, index=index)

    item = IngestItem(
        doi_original="10.1/x",
        doi_normalized="10.1/x",
        title=None,
        journal=None,
        year=None,
        authors=[],
        disease=None,
        source_type=None,
        content_text="hello",
    )

    # When, vectors mismatch chunks
    with pytest.raises(SystemError) as exc:
        await pipe.ingest_item(job_id="j", embedding_model_id="m", doc_id="d", item=item)

    # Then
    assert exc.value.code == "embedding_count_mismatch"

    # Given, one chunk and a zero length vector
    chunker.chunk = MagicMock(return_value=[_Chunk(0, 0, 1, "a")])
    embedder.embed_texts = AsyncMock(return_value=[[]])

    # When
    with pytest.raises(SystemError) as exc2:
        await pipe.ingest_item(job_id="j", embedding_model_id="m", doc_id="d", item=item)

    # Then
    assert exc2.value.code == "invalid_vector_size"


@pytest.mark.anyio
async def test_ingest_item_detects_duplicate_doc_and_success_upserts_points() -> None:
    # Given, a pipeline with valid chunking and embeddings
    chunker = MagicMock()
    chunker.chunk = MagicMock(return_value=[_Chunk(0, 0, 5, "hello"), _Chunk(1, 5, 10, "world")])

    embedder = AsyncMock()
    embedder.embed_texts = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

    index = AsyncMock()
    index.ensure_collection = AsyncMock()

    pipe = DefaultIngestionPipeline(chunker=chunker, embedder=embedder, index=index)

    item = IngestItem(
        doi_original="10.1/x",
        doi_normalized="10.1/x",
        title="t",
        journal="j",
        year=2020,
        authors=["a"],
        disease="disease",
        source_type="abstract",
        content_text="hello world",
    )

    # When, document already exists
    index.exists = AsyncMock(return_value=True)
    with pytest.raises(SystemError) as exc:
        await pipe.ingest_item(job_id="j1", embedding_model_id="m1", doc_id="doc", item=item)

    # Then
    assert exc.value.code == "duplicate_doc"

    # Given, it does not exist
    index.exists = AsyncMock(return_value=False)
    index.upsert = AsyncMock()

    # When, ingest succeeds
    await pipe.ingest_item(job_id="j1", embedding_model_id="m1", doc_id="doc", item=item)

    # Then, it upserts exactly two points with stable point ids
    args = index.upsert.await_args.kwargs
    assert args["embedding_model_id"] == "m1"
    pts = list(args["points"])
    assert len(pts) == 2
    assert pts[0].point_id == _point_id(doc_id="doc", chunk_index=0)
    assert pts[1].point_id == _point_id(doc_id="doc", chunk_index=1)
    assert pts[0].payload["doc_id"] == "doc"
    assert pts[0].payload["chunk_index"] == 0


@pytest.mark.anyio
async def test_ingest_item_stores_section_from_chunks() -> None:
    chunker = MagicMock()
    chunker.chunk = MagicMock(return_value=[_Chunk(0, 0, 5, "text", section="Methods")])

    embedder = AsyncMock()
    embedder.embed_texts = AsyncMock(return_value=[[0.1, 0.2]])

    index = AsyncMock()
    index.ensure_collection = AsyncMock()
    index.exists = AsyncMock(return_value=False)
    index.upsert = AsyncMock()

    pipe = DefaultIngestionPipeline(chunker=chunker, embedder=embedder, index=index)

    item = IngestItem(
        doi_original="10.1/x",
        doi_normalized="10.1/x",
        title="t",
        journal="j",
        year=2020,
        authors=["a"],
        disease="disease",
        source_type="abstract",
        content_text="hello world",
    )

    await pipe.ingest_item(job_id="j1", embedding_model_id="m1", doc_id="doc", item=item)

    args = index.upsert.await_args.kwargs
    pts = list(args["points"])
    assert len(pts) == 1
    assert pts[0].payload["section"] == "Methods"
