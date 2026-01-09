from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import biomed_platform.core.services.ingestion.pipeline as pipeline_mod
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.services.ingestion.pipeline import DefaultIngestionPipeline


pytestmark = pytest.mark.asyncio


@dataclass(frozen=True)
class _Chunk:
    index: int
    text: str
    start: int
    end: int


class _FakeChunker:
    def __init__(self, chunks: list[_Chunk]) -> None:
        self._chunks = chunks
        self.calls: list[dict[str, Any]] = []

    def chunk(self, *, text: str) -> list[Any]:
        self.calls.append({"text": text})
        return list(self._chunks)


class _FakeEmbedder:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[dict[str, Any]] = []

    async def embed_texts(self, *, model_id: str, texts: list[str]) -> list[list[float]]:
        self.calls.append({"model_id": model_id, "texts": list(texts)})
        return [list(v) for v in self._vectors]


class _FakeIndex:
    def __init__(self, *, exists_result: bool = False) -> None:
        self._exists_result = exists_result
        self.exists_calls: list[dict[str, Any]] = []
        self.ensure_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []

    async def ensure_collection(self, *, embedding_model_id: str, vector_size: int) -> None:
        self.ensure_calls.append({"embedding_model_id": embedding_model_id, "vector_size": vector_size})

    async def exists(self, *, embedding_model_id: str, doc_id: str) -> bool:
        self.exists_calls.append({"embedding_model_id": embedding_model_id, "doc_id": doc_id})
        return self._exists_result

    async def upsert(self, *, embedding_model_id: str, points: list[Any]) -> None:
        self.upsert_calls.append({"embedding_model_id": embedding_model_id, "points": list(points)})


def _item(
    *,
    content_text: str | None = "hello world",
    doi_original: str = "10.1/A",
    doi_normalized: str = "10.1/a",
) -> Any:
    return type(
        "Item",
        (),
        {
            "content_text": content_text,
            "doi_original": doi_original,
            "doi_normalized": doi_normalized,
            "title": "t",
            "journal": "j",
            "year": 2026,
            "authors": ["a1", "a2"],
            "disease": "d",
            "source_type": "s",
        },
    )()


class TestDefaultIngestionPipeline:
    async def test_given_empty_content_text_when_ingest_item_then_raises_system_error_and_calls_nothing(
        self,
    ) -> None:
        # Given
        chunker = _FakeChunker(chunks=[_Chunk(index=0, text="x", start=0, end=1)])
        embedder = _FakeEmbedder(vectors=[[0.1, 0.2]])
        index = _FakeIndex()
        pipeline = DefaultIngestionPipeline(chunker=chunker, embedder=embedder, index=index)

        # When, Then
        with pytest.raises(SystemError) as exc:
            await pipeline.ingest_item(
                job_id="j1",
                embedding_model_id="e1",
                doc_id="d1",
                item=_item(content_text="   "),
            )

        assert exc.value.code == "empty_document_text"
        assert getattr(exc.value, "retryable", None) is False
        assert chunker.calls == []
        assert embedder.calls == []
        assert index.ensure_calls == []
        assert index.upsert_calls == []

    async def test_given_chunker_returns_no_chunks_when_ingest_item_then_raises_system_error(
        self,
    ) -> None:
        # Given
        chunker = _FakeChunker(chunks=[])
        embedder = _FakeEmbedder(vectors=[])
        index = _FakeIndex()
        pipeline = DefaultIngestionPipeline(chunker=chunker, embedder=embedder, index=index)

        # When, Then
        with pytest.raises(SystemError) as exc:
            await pipeline.ingest_item(
                job_id="j1",
                embedding_model_id="e1",
                doc_id="d1",
                item=_item(content_text="abc"),
            )

        assert exc.value.code == "no_chunks"
        assert getattr(exc.value, "retryable", None) is False
        assert len(chunker.calls) == 1
        assert embedder.calls == []
        assert index.ensure_calls == []
        assert index.upsert_calls == []

    async def test_given_embedding_count_mismatch_when_ingest_item_then_raises_retryable_system_error(
        self,
    ) -> None:
        # Given
        chunker = _FakeChunker(
            chunks=[
                _Chunk(index=0, text="c0", start=0, end=2),
                _Chunk(index=1, text="c1", start=2, end=4),
            ]
        )
        embedder = _FakeEmbedder(vectors=[[0.1, 0.2]])
        index = _FakeIndex()
        pipeline = DefaultIngestionPipeline(chunker=chunker, embedder=embedder, index=index)

        # When, Then
        with pytest.raises(SystemError) as exc:
            await pipeline.ingest_item(
                job_id="j1",
                embedding_model_id="e1",
                doc_id="d1",
                item=_item(content_text="abcd"),
            )

        assert exc.value.code == "embedding_count_mismatch"
        assert getattr(exc.value, "retryable", None) is True
        assert len(chunker.calls) == 1
        assert len(embedder.calls) == 1
        assert index.ensure_calls == []
        assert index.upsert_calls == []

    async def test_given_invalid_vector_size_when_ingest_item_then_raises_system_error(
        self,
    ) -> None:
        # Given
        chunker = _FakeChunker(chunks=[_Chunk(index=0, text="c0", start=0, end=2)])
        embedder = _FakeEmbedder(vectors=[[]])
        index = _FakeIndex()
        pipeline = DefaultIngestionPipeline(chunker=chunker, embedder=embedder, index=index)

        # When, Then
        with pytest.raises(SystemError) as exc:
            await pipeline.ingest_item(
                job_id="j1",
                embedding_model_id="e1",
                doc_id="d1",
                item=_item(content_text="ab"),
            )

        assert exc.value.code == "invalid_vector_size"
        assert getattr(exc.value, "retryable", None) is False
        assert len(index.ensure_calls) == 0
        assert len(index.upsert_calls) == 0

    async def test_given_happy_path_when_ingest_item_then_ensures_collection_and_upserts_points_with_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Given
        fixed_created_at = "2026-01-01T00:00:00Z"
        monkeypatch.setattr(pipeline_mod, "_utc_now_iso", lambda: fixed_created_at)

        chunker = _FakeChunker(
            chunks=[
                _Chunk(index=0, text="c0", start=0, end=2),
                _Chunk(index=1, text="c1", start=2, end=4),
            ]
        )
        embedder = _FakeEmbedder(vectors=[[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])
        index = _FakeIndex()
        pipeline = DefaultIngestionPipeline(chunker=chunker, embedder=embedder, index=index)

        item = _item(content_text="abcd", doi_original="10.1/A", doi_normalized="10.1/a")

        # When
        await pipeline.ingest_item(
            job_id="j1",
            embedding_model_id="e1",
            doc_id="doc123",
            item=item,
        )

        # Then
        assert index.ensure_calls == [{"embedding_model_id": "e1", "vector_size": 3}]
        assert len(index.upsert_calls) == 1

        call = index.upsert_calls[0]
        assert call["embedding_model_id"] == "e1"
        points = call["points"]
        assert len(points) == 2

        p0 = points[0]
        p1 = points[1]

        expected_p0_id = pipeline_mod._point_id(doc_id="doc123", chunk_index=0)
        expected_p1_id = pipeline_mod._point_id(doc_id="doc123", chunk_index=1)

        assert p0.point_id == expected_p0_id
        assert p1.point_id == expected_p1_id

        assert p0.vector == [0.1, 0.2, 0.3]
        assert p1.vector == [1.0, 2.0, 3.0]

        assert p0.payload["job_id"] == "j1"
        assert p0.payload["doc_id"] == "doc123"
        assert p0.payload["doi_original"] == "10.1/A"
        assert p0.payload["doi_normalized"] == "10.1/a"
        assert p0.payload["embedding_model_id"] == "e1"
        assert p0.payload["chunk_index"] == 0
        assert p0.payload["chunk_start"] == 0
        assert p0.payload["chunk_end"] == 2
        assert p0.payload["text"] == "c0"
        assert p0.payload["created_at"] == fixed_created_at

        assert p1.payload["chunk_index"] == 1
        assert p1.payload["chunk_start"] == 2
        assert p1.payload["chunk_end"] == 4
        assert p1.payload["text"] == "c1"
        assert p1.payload["created_at"] == fixed_created_at

        assert p0.payload["authors"] == ["a1", "a2"]
        assert p0.payload["title"] == "t"
        assert p0.payload["journal"] == "j"
        assert p0.payload["year"] == 2026
        assert p0.payload["disease"] == "d"
        assert p0.payload["source_type"] == "s"

    async def test_given_doc_already_indexed_when_ingest_item_then_raises_duplicate_doc_and_does_not_upsert(
            self,
    ) -> None:
        # Given
        chunker = _FakeChunker(chunks=[_Chunk(index=0, text="c0", start=0, end=2)])
        embedder = _FakeEmbedder(vectors=[[0.1, 0.2, 0.3]])
        index = _FakeIndex(exists_result=True)
        pipeline = DefaultIngestionPipeline(chunker=chunker, embedder=embedder, index=index)

        # When, Then
        with pytest.raises(SystemError) as exc:
            await pipeline.ingest_item(
                job_id="j1",
                embedding_model_id="e1",
                doc_id="doc123",
                item=_item(content_text="abcd"),
            )

        assert exc.value.code == "duplicate_doc"
        assert getattr(exc.value, "retryable", None) is False
        assert len(index.ensure_calls) == 1
        assert index.exists_calls == [{"embedding_model_id": "e1", "doc_id": "doc123"}]
        assert index.upsert_calls == []

