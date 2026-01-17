# tests/api/endpoints/test_retrieval.py
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, Range

from biomed_platform.api.endpoints import retrieval as mod
from biomed_platform.api.models.generated import schemas
from biomed_platform.core.domains.retrieval import ChunkPart
from biomed_platform.core.errors.errors import SystemError


@dataclass
class _FakeSettings:
    provider: str | None

    def require_rag(self) -> dict[str, Any]:
        return {"embedding": {"provider": self.provider}}


class _FakeEmbedder:
    def __init__(self, vectors: list[list[float]] | None) -> None:
        self.vectors = vectors
        self.calls: list[dict[str, Any]] = []

    async def embed_texts(self, *, model_id: str, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append({"model_id": model_id, "texts": list(texts)})
        return list(self.vectors or [])


class _FakeVectorIndex:
    def __init__(
        self,
        *,
        search_hits: list[Any] | None = None,
        fetch_hits: list[Any] | None = None,
        has_fetch: bool = True,
    ) -> None:
        self.search_hits = list(search_hits or [])
        self.fetch_hits = list(fetch_hits or [])
        self.has_fetch = has_fetch

        self.calls: list[dict[str, Any]] = []

    async def search(
        self,
        *,
        embedding_model_id: str,
        query_vector: Sequence[float],
        top_k: int,
        qfilter: object | None,
    ) -> list[Any]:
        self.calls.append(
            {
                "op": "search",
                "embedding_model_id": embedding_model_id,
                "query_vector": list(query_vector),
                "top_k": int(top_k),
                "qfilter": qfilter,
            }
        )
        return list(self.search_hits)

    async def fetch_by_doc_ids(
        self,
        *,
        embedding_model_id: str,
        doc_ids: Sequence[str],
        base_filter: object | None,
        limit: int = 20000,
    ) -> list[Any]:
        if not self.has_fetch:
            raise AssertionError("fetch_by_doc_ids should not be called")

        self.calls.append(
            {
                "op": "fetch_by_doc_ids",
                "embedding_model_id": embedding_model_id,
                "doc_ids": list(doc_ids),
                "base_filter": base_filter,
                "limit": int(limit),
            }
        )
        return list(self.fetch_hits)


def _mk_app(*, provider: str | None, embedder: object | None, index: object | None) -> FastAPI:
    app = FastAPI()
    app.include_router(mod.router)

    app.state.settings = _FakeSettings(provider=provider)
    app.state.embedding_provider = embedder
    app.state.vector_index = index
    return app


def _raw_hit(*, doc_id: str, score: float, extra: dict[str, Any] | None = None) -> Any:
    payload = {"doc_id": doc_id}
    if extra:
        payload.update(extra)
    return SimpleNamespace(point_id="p0", score=float(score), payload=payload)


def _chunk_hit(
    *,
    point_id: str,
    doc_id: str,
    chunk_index: int,
    start: int,
    end: int,
    text: str,
) -> Any:
    return SimpleNamespace(
        point_id=point_id,
        score=0.0,
        payload={
            "doc_id": doc_id,
            "chunk_index": int(chunk_index),
            "chunk_start": int(start),
            "chunk_end": int(end),
            "text": text,
        },
    )


class TestResolveEffectiveEmbeddingModelId:
    def test_given_valid_provider_when_resolve_then_returns_trimmed(self) -> None:
        app = _mk_app(provider="  m1  ", embedder=None, index=None)
        req = SimpleNamespace(app=app)

        got = mod._resolve_effective_embedding_model_id(request=req)  # type: ignore[arg-type]

        assert got == "m1"

    def test_given_missing_provider_when_resolve_then_raises_system_error(self) -> None:
        app = _mk_app(provider="   ", embedder=None, index=None)
        req = SimpleNamespace(app=app)

        with pytest.raises(SystemError) as exc:
            mod._resolve_effective_embedding_model_id(request=req)  # type: ignore[arg-type]

        assert exc.value.code == "missing_embedding_model_id"
        assert exc.value.retryable is False


class TestHelpers:
    def test_truncate_text_when_below_limit_then_unchanged(self) -> None:
        assert mod._truncate_text("abc") == "abc"

    def test_truncate_text_when_above_limit_then_truncated(self) -> None:
        s = "x" * (mod._TEXT_TRUNCATE_LIMIT + 10)
        got = mod._truncate_text(s)
        assert len(got) == mod._TEXT_TRUNCATE_LIMIT
        assert got == s[: mod._TEXT_TRUNCATE_LIMIT]

    def test_to_qdrant_filter_when_none_then_none(self) -> None:
        assert mod._to_qdrant_filter(None) is None

    def test_to_qdrant_filter_when_all_empty_then_none(self) -> None:
        f = schemas.SearchFilters()
        assert mod._to_qdrant_filter(f) is None

    def test_to_qdrant_filter_when_disease_then_field_condition(self) -> None:
        f = schemas.SearchFilters(disease=schemas.Disease.thrombosis)
        q = mod._to_qdrant_filter(f)
        assert isinstance(q, Filter)
        assert q.must and len(q.must) == 1
        c0 = q.must[0]
        assert isinstance(c0, FieldCondition)
        assert c0.key == "disease"

    def _any_source_type(self) -> schemas.SourceType:
        members = list(schemas.SourceType)  # type: ignore[arg-type]
        assert members, "schemas.SourceType has no members"
        return members[0]

    def test_to_qdrant_filter_when_source_type_then_field_condition(self) -> None:
        f = schemas.SearchFilters(source_type=self._any_source_type())
        q = mod._to_qdrant_filter(f)

        assert isinstance(q, Filter)
        assert q.must and len(q.must) == 1
        c0 = q.must[0]
        assert isinstance(c0, FieldCondition)
        assert c0.key == "source_type"
        assert getattr(getattr(c0, "match", None), "value", None) == f.source_type.value

    def test_to_qdrant_filter_when_year_range_then_range_condition(self) -> None:
        f = schemas.SearchFilters(year_min=2019, year_max=2023)
        q = mod._to_qdrant_filter(f)
        assert isinstance(q, Filter)
        assert q.must and len(q.must) == 1
        c0 = q.must[0]
        assert isinstance(c0, FieldCondition)
        assert c0.key == "year"
        assert isinstance(c0.range, Range)
        assert c0.range.gte == 2019
        assert c0.range.lte == 2023

    def test_assemble_full_text_orders_and_dedupes_overlaps(self) -> None:
        parts = [
            ChunkPart(chunk_id="c1", chunk_index=1, start=0, end=5, text="hello"),
            ChunkPart(chunk_id="c2", chunk_index=2, start=3, end=8, text="lo123"),
            ChunkPart(chunk_id="c3", chunk_index=3, start=8, end=10, text="45"),
        ]
        got = mod._assemble_full_text(parts)
        assert got == "hello12345"

    def test_assemble_full_text_skips_empty_text(self) -> None:
        parts = [
            ChunkPart(chunk_id="c1", chunk_index=1, start=0, end=1, text=""),
            ChunkPart(chunk_id="c2", chunk_index=2, start=0, end=2, text="ab"),
        ]
        got = mod._assemble_full_text(parts)
        assert got == "ab"

    def test_parse_authors_accepts_strings_and_objects(self) -> None:
        assert mod._parse_authors(None) is None
        assert mod._parse_authors("  A  ") == ["A"]
        assert mod._parse_authors("   ") is None

        assert mod._parse_authors([" a ", "", "b"]) == ["a", "b"]
        assert mod._parse_authors([{"root": " x "}, {"root": " "}, {}]) == ["x"]

        obj = SimpleNamespace(root=" y ")
        assert mod._parse_authors([obj]) == ["y"]


@pytest.mark.parametrize(
    "code, expected",
    [
        ("validation_error", 400),
        ("invalid_model_id", 400),
        ("too_many_requests", 429),
        ("other", 500),
    ],
)
def test_status_for_error_code(code: str, expected: int) -> None:
    assert mod._status_for_error_code(code) == expected


class TestSearchChunksEndpoint:
    def test_given_happy_path_with_fetch_then_returns_hits_with_assembled_text(self) -> None:
        embedder = _FakeEmbedder(vectors=[[0.1, 0.2]])
        index = _FakeVectorIndex(
            search_hits=[
                _raw_hit(
                    doc_id="d1",
                    score=0.9,
                    extra={
                        "doi_original": "10.1/x",
                        "title": "T",
                        "year": 2020,
                        "source_type": "abstract",
                        "journal": "J",
                        "authors": ["Alice", {"root": "Bob"}],
                    },
                ),
                _raw_hit(doc_id="d1", score=0.3, extra={"title": "ignored lower score"}),
                _raw_hit(doc_id="d2", score=0.8, extra={"doi": "10.2/y"}),
            ],
            fetch_hits=[
                _chunk_hit(point_id="c1", doc_id="d1", chunk_index=0, start=0, end=5, text="hello"),
                _chunk_hit(point_id="c2", doc_id="d1", chunk_index=1, start=3, end=8, text="lo123"),
                _chunk_hit(point_id="c3", doc_id="d2", chunk_index=0, start=0, end=4, text="abcd"),
            ],
            has_fetch=True,
        )

        app = _mk_app(provider="emb1", embedder=embedder, index=index)
        client = TestClient(app)

        body = schemas.SearchRequest(
            query="what is pain",
            top_k=2,
            filters=None,
        )

        res = client.post("/v1/search", json=body.model_dump(mode="json"))
        assert res.status_code == 200

        payload = res.json()
        assert payload.get("request_id")
        assert payload.get("effective_embedding_model_id") == "emb1"
        assert payload.get("next_cursor") is None

        hits = payload.get("hits") or []
        assert len(hits) == 2

        h0 = hits[0]
        assert h0["doc_id"] == "d1"
        assert h0["doi"] == "10.1/x"
        assert h0["title"] == "T"
        assert h0["year"] == 2020
        assert h0["journal"] == "J"
        assert h0["authors"] == ["Alice", "Bob"]
        assert h0["chunk_ids"] == ["c1", "c2"]
        assert h0["content_text"] == "hello123"

        h1 = hits[1]
        assert h1["doc_id"] == "d2"
        assert h1["doi"] == "10.2/y"
        assert h1["chunk_ids"] == ["c3"]
        assert h1["content_text"] == "abcd"

        ops = [c["op"] for c in index.calls]
        assert ops == ["search", "fetch_by_doc_ids"]

    def _any_disease(self) -> schemas.Disease:
        members = list(schemas.Disease)  # type: ignore[arg-type]
        assert members, "schemas.Disease has no members"
        return members[0]

    def _any_source_type(self) -> schemas.SourceType:
        members = list(schemas.SourceType)  # type: ignore[arg-type]
        assert members, "schemas.SourceType has no members"
        return members[0]

    def test_given_filters_when_search_then_passes_qdrant_filter_to_index(self) -> None:
        embedder = _FakeEmbedder(vectors=[[0.1, 0.2]])
        index = _FakeVectorIndex(search_hits=[], fetch_hits=[], has_fetch=True)

        app = _mk_app(provider="emb1", embedder=embedder, index=index)
        client = TestClient(app)

        f = schemas.SearchFilters(
            disease=self._any_disease(),
            source_type=self._any_source_type(),
            year_min=2019,
            year_max=2020,
        )
        body = schemas.SearchRequest(query="q", top_k=1, filters=f)

        res = client.post("/v1/search", json=body.model_dump(mode="json"))
        assert res.status_code == 200

        call = index.calls[0]
        qfilter = call["qfilter"]
        assert isinstance(qfilter, Filter)
        assert qfilter.must and len(qfilter.must) == 3

    def test_given_empty_query_when_search_then_400_error_response(self) -> None:
        embedder = _FakeEmbedder(vectors=[[0.1, 0.2]])
        index = _FakeVectorIndex(search_hits=[], fetch_hits=[], has_fetch=True)

        app = _mk_app(provider="emb1", embedder=embedder, index=index)
        client = TestClient(app)

        body = schemas.SearchRequest(query="   ", top_k=1, filters=None)

        res = client.post("/v1/search", json=body.model_dump(mode="json"))
        assert res.status_code == 400

        payload = res.json()
        assert payload.get("request_id")
        assert payload.get("error") in ("validation_error", "system_error")
        assert payload.get("message")

    def test_given_embedder_returns_empty_vectors_when_search_then_500_error_response(self) -> None:
        embedder = _FakeEmbedder(vectors=[])
        index = _FakeVectorIndex(search_hits=[], fetch_hits=[], has_fetch=True)

        app = _mk_app(provider="emb1", embedder=embedder, index=index)
        client = TestClient(app)

        body = schemas.SearchRequest(query="q", top_k=1, filters=None)

        res = client.post("/v1/search", json=body.model_dump(mode="json"))
        assert res.status_code == 500

        payload = res.json()
        assert payload.get("request_id")
        assert payload.get("message")

    def test_given_missing_embedder_when_search_then_500_error_response(self) -> None:
        index = _FakeVectorIndex(search_hits=[], fetch_hits=[], has_fetch=True)
        app = _mk_app(provider="emb1", embedder=None, index=index)
        client = TestClient(app)

        body = schemas.SearchRequest(query="q", top_k=1, filters=None)

        res = client.post("/v1/search", json=body.model_dump(mode="json"))
        assert res.status_code == 500

    def test_given_missing_index_when_search_then_500_error_response(self) -> None:
        embedder = _FakeEmbedder(vectors=[[0.1, 0.2]])
        app = _mk_app(provider="emb1", embedder=embedder, index=None)
        client = TestClient(app)

        body = schemas.SearchRequest(query="q", top_k=1, filters=None)

        res = client.post("/v1/search", json=body.model_dump(mode="json"))
        assert res.status_code == 500

    def test_given_index_without_fetch_by_doc_ids_when_search_then_still_returns_hits(self) -> None:
        class _IndexNoFetch:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            async def search(
                    self,
                    *,
                    embedding_model_id: str,
                    query_vector: Sequence[float],
                    top_k: int,
                    qfilter: object | None,
            ) -> list[Any]:
                self.calls.append({"op": "search"})
                return [_raw_hit(doc_id="d1", score=0.9, extra={"doi": "10.1/x"})]

        embedder = _FakeEmbedder(vectors=[[0.1, 0.2]])
        index = _IndexNoFetch()

        app = _mk_app(provider="emb1", embedder=embedder, index=index)
        client = TestClient(app)

        body = schemas.SearchRequest(query="q", top_k=1, filters=None)

        res = client.post("/v1/search", json=body.model_dump(mode="json"))
        assert res.status_code == 200

        payload = res.json()
        hits = payload.get("hits") or []
        assert len(hits) == 1
        assert hits[0]["doc_id"] == "d1"
        assert hits[0]["content_text"] is None

    def test_given_too_many_requests_error_when_search_then_sets_retry_after_header(self) -> None:
        class _EmbedderTooMany:
            async def embed_texts(self, *, model_id: str, texts: Sequence[str]) -> list[list[float]]:
                raise SystemError(
                    code="too_many_requests",
                    message="rate limited",
                    details={"retry_after_seconds": 12},
                    retryable=True,
                )

        embedder = _EmbedderTooMany()
        index = _FakeVectorIndex(search_hits=[], fetch_hits=[], has_fetch=True)
        app = _mk_app(provider="emb1", embedder=embedder, index=index)
        client = TestClient(app)

        body = schemas.SearchRequest(query="q", top_k=1, filters=None)

        res = client.post("/v1/search", json=body.model_dump(mode="json"))
        assert res.status_code == 429
        assert res.headers.get("Retry-After") == "12"
