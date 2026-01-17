# tests/core/services/ingestion/test_qdrant_vector_index.py
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import threading
import pytest
from qdrant_client.http.models import Distance, Filter

from biomed_platform.core.domains.ingestion import VectorPoint
from biomed_platform.core.domains.retrieval import VectorSearchHit
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.services.ingestion import qdrant_vector_index as mod
from biomed_platform.core.services.ingestion.qdrant_vector_index import QdrantVectorIndex, parse_distance


@dataclass
class _Call:
    name: str
    kwargs: dict[str, Any]


class _FakeUnexpectedResponse(Exception):
    def __init__(self, *, status_code: int) -> None:
        super().__init__(f"unexpected_response_{status_code}")
        self.response = SimpleNamespace(status_code=status_code)


class _FakeQdrantClientBase:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[_Call] = []

        self.existing_collections: list[str] = []
        self.fail_get: Exception | None = None
        self.fail_create: Exception | None = None
        self.fail_upsert: Exception | None = None
        self.fail_count: Exception | None = None
        self.fail_search: Exception | None = None
        self.fail_scroll: Exception | None = None

        self.count_value: int = 0
        self.search_points: list[Any] = []

        self.scroll_pages: list[tuple[list[Any], Any]] = []
        self._scroll_i: int = 0

    def _record(self, name: str, kwargs: dict[str, Any]) -> None:
        with self._lock:
            self.calls.append(_Call(name, kwargs))

    def get_collections(self) -> Any:
        self._record("get_collections", {})
        if self.fail_get is not None:
            raise self.fail_get
        collections = [SimpleNamespace(name=n) for n in self.existing_collections]
        return SimpleNamespace(collections=collections)

    def create_collection(self, **kwargs: Any) -> None:
        self._record("create_collection", dict(kwargs))
        if self.fail_create is not None:
            raise self.fail_create

    def upsert(self, **kwargs: Any) -> None:
        self._record("upsert", dict(kwargs))
        if self.fail_upsert is not None:
            raise self.fail_upsert

    def count(self, **kwargs: Any) -> Any:
        self._record("count", dict(kwargs))
        if self.fail_count is not None:
            raise self.fail_count
        return SimpleNamespace(count=self.count_value)

    def scroll(self, **kwargs: Any) -> Any:
        self._record("scroll", dict(kwargs))
        if self.fail_scroll is not None:
            raise self.fail_scroll

        if self._scroll_i >= len(self.scroll_pages):
            return ([], None)

        page, next_offset = self.scroll_pages[self._scroll_i]
        self._scroll_i += 1
        return (page, next_offset)


class _FakeQdrantClientWithQueryPoints(_FakeQdrantClientBase):
    def query_points(self, **kwargs: Any) -> Any:
        self._record("query_points", dict(kwargs))
        if self.fail_search is not None:
            raise self.fail_search
        return SimpleNamespace(points=list(self.search_points))


class _FakeQdrantClientWithSearch(_FakeQdrantClientBase):
    def search(self, **kwargs: Any) -> Any:
        self._record("search", dict(kwargs))
        if self.fail_search is not None:
            raise self.fail_search
        return list(self.search_points)


class _FakeQdrantClientIncompatible(_FakeQdrantClientBase):
    pass


def _vp(*, point_id: str = "p1") -> VectorPoint:
    return VectorPoint(
        point_id=point_id,
        vector=[0.1, 0.2],
        payload={"k": "v", "n": 1},
    )


@pytest.fixture(autouse=True)
def _patch_unexpected_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "UnexpectedResponse", _FakeUnexpectedResponse)


class TestParseDistance:
    def test_given_cosine_when_parse_then_returns_distance_cosine(self) -> None:
        assert parse_distance("cosine") == Distance.COSINE
        assert parse_distance(" COSINE ") == Distance.COSINE

    def test_given_dot_when_parse_then_returns_distance_dot(self) -> None:
        assert parse_distance("dot") == Distance.DOT
        assert parse_distance(" DOT ") == Distance.DOT

    def test_given_euclid_when_parse_then_returns_distance_euclid(self) -> None:
        assert parse_distance("euclid") == Distance.EUCLID
        assert parse_distance(" EUCLID ") == Distance.EUCLID

    def test_given_invalid_value_when_parse_then_raises_system_error(self) -> None:
        with pytest.raises(SystemError) as exc:
            parse_distance("manhattan")

        assert exc.value.code == "invalid_qdrant_distance"
        assert exc.value.retryable is False
        assert exc.value.details and exc.value.details.get("distance") == "manhattan"


class TestQdrantVectorIndexCollectionName:
    def test_given_embedding_model_id_with_separators_when_collection_name_then_is_sanitized(self) -> None:
        client = _FakeQdrantClientBase()
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        got = idx._collection_name(embedding_model_id="a/b:c")

        assert got == "docs_a_b_c"

    def test_given_blank_prefix_when_collection_name_then_defaults_to_docs(self) -> None:
        client = _FakeQdrantClientBase()
        idx = QdrantVectorIndex(client=client, collection_name_prefix="   ", distance=Distance.COSINE)

        got = idx._collection_name(embedding_model_id="m1")

        assert got == "docs_m1"


@pytest.mark.asyncio
class TestQdrantVectorIndexEnsureCollection:
    async def test_given_vector_size_non_positive_when_ensure_collection_then_raises(self) -> None:
        client = _FakeQdrantClientBase()
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        with pytest.raises(SystemError) as exc:
            await idx.ensure_collection(embedding_model_id="m1", vector_size=0)

        assert exc.value.code == "invalid_vector_size"
        assert exc.value.retryable is False

    async def test_given_get_collections_raises_when_ensure_collection_then_maps_to_unavailable(self) -> None:
        client = _FakeQdrantClientBase()
        client.fail_get = RuntimeError("down")
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        with pytest.raises(SystemError) as exc:
            await idx.ensure_collection(embedding_model_id="m1", vector_size=4)

        assert exc.value.code == "qdrant_unavailable"
        assert exc.value.retryable is True
        assert exc.value.details and exc.value.details.get("operation") == "get_collections"

    async def test_given_collection_exists_when_ensure_collection_then_does_not_create(self) -> None:
        client = _FakeQdrantClientBase()
        client.existing_collections = ["docs_m1"]
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        await idx.ensure_collection(embedding_model_id="m1", vector_size=4)

        names = [c.name for c in client.calls]
        assert names == ["get_collections"]

    async def test_given_collection_missing_when_ensure_collection_then_creates_collection(self) -> None:
        client = _FakeQdrantClientBase()
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.DOT)

        await idx.ensure_collection(embedding_model_id="m1", vector_size=7)

        assert [c.name for c in client.calls] == ["get_collections", "create_collection"]

        create_call = client.calls[1]
        assert create_call.kwargs["collection_name"] == "docs_m1"

        vectors_cfg = create_call.kwargs["vectors_config"]
        assert getattr(vectors_cfg, "size") == 7
        assert getattr(vectors_cfg, "distance") == Distance.DOT

    async def test_given_create_collection_conflict_when_ensure_collection_then_succeeds(self) -> None:
        client = _FakeQdrantClientBase()
        client.fail_create = _FakeUnexpectedResponse(status_code=409)
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        await idx.ensure_collection(embedding_model_id="m1", vector_size=3)

        assert [c.name for c in client.calls] == ["get_collections", "create_collection"]

    async def test_given_create_collection_raises_when_ensure_collection_then_maps_to_system_error(self) -> None:
        client = _FakeQdrantClientBase()
        client.fail_create = RuntimeError("boom")
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        with pytest.raises(SystemError) as exc:
            await idx.ensure_collection(embedding_model_id="m1", vector_size=3)

        assert exc.value.code == "qdrant_create_collection_failed"
        assert exc.value.retryable is True
        assert exc.value.details and exc.value.details.get("collection") == "docs_m1"


@pytest.mark.asyncio
class TestQdrantVectorIndexUpsert:
    async def test_given_no_points_when_upsert_then_no_client_call(self) -> None:
        client = _FakeQdrantClientBase()
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        await idx.upsert(embedding_model_id="m1", points=[])

        assert client.calls == []

    async def test_given_points_when_upsert_then_converts_and_calls_client(self) -> None:
        client = _FakeQdrantClientBase()
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        points: Sequence[VectorPoint] = [_vp(point_id="p1"), _vp(point_id="p2")]

        await idx.upsert(embedding_model_id="m1", points=points)

        assert len(client.calls) == 1
        call = client.calls[0]
        assert call.name == "upsert"
        assert call.kwargs["collection_name"] == "docs_m1"
        assert call.kwargs["wait"] is True

        qpoints = call.kwargs["points"]
        assert len(qpoints) == 2

        assert qpoints[0].id == "p1"
        assert qpoints[0].vector == [0.1, 0.2]
        assert qpoints[0].payload == {"k": "v", "n": 1}

        assert qpoints[1].id == "p2"

    async def test_given_upsert_raises_when_upsert_then_maps_to_system_error(self) -> None:
        client = _FakeQdrantClientBase()
        client.fail_upsert = RuntimeError("down")
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        with pytest.raises(SystemError) as exc:
            await idx.upsert(embedding_model_id="m1", points=[_vp(point_id="p1")])

        assert exc.value.code == "qdrant_upsert_failed"
        assert exc.value.retryable is True
        assert exc.value.details and exc.value.details.get("collection") == "docs_m1"
        assert exc.value.details and exc.value.details.get("points") == 1


@pytest.mark.asyncio
class TestQdrantVectorIndexExists:
    async def test_given_blank_doc_id_when_exists_then_false(self) -> None:
        client = _FakeQdrantClientBase()
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        assert await idx.exists(embedding_model_id="m1", doc_id="") is False
        assert client.calls == []

    async def test_given_whitespace_doc_id_when_exists_then_calls_count_and_returns_false(self) -> None:
        client = _FakeQdrantClientBase()
        client.count_value = 0
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        assert await idx.exists(embedding_model_id="m1", doc_id="   ") is False

        assert [c.name for c in client.calls] == ["count"]
        call = client.calls[0]
        assert call.kwargs["collection_name"] == "docs_m1"


    async def test_given_count_unexpected_response_non_404_when_exists_then_system_error(self) -> None:
        client = _FakeQdrantClientBase()
        client.fail_count = _FakeUnexpectedResponse(status_code=500)
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        with pytest.raises(SystemError) as exc:
            await idx.exists(embedding_model_id="m1", doc_id="d1")

        assert exc.value.code == "qdrant_exists_failed"
        assert exc.value.retryable is True
        assert exc.value.details and exc.value.details.get("collection") == "docs_m1"
        assert exc.value.details and exc.value.details.get("doc_id") == "d1"
        assert exc.value.details and exc.value.details.get("status_code") == 500

    async def test_given_count_other_exception_when_exists_then_system_error(self) -> None:
        client = _FakeQdrantClientBase()
        client.fail_count = RuntimeError("boom")
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        with pytest.raises(SystemError) as exc:
            await idx.exists(embedding_model_id="m1", doc_id="d1")

        assert exc.value.code == "qdrant_exists_failed"
        assert exc.value.retryable is True
        assert exc.value.details and exc.value.details.get("collection") == "docs_m1"
        assert exc.value.details and exc.value.details.get("doc_id") == "d1"


@pytest.mark.asyncio
class TestQdrantVectorIndexSearch:
    async def test_given_top_k_non_positive_when_search_then_empty(self) -> None:
        client = _FakeQdrantClientWithQueryPoints()
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        got = await idx.search(
            embedding_model_id="m1",
            query_vector=[0.1, 0.2],
            top_k=0,
            qfilter=None,
        )

        assert got == []
        assert client.calls == []

    async def test_given_query_points_available_when_search_then_uses_query_points(self) -> None:
        client = _FakeQdrantClientWithQueryPoints()
        client.search_points = [
            SimpleNamespace(id="p1", score=0.9, payload={"doc_id": "d1"}),
            SimpleNamespace(id=2, score=0.1, payload=None),
        ]
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        got = await idx.search(
            embedding_model_id="m1",
            query_vector=[0.1, 0.2],
            top_k=2,
            qfilter=None,
        )

        assert [c.name for c in client.calls] == ["query_points"]
        assert got == [
            VectorSearchHit(point_id="p1", score=0.9, payload={"doc_id": "d1"}),
            VectorSearchHit(point_id="2", score=0.1, payload={}),
        ]

    async def test_given_only_search_available_when_search_then_uses_search(self) -> None:
        client = _FakeQdrantClientWithSearch()
        client.search_points = [
            SimpleNamespace(id="p1", score=1.0, payload={"k": "v"}),
        ]
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        got = await idx.search(
            embedding_model_id="m1",
            query_vector=[0.1, 0.2],
            top_k=3,
            qfilter=None,
        )

        assert [c.name for c in client.calls] == ["search"]
        assert got == [VectorSearchHit(point_id="p1", score=1.0, payload={"k": "v"})]

    async def test_given_incompatible_client_when_search_then_system_error(self) -> None:
        client = _FakeQdrantClientIncompatible()
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        with pytest.raises(SystemError) as exc:
            await idx.search(
                embedding_model_id="m1",
                query_vector=[0.1, 0.2],
                top_k=3,
                qfilter=None,
            )

        assert exc.value.code == "qdrant_client_incompatible"
        assert exc.value.retryable is False

    async def test_given_collection_missing_when_search_then_empty(self) -> None:
        client = _FakeQdrantClientWithQueryPoints()
        client.fail_search = _FakeUnexpectedResponse(status_code=404)
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        got = await idx.search(
            embedding_model_id="m1",
            query_vector=[0.1, 0.2],
            top_k=3,
            qfilter=None,
        )

        assert got == []

    async def test_given_search_unexpected_response_non_404_when_search_then_system_error(self) -> None:
        client = _FakeQdrantClientWithQueryPoints()
        client.fail_search = _FakeUnexpectedResponse(status_code=500)
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        with pytest.raises(SystemError) as exc:
            await idx.search(
                embedding_model_id="m1",
                query_vector=[0.1, 0.2],
                top_k=3,
                qfilter=None,
            )

        assert exc.value.code == "qdrant_search_failed"
        assert exc.value.retryable is True
        assert exc.value.details and exc.value.details.get("collection") == "docs_m1"
        assert exc.value.details and exc.value.details.get("status_code") == 500

    async def test_given_filter_object_not_filter_when_search_then_ignores_filter(self) -> None:
        client = _FakeQdrantClientWithQueryPoints()
        client.search_points = [SimpleNamespace(id="p1", score=0.5, payload={})]
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        got = await idx.search(
            embedding_model_id="m1",
            query_vector=[0.1, 0.2],
            top_k=1,
            qfilter={"not": "a_filter"},
        )

        assert got and got[0].point_id == "p1"
        call = client.calls[0]
        assert call.name == "query_points"
        assert call.kwargs.get("query_filter") is None

    async def test_given_filter_is_filter_when_search_then_passes_filter(self) -> None:
        client = _FakeQdrantClientWithQueryPoints()
        client.search_points = [SimpleNamespace(id="p1", score=0.5, payload={})]
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        f = Filter()
        await idx.search(
            embedding_model_id="m1",
            query_vector=[0.1, 0.2],
            top_k=1,
            qfilter=f,
        )

        call = client.calls[0]
        assert call.name == "query_points"
        assert call.kwargs.get("query_filter") is f


@pytest.mark.asyncio
class TestQdrantVectorIndexFetchByDocIds:
    async def test_given_all_doc_ids_empty_string_when_fetch_then_empty_and_no_calls(self) -> None:
        client = _FakeQdrantClientBase()
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        got = await idx.fetch_by_doc_ids(
            embedding_model_id="m1",
            doc_ids=["", ""],
            base_filter=None,
        )

        assert got == []
        assert client.calls == []

    async def test_given_whitespace_doc_id_when_fetch_then_calls_scroll_and_returns_empty(self) -> None:
        client = _FakeQdrantClientBase()
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        got = await idx.fetch_by_doc_ids(
            embedding_model_id="m1",
            doc_ids=["   "],
            base_filter=None,
        )

        assert got == []
        assert [c.name for c in client.calls] == ["scroll"]

