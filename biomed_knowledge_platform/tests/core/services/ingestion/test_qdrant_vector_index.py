# tests/core/services/ingestion/test_qdrant_vector_index.py
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import threading
import pytest
from qdrant_client.http.models import Distance

from biomed_platform.core.domains.ingestion import VectorPoint
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.services.ingestion.qdrant_vector_index import (
    QdrantVectorIndex,
    parse_distance,
)


@dataclass
class _Call:
    name: str
    kwargs: dict[str, Any]


class _FakeQdrantClient:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[_Call] = []
        self.existing_collections: list[str] = []
        self.fail_get: Exception | None = None
        self.fail_create: Exception | None = None
        self.fail_upsert: Exception | None = None

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


def _vp(*, point_id: str = "p1") -> VectorPoint:
    return VectorPoint(
        point_id=point_id,
        vector=[0.1, 0.2],
        payload={"k": "v", "n": 1},
    )


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
        client = _FakeQdrantClient()
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        got = idx._collection_name(embedding_model_id="a/b:c")

        assert got == "docs_a_b_c"

    def test_given_blank_prefix_when_collection_name_then_defaults_to_docs(self) -> None:
        client = _FakeQdrantClient()
        idx = QdrantVectorIndex(client=client, collection_name_prefix="   ", distance=Distance.COSINE)

        got = idx._collection_name(embedding_model_id="m1")

        assert got == "docs_m1"


@pytest.mark.asyncio
class TestQdrantVectorIndexEnsureCollection:
    async def test_given_vector_size_non_positive_when_ensure_collection_then_raises(self) -> None:
        client = _FakeQdrantClient()
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        with pytest.raises(SystemError) as exc:
            await idx.ensure_collection(embedding_model_id="m1", vector_size=0)

        assert exc.value.code == "invalid_vector_size"
        assert exc.value.retryable is False

    async def test_given_get_collections_raises_when_ensure_collection_then_maps_to_unavailable(self) -> None:
        client = _FakeQdrantClient()
        client.fail_get = RuntimeError("down")
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        with pytest.raises(SystemError) as exc:
            await idx.ensure_collection(embedding_model_id="m1", vector_size=4)

        assert exc.value.code == "qdrant_unavailable"
        assert exc.value.retryable is True
        assert exc.value.details and exc.value.details.get("operation") == "get_collections"

    async def test_given_collection_exists_when_ensure_collection_then_does_not_create(self) -> None:
        client = _FakeQdrantClient()
        client.existing_collections = ["docs_m1"]
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        await idx.ensure_collection(embedding_model_id="m1", vector_size=4)

        names = [c.name for c in client.calls]
        assert names == ["get_collections"]

    async def test_given_collection_missing_when_ensure_collection_then_creates_collection(self) -> None:
        client = _FakeQdrantClient()
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.DOT)

        await idx.ensure_collection(embedding_model_id="m1", vector_size=7)

        assert [c.name for c in client.calls] == ["get_collections", "create_collection"]

        create_call = client.calls[1]
        assert create_call.kwargs["collection_name"] == "docs_m1"

        vectors_cfg = create_call.kwargs["vectors_config"]
        assert getattr(vectors_cfg, "size") == 7
        assert getattr(vectors_cfg, "distance") == Distance.DOT

    async def test_given_create_collection_raises_when_ensure_collection_then_maps_to_system_error(self) -> None:
        client = _FakeQdrantClient()
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
        client = _FakeQdrantClient()
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        await idx.upsert(embedding_model_id="m1", points=[])

        assert client.calls == []

    async def test_given_points_when_upsert_then_converts_and_calls_client(self) -> None:
        client = _FakeQdrantClient()
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
        client = _FakeQdrantClient()
        client.fail_upsert = RuntimeError("down")
        idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

        with pytest.raises(SystemError) as exc:
            await idx.upsert(embedding_model_id="m1", points=[_vp(point_id="p1")])

        assert exc.value.code == "qdrant_upsert_failed"
        assert exc.value.retryable is True
        assert exc.value.details and exc.value.details.get("collection") == "docs_m1"
        assert exc.value.details and exc.value.details.get("points") == 1
