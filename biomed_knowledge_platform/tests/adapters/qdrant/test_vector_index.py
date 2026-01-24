from __future__ import annotations

import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from types import SimpleNamespace
from unittest.mock import MagicMock

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, Filter

from biomed_platform.adapters.qdrant.vector_index import QdrantVectorIndex, parse_distance
from biomed_platform.core.domains.ingestion import VectorPoint
from biomed_platform.core.errors.errors import SystemError


def _unexpected_response_with_status(status_code: int) -> UnexpectedResponse:
    # Create instance without calling __init__ to avoid signature drift.
    exc = UnexpectedResponse.__new__(UnexpectedResponse)  # type: ignore[misc]
    exc.status_code = status_code  # qdrant_client uses this in some versions
    exc.response = SimpleNamespace(status_code=status_code)  # your code may read response.status_code
    return exc



@pytest.mark.anyio
async def test_parse_distance_and_validate_vector_size() -> None:
    # Given, valid values
    assert parse_distance("cosine") == Distance.COSINE
    assert parse_distance("DOT") == Distance.DOT

    # When, invalid value
    with pytest.raises(SystemError) as exc:
        parse_distance("bad")

    # Then
    assert exc.value.code == "invalid_qdrant_distance"


@pytest.mark.anyio
async def test_ensure_collection_creates_when_missing_and_skips_when_exists(monkeypatch) -> None:
    # Given, a qdrant client with no collections
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(collections=[])
    client.create_collection = MagicMock()

    idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr(
        "biomed_platform.adapters.qdrant.vector_index.asyncio.to_thread",
        _to_thread,
        raising=True,
    )

    # When, ensuring the collection
    await idx.ensure_collection(embedding_model_id="m", vector_size=2)

    # Then, create_collection called
    assert client.create_collection.call_count == 1

    # Given, collection now exists
    client.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name="docs_m")])

    # When
    await idx.ensure_collection(embedding_model_id="m", vector_size=2)

    # Then, create not called again
    assert client.create_collection.call_count == 1


@pytest.mark.anyio
async def test_upsert_exists_search_and_filter_paths(monkeypatch) -> None:
    # Given
    client = MagicMock(
        spec_set=[
            "upsert",
            "count",
            "search",
        ]
    )
    client.upsert = MagicMock()
    client.count.return_value = SimpleNamespace(count=1)
    client.search = MagicMock()

    idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr(
        "biomed_platform.adapters.qdrant.vector_index.asyncio.to_thread",
        _to_thread,
        raising=True,
    )

    # When, upserting points
    await idx.upsert(
        embedding_model_id="m",
        points=[VectorPoint(point_id="p1", vector=[0.1], payload={"doc_id": "d"})],
    )

    # Then
    assert client.upsert.call_count == 1

    # When, exists
    ok = await idx.exists(embedding_model_id="m", doc_id="d")

    # Then
    assert ok is True

    # Given, exists on missing collection returns false
    client.count.side_effect = _unexpected_response_with_status(404)

    # When
    ok2 = await idx.exists(embedding_model_id="m", doc_id="d")

    # Then
    assert ok2 is False

    # Given, search uses client.search and returns points
    client.count.side_effect = None
    client.search.return_value = [SimpleNamespace(id="x", score=0.5, payload={"doc_id": "d"})]

    # When
    hits = await idx.search(
        embedding_model_id="m",
        query_vector=[0.1],
        top_k=1,
        qfilter={"disease": "x", "year_min": 2020},
    )

    # Then
    assert len(hits) == 1
    assert hits[0].point_id == "x"

    # Given, search fails with incompatible client
    bad_client = MagicMock(spec_set=["upsert", "count"])
    bad_client.upsert = MagicMock()
    bad_client.count = MagicMock()

    idx_bad = QdrantVectorIndex(client=bad_client, collection_name_prefix="docs", distance=Distance.COSINE)

    monkeypatch.setattr(
        "biomed_platform.adapters.qdrant.vector_index.asyncio.to_thread",
        _to_thread,
        raising=True,
    )

    # When
    with pytest.raises(SystemError) as exc:
        await idx_bad.search(
            embedding_model_id="m",
            query_vector=[0.1],
            top_k=1,
            qfilter=Filter(),
        )

    # Then
    assert exc.value.code == "qdrant_client_incompatible"


@pytest.mark.anyio
async def test_search_chunks_maps_candidate_fields_and_applies_filters(monkeypatch) -> None:
    # Given
    client = MagicMock(
        spec_set=[
            "search",
        ]
    )
    client.search = MagicMock()

    client.search.return_value = [
        SimpleNamespace(
            id="c1",
            score=0.9,
            payload={
                "doc_id": "d1",
                "doi_original": "10.1/abc",
                "title": "t1",
                "year": 2021,
                "source_type": "pubmed_abstract",
                "text": "chunk text",
            },
        )
    ]

    idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr(
        "biomed_platform.adapters.qdrant.vector_index.asyncio.to_thread",
        _to_thread,
        raising=True,
    )

    # When
    out = await idx.search_chunks(
        embedding_model_id="m",
        query_vector=[0.1],
        top_k=10,
        qfilter={"doi": "10.1/abc", "year": 2021, "source_type": "pubmed_abstract"},
    )

    # Then
    assert len(out) == 1
    assert out[0].chunk_id == "c1"
    assert out[0].doc_id == "d1"
    assert out[0].doi == "10.1/abc"
    assert out[0].title == "t1"
    assert out[0].year == 2021
    assert out[0].source_type is not None
    assert out[0].source_type.value == "pubmed_abstract"
    assert out[0].chunk_text == "chunk text"

    # And filter was passed to qdrant
    args, kwargs = client.search.call_args
    filt = kwargs.get("query_filter")
    assert isinstance(filt, Filter)
    keys = {getattr(c, "key", None) for c in (filt.must or [])}
    assert "doi_original" in keys
    assert "year" in keys
    assert "source_type" in keys


@pytest.mark.anyio
async def test_fetch_chunks_by_ids_uses_retrieve_and_maps(monkeypatch) -> None:
    # Given
    client = MagicMock(spec_set=["retrieve"])
    client.retrieve = MagicMock()
    client.retrieve.return_value = [
        SimpleNamespace(
            id="c1",
            payload={
                "doc_id": "d1",
                "doi_original": "10.1/abc",
                "title": "t1",
                "year": 2021,
                "source_type": "pubmed_abstract",
                "text": "chunk text",
            },
        )
    ]

    idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr(
        "biomed_platform.adapters.qdrant.vector_index.asyncio.to_thread",
        _to_thread,
        raising=True,
    )

    # When
    out = await idx.fetch_chunks_by_ids(embedding_model_id="m", chunk_ids=["c1"])

    # Then
    assert len(out) == 1
    assert out[0].chunk_id == "c1"
    assert out[0].chunk_text == "chunk text"
    assert client.retrieve.call_count == 1


@pytest.mark.anyio
async def test_collection_name_sanitizes_and_prefix_fallback(monkeypatch) -> None:
    # Given
    client = MagicMock()
    idx = QdrantVectorIndex(client=client, collection_name_prefix="  ", distance=Distance.COSINE)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr("biomed_platform.adapters.qdrant.vector_index.asyncio.to_thread", _to_thread, raising=True)

    # When
    name = idx._collection_name(embedding_model_id="a/b:c")

    # Then
    assert name == "docs_a_b_c"


@pytest.mark.anyio
async def test_validate_vector_size_raises() -> None:
    # Given
    idx = QdrantVectorIndex(client=MagicMock(), collection_name_prefix="docs", distance=Distance.COSINE)

    # When, Then
    with pytest.raises(SystemError) as exc:
        idx._validate_vector_size(embedding_model_id="m", vector_size=0)

    assert exc.value.code == "invalid_vector_size"


@pytest.mark.anyio
async def test_collection_lock_is_cached_per_collection() -> None:
    # Given
    idx = QdrantVectorIndex(client=MagicMock(), collection_name_prefix="docs", distance=Distance.COSINE)

    # When
    l1 = await idx._collection_lock(name="c1")
    l2 = await idx._collection_lock(name="c1")
    l3 = await idx._collection_lock(name="c2")

    # Then
    assert l1 is l2
    assert l1 is not l3


@pytest.mark.anyio
async def test_get_collections_wraps_exceptions(monkeypatch) -> None:
    # Given
    client = MagicMock()
    client.get_collections.side_effect = RuntimeError("down")
    idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr("biomed_platform.adapters.qdrant.vector_index.asyncio.to_thread", _to_thread, raising=True)

    # When, Then
    with pytest.raises(SystemError) as exc:
        await idx._get_collections(name="docs_m")

    assert exc.value.code == "qdrant_unavailable"


@pytest.mark.anyio
async def test_create_collection_handles_conflict_and_other_errors(monkeypatch) -> None:
    # Given
    client = MagicMock()
    idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr("biomed_platform.adapters.qdrant.vector_index.asyncio.to_thread", _to_thread, raising=True)

    # When, conflict 409, Then no error
    client.create_collection.side_effect = _unexpected_response_with_status(409)
    await idx._create_collection(name="docs_m", vector_size=2)

    # When, unexpected response other than 409, Then SystemError
    client.create_collection.side_effect = _unexpected_response_with_status(500)
    with pytest.raises(SystemError) as exc:
        await idx._create_collection(name="docs_m", vector_size=2)
    assert exc.value.code == "qdrant_create_collection_failed"

    # When, generic exception, Then SystemError
    client.create_collection.side_effect = RuntimeError("boom")
    with pytest.raises(SystemError) as exc2:
        await idx._create_collection(name="docs_m", vector_size=2)
    assert exc2.value.code == "qdrant_create_collection_failed"


@pytest.mark.anyio
async def test_ensure_collection_skips_when_exists_and_creates_otherwise(monkeypatch) -> None:
    # Given
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name="docs_m")])

    idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr("biomed_platform.adapters.qdrant.vector_index.asyncio.to_thread", _to_thread, raising=True)

    # When, exists
    await idx.ensure_collection(embedding_model_id="m", vector_size=3)

    # Then, no create
    assert client.create_collection.call_count == 0

    # Given, missing
    client.get_collections.return_value = SimpleNamespace(collections=[])

    # When
    await idx.ensure_collection(embedding_model_id="m", vector_size=3)

    # Then
    assert client.create_collection.call_count == 1


@pytest.mark.anyio
async def test_upsert_no_points_is_noop_and_error_is_wrapped(monkeypatch) -> None:
    # Given
    client = MagicMock()
    idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr("biomed_platform.adapters.qdrant.vector_index.asyncio.to_thread", _to_thread, raising=True)

    # When, empty points
    await idx.upsert(embedding_model_id="m", points=[])

    # Then
    assert client.upsert.call_count == 0

    # Given, failing upsert
    client.upsert.side_effect = RuntimeError("fail")

    # When, Then
    with pytest.raises(SystemError) as exc:
        await idx.upsert(
            embedding_model_id="m",
            points=[VectorPoint(point_id="p1", vector=[0.1], payload={"doc_id": "d"})],
        )

    assert exc.value.code == "qdrant_upsert_failed"


@pytest.mark.anyio
async def test_exists_empty_doc_is_false_and_non404_errors_are_wrapped(monkeypatch) -> None:
    # Given
    client = MagicMock()
    idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr("biomed_platform.adapters.qdrant.vector_index.asyncio.to_thread", _to_thread, raising=True)

    # When
    assert await idx.exists(embedding_model_id="m", doc_id="") is False

    # Given, non 404 unexpected response
    client.count.side_effect = _unexpected_response_with_status(500)

    # When, Then
    with pytest.raises(SystemError) as exc:
        await idx.exists(embedding_model_id="m", doc_id="d1")

    assert exc.value.code == "qdrant_exists_failed"

    # Given, generic exception
    client.count.side_effect = RuntimeError("x")

    with pytest.raises(SystemError) as exc2:
        await idx.exists(embedding_model_id="m", doc_id="d1")

    assert exc2.value.code == "qdrant_exists_failed"


@pytest.mark.anyio
async def test_search_top_k_zero_returns_empty(monkeypatch) -> None:
    # Given
    idx = QdrantVectorIndex(client=MagicMock(), collection_name_prefix="docs", distance=Distance.COSINE)

    # When
    out = await idx.search(embedding_model_id="m", query_vector=[0.1], top_k=0, qfilter=None)

    # Then
    assert out == []


@pytest.mark.anyio
async def test_search_uses_query_points_when_available_and_handles_result_shapes(monkeypatch) -> None:
    # Given
    client = MagicMock(spec_set=["query_points"])
    client.query_points = MagicMock(
        return_value=SimpleNamespace(
            points=[
                SimpleNamespace(id="p1", score=0.7, payload={"doc_id": "d1"}),
                SimpleNamespace(id="p2", score=None, payload=None),
            ]
        )
    )

    idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr("biomed_platform.adapters.qdrant.vector_index.asyncio.to_thread", _to_thread, raising=True)

    # When
    hits = await idx.search(
        embedding_model_id="m",
        query_vector=[0.1, 0.2],
        top_k=2,
        qfilter={"disease": "x", "source_type": "full_text", "year_min": 2000, "year_max": 2020},
    )

    # Then
    assert [h.point_id for h in hits] == ["p1", "p2"]
    assert hits[0].payload["doc_id"] == "d1"
    assert hits[1].score == 0.0


@pytest.mark.anyio
async def test_search_falls_back_to_search_method_and_incompatible_client(monkeypatch) -> None:
    # Given, legacy client.search path
    client = MagicMock(spec_set=["search"])
    client.search = MagicMock(return_value=[SimpleNamespace(id="x", score=0.5, payload={"doc_id": "d"})])

    idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr("biomed_platform.adapters.qdrant.vector_index.asyncio.to_thread", _to_thread, raising=True)

    # When
    hits = await idx.search(embedding_model_id="m", query_vector=[0.1], top_k=1, qfilter=Filter())

    # Then
    assert len(hits) == 1
    assert hits[0].point_id == "x"

    # Given, incompatible client with neither method
    bad_client = MagicMock(spec_set=[])
    idx_bad = QdrantVectorIndex(client=bad_client, collection_name_prefix="docs", distance=Distance.COSINE)

    # When, Then
    with pytest.raises(SystemError) as exc:
        await idx_bad.search(embedding_model_id="m", query_vector=[0.1], top_k=1, qfilter=None)

    assert exc.value.code == "qdrant_client_incompatible"


@pytest.mark.anyio
async def test_search_handles_unexpected_response_and_generic_errors(monkeypatch) -> None:
    # Given
    client = MagicMock(spec_set=["search"])
    idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr("biomed_platform.adapters.qdrant.vector_index.asyncio.to_thread", _to_thread, raising=True)

    # When, 404, Then empty
    client.search.side_effect = _unexpected_response_with_status(404)
    out = await idx.search(embedding_model_id="m", query_vector=[0.1], top_k=3, qfilter=None)
    assert out == []

    # When, 500, Then wrapped
    client.search.side_effect = _unexpected_response_with_status(500)
    with pytest.raises(SystemError) as exc:
        await idx.search(embedding_model_id="m", query_vector=[0.1], top_k=3, qfilter=None)
    assert exc.value.code == "qdrant_search_failed"

    # When, generic exception, Then wrapped
    client.search.side_effect = RuntimeError("x")
    with pytest.raises(SystemError) as exc2:
        await idx.search(embedding_model_id="m", query_vector=[0.1], top_k=3, qfilter=None)
    assert exc2.value.code == "qdrant_search_failed"


@pytest.mark.anyio
async def test_fetch_by_doc_ids_empty_short_circuit() -> None:
    # Given
    idx = QdrantVectorIndex(client=MagicMock(), collection_name_prefix="docs", distance=Distance.COSINE)

    # When
    out = await idx.fetch_by_doc_ids(embedding_model_id="m", doc_ids=["", ""], base_filter=None)

    # Then
    assert out == []


@pytest.mark.anyio
async def test_fetch_by_doc_ids_scroll_paginates_and_merges_base_filter(monkeypatch) -> None:
    # Given
    client = MagicMock(spec_set=["scroll"])
    idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

    pages = [
        ([SimpleNamespace(id="c1", payload={"doc_id": "d1", "chunk_index": 0})], "off1"),
        ([SimpleNamespace(id="c2", payload={"doc_id": "d1", "chunk_index": 1})], None),
    ]
    client.scroll.side_effect = list(pages)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr("biomed_platform.adapters.qdrant.vector_index.asyncio.to_thread", _to_thread, raising=True)

    # When
    hits = await idx.fetch_by_doc_ids(
        embedding_model_id="m",
        doc_ids=["d1", "d2"],
        base_filter={"disease": "thrombosis"},
        limit=10,
    )

    # Then
    assert [h.point_id for h in hits] == ["c1", "c2"]
    assert hits[0].payload["doc_id"] == "d1"
    assert client.scroll.call_count == 2


@pytest.mark.anyio
async def test_fetch_by_doc_ids_handles_404_and_errors(monkeypatch) -> None:
    # Given
    client = MagicMock(spec_set=["scroll"])
    idx = QdrantVectorIndex(client=client, collection_name_prefix="docs", distance=Distance.COSINE)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr("biomed_platform.adapters.qdrant.vector_index.asyncio.to_thread", _to_thread, raising=True)

    # When, 404 returns empty
    client.scroll.side_effect = _unexpected_response_with_status(404)
    out = await idx.fetch_by_doc_ids(embedding_model_id="m", doc_ids=["d1"], base_filter=None, limit=10)
    assert out == []

    # When, non 404 is wrapped
    client.scroll.side_effect = _unexpected_response_with_status(500)
    with pytest.raises(SystemError) as exc:
        await idx.fetch_by_doc_ids(embedding_model_id="m", doc_ids=["d1"], base_filter=None, limit=10)
    assert exc.value.code == "qdrant_fetch_failed"

    # When, generic exception is wrapped
    client.scroll.side_effect = RuntimeError("x")
    with pytest.raises(SystemError) as exc2:
        await idx.fetch_by_doc_ids(embedding_model_id="m", doc_ids=["d1"], base_filter=None, limit=10)
    assert exc2.value.code == "qdrant_fetch_failed"