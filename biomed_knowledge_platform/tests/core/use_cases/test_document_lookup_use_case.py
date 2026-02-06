from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from biomed_platform.core.domains.retrieval import VectorSearchHit
from biomed_platform.core.errors.errors import BusinessError
from biomed_platform.core.use_cases.document_lookup import DocumentLookupUseCase


def _hit(*, point_id: str, payload: dict[str, object]) -> VectorSearchHit:
    return VectorSearchHit(point_id=point_id, score=0.0, payload=payload)


@pytest.mark.anyio
async def test_get_by_doi_returns_document_response() -> None:
    # Given
    vector_index = AsyncMock()
    vector_index.fetch_by_doc_ids = AsyncMock(
        return_value=[
            _hit(
                point_id="c1",
                payload={
                    "doc_id": "doc1",
                    "doi_original": "10.1000/xyz123",
                    "chunk_index": 1,
                    "text": "world",
                    "section": "Results",
                    "created_at": "2026-01-18T14:32:05Z",
                },
            ),
            _hit(
                point_id="c0",
                payload={
                    "doc_id": "doc1",
                    "doi_original": "10.1000/xyz123",
                    "chunk_index": 0,
                    "text": "hello",
                    "section": "Introduction",
                    "created_at": "2026-01-18T14:32:04Z",
                },
            ),
        ]
    )

    use_case = DocumentLookupUseCase(vector_index=vector_index)

    # When
    res = await use_case.get_by_doi(
        request_id="rid",
        embedding_model_id="m",
        doi="10.1000/xyz123",
    )

    # Then
    assert res.request_id == "rid"
    assert res.doi == "10.1000/xyz123"
    assert res.chunk_ids == ["c0", "c1"]
    assert [s.section for s in res.sections] == ["Introduction", "Results"]
    assert res.content_text == "hello\nworld"


@pytest.mark.anyio
async def test_get_by_doi_invalid_raises() -> None:
    use_case = DocumentLookupUseCase(vector_index=AsyncMock())

    with pytest.raises(BusinessError) as exc:
        await use_case.get_by_doi(
            request_id="rid",
            embedding_model_id="m",
            doi="not-a-doi",
        )

    assert exc.value.code == "validation_error"


@pytest.mark.anyio
async def test_get_by_doi_not_found_raises() -> None:
    vector_index = AsyncMock()
    vector_index.fetch_by_doc_ids = AsyncMock(return_value=[])
    use_case = DocumentLookupUseCase(vector_index=vector_index)

    with pytest.raises(BusinessError) as exc:
        await use_case.get_by_doi(
            request_id="rid",
            embedding_model_id="m",
            doi="10.1000/xyz123",
        )

    assert exc.value.code == "not_found"


@pytest.mark.anyio
async def test_list_dois_simple() -> None:
    vector_index = AsyncMock()
    vector_index.fetch_all = AsyncMock(
        return_value=[
            _hit(
                point_id="c1",
                payload={
                    "doc_id": "doc1",
                    "doi_original": "10.1000/xyz123",
                    "chunk_index": 0,
                    "text": "hello",
                },
            ),
            _hit(
                point_id="c2",
                payload={
                    "doc_id": "doc2",
                    "doi_original": "10.2000/abc456",
                    "chunk_index": 0,
                    "text": "hi",
                },
            ),
        ]
    )
    use_case = DocumentLookupUseCase(vector_index=vector_index)

    res = await use_case.list_dois(
        request_id="rid",
        embedding_model_id="m",
        include_document_info=False,
    )

    assert res.request_id == "rid"
    assert set(res.dois) == {"10.1000/xyz123", "10.2000/abc456"}


@pytest.mark.anyio
async def test_list_dois_expanded() -> None:
    vector_index = AsyncMock()
    vector_index.fetch_all = AsyncMock(
        return_value=[
            _hit(
                point_id="c1",
                payload={
                    "doc_id": "doc1",
                    "doi_original": "10.1000/xyz123",
                    "chunk_index": 0,
                    "text": "hello",
                    "section": "Introduction",
                    "created_at": "2026-01-18T14:32:05Z",
                },
            )
        ]
    )
    use_case = DocumentLookupUseCase(vector_index=vector_index)

    res = await use_case.list_dois(
        request_id="rid",
        embedding_model_id="m",
        include_document_info=True,
    )

    assert res.request_id == "rid"
    assert len(res.documents) == 1
    assert res.documents[0].doi == "10.1000/xyz123"
    assert res.documents[0].sections[0].section == "Introduction"
