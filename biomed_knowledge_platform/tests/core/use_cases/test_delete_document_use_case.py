from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from biomed_platform.common.utils import compute_doc_id, normalize_doi
from biomed_platform.core.errors.errors import BusinessError
from biomed_platform.core.use_cases.delete_document import DeleteDocumentUseCase


@pytest.mark.anyio
async def test_delete_document_rejects_invalid_doi() -> None:
    # Given
    use_case = DeleteDocumentUseCase(
        vector_index=AsyncMock(),
        document_registry=AsyncMock(),
    )

    # When
    with pytest.raises(BusinessError) as exc:
        await use_case.execute(
            request_id="rid",
            embedding_model_id="m",
            doi="not-a-doi",
        )

    # Then
    assert exc.value.code == "validation_error"


@pytest.mark.anyio
async def test_delete_document_conflict_when_reserved() -> None:
    # Given
    vector_index = AsyncMock()
    registry = AsyncMock()
    registry.is_reserved = AsyncMock(return_value=True)

    use_case = DeleteDocumentUseCase(
        vector_index=vector_index,
        document_registry=registry,
    )

    # When
    with pytest.raises(BusinessError) as exc:
        await use_case.execute(
            request_id="rid",
            embedding_model_id="m",
            doi="10.1000/xyz123",
        )

    # Then
    assert exc.value.code == "duplicate_doi"
    vector_index.exists.assert_not_called()


@pytest.mark.anyio
async def test_delete_document_not_found() -> None:
    # Given
    vector_index = AsyncMock()
    vector_index.exists = AsyncMock(return_value=False)
    registry = AsyncMock()
    registry.is_reserved = AsyncMock(return_value=False)

    use_case = DeleteDocumentUseCase(
        vector_index=vector_index,
        document_registry=registry,
    )

    # When
    with pytest.raises(BusinessError) as exc:
        await use_case.execute(
            request_id="rid",
            embedding_model_id="m",
            doi="10.1000/xyz123",
        )

    # Then
    assert exc.value.code == "not_found"


@pytest.mark.anyio
async def test_delete_document_deletes_vectors_and_registry() -> None:
    # Given
    vector_index = AsyncMock()
    vector_index.exists = AsyncMock(return_value=True)
    registry = AsyncMock()
    registry.is_reserved = AsyncMock(return_value=False)

    use_case = DeleteDocumentUseCase(
        vector_index=vector_index,
        document_registry=registry,
    )

    doi = "10.1000/xyz123"
    doi_normalized = normalize_doi(doi)
    doc_id = compute_doc_id(doi_normalized=doi_normalized)

    # When
    await use_case.execute(
        request_id="rid",
        embedding_model_id="m",
        doi=doi,
    )

    # Then
    vector_index.delete_by_doc_id.assert_awaited_once_with(
        embedding_model_id="m",
        doc_id=doc_id,
    )
    registry.delete.assert_awaited_once_with(
        embedding_model_id="m",
        doc_id=doc_id,
    )
