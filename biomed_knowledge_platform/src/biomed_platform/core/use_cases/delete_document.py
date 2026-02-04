from __future__ import annotations

from biomed_platform.common.logging import get_logger
from biomed_platform.common.utils import compute_doc_id, normalize_doi
from biomed_platform.core.errors.errors import business_error
from biomed_platform.core.ports.ingestion import DocumentRegistry, VectorWriter

log = get_logger(__name__)


class DeleteDocumentUseCase:
    def __init__(
        self,
        *,
        vector_index: VectorWriter,
        document_registry: DocumentRegistry,
    ) -> None:
        self._vector_index = vector_index
        self._document_registry = document_registry

    async def execute(
        self,
        *,
        request_id: str,
        embedding_model_id: str,
        doi: str,
    ) -> None:
        doi_normalized = normalize_doi(doi)
        if not doi_normalized:
            raise business_error(
                code="validation_error",
                message="Invalid DOI",
                details={"doi": doi},
            )

        doc_id = compute_doc_id(doi_normalized=doi_normalized)

        if await self._document_registry.is_reserved(
            embedding_model_id=embedding_model_id,
            doc_id=doc_id,
        ):
            raise business_error(
                code="duplicate_doi",
                message="Document deletion blocked by in-flight ingestion",
                details={"doi_normalized": doi_normalized, "doc_id": doc_id},
            )

        exists = await self._vector_index.exists(
            embedding_model_id=embedding_model_id,
            doc_id=doc_id,
        )
        if not exists:
            raise business_error(
                code="not_found",
                message="DOI not found",
                details={"doi_normalized": doi_normalized},
            )

        await self._vector_index.delete_by_doc_id(
            embedding_model_id=embedding_model_id,
            doc_id=doc_id,
        )
        await self._document_registry.delete(
            embedding_model_id=embedding_model_id,
            doc_id=doc_id,
        )

        log.info(
            "Document deleted, request_id=%s, embedding_model_id=%s, doc_id=%s",
            request_id,
            embedding_model_id,
            doc_id,
        )
