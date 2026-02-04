from __future__ import annotations

from fastapi import APIRouter, Header, Request, Response, status

from biomed_platform.api.models.generated import schemas
from biomed_platform.common.logging import get_logger
from biomed_platform.common.middleware.trace import get_request_id
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.use_cases.document_lookup import DocumentLookupUseCase
from biomed_platform.core.use_cases.delete_document import DeleteDocumentUseCase

log = get_logger(__name__)

router = APIRouter(prefix="/v1/documents", tags=["Documents"])


def _resolve_effective_embedding_model_id(*, request: Request) -> str:
    cfg = getattr(request.app.state, "settings", None)
    default_model_id: str | None = None

    if cfg is not None:
        rag_cfg = cfg.require_rag()
        emb_cfg = rag_cfg.get("embedding", {})
        if isinstance(emb_cfg, dict):
            default_model_id = emb_cfg.get("provider")

    effective = (default_model_id or "").strip()
    if not effective:
        raise SystemError(
            code="missing_embedding_model_id",
            message="Missing embedding model id, set rag.embedding.provider",
            details=None,
            retryable=False,
        )

    return effective


@router.delete(
    "/{doi:path}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Deleted"},
        404: {"model": schemas.ErrorResponse},
        409: {"model": schemas.ErrorResponse},
        429: {"model": schemas.ErrorResponse},
    },
    summary="Delete a document by DOI",
)
async def delete_document(request: Request, doi: str) -> Response:
    request_id = get_request_id()
    embedding_model_id = _resolve_effective_embedding_model_id(request=request)

    vector_index = getattr(request.app.state, "vector_index", None)
    document_registry = getattr(request.app.state, "document_registry", None)

    if vector_index is None or document_registry is None:
        raise SystemError(
            code="service_not_configured",
            message="Document deletion service not configured",
            details=None,
            retryable=False,
        )

    use_case = DeleteDocumentUseCase(
        vector_index=vector_index,
        document_registry=document_registry,
    )

    await use_case.execute(
        request_id=request_id,
        embedding_model_id=embedding_model_id,
        doi=doi,
    )

    log.info(
        "Delete document completed, request_id=%s, embedding_model_id=%s",
        request_id,
        embedding_model_id,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": schemas.DoiListSimpleResponse},
        429: {"model": schemas.ErrorResponse},
        500: {"model": schemas.ErrorResponse},
    },
    summary="List ingested DOIs",
)
async def list_dois(
    request: Request,
    x_include_document_info: bool | None = Header(default=None, alias="X-Include-Document-Info"),
) -> schemas.DoiListSimpleResponse | schemas.DoiListExpandedResponse:
    request_id = get_request_id()
    embedding_model_id = _resolve_effective_embedding_model_id(request=request)

    vector_index = getattr(request.app.state, "vector_index", None)
    if vector_index is None:
        raise SystemError(
            code="service_not_configured",
            message="Document lookup service not configured",
            details=None,
            retryable=False,
        )

    include_info = bool(x_include_document_info) if x_include_document_info is not None else False
    use_case = DocumentLookupUseCase(vector_index=vector_index)
    return await use_case.list_dois(
        request_id=request_id,
        embedding_model_id=embedding_model_id,
        include_document_info=include_info,
    )


@router.get(
    "/{doi:path}",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": schemas.DocumentResponse},
        400: {"model": schemas.ErrorResponse},
        404: {"model": schemas.ErrorResponse},
        429: {"model": schemas.ErrorResponse},
        500: {"model": schemas.ErrorResponse},
    },
    summary="Get a document by DOI",
)
async def get_document(request: Request, doi: str) -> schemas.DocumentResponse:
    request_id = get_request_id()
    embedding_model_id = _resolve_effective_embedding_model_id(request=request)

    vector_index = getattr(request.app.state, "vector_index", None)
    if vector_index is None:
        raise SystemError(
            code="service_not_configured",
            message="Document lookup service not configured",
            details=None,
            retryable=False,
        )

    use_case = DocumentLookupUseCase(vector_index=vector_index)
    return await use_case.get_by_doi(
        request_id=request_id,
        embedding_model_id=embedding_model_id,
        doi=doi,
    )
