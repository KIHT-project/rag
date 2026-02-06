from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Header, Request, Response, status

from biomed_platform.api.mappers.documents_mapper import (
    to_api_document_fetch_response,
    to_api_document_response,
    to_api_doi_list_response,
)
from biomed_platform.api.models.generated import schemas
from biomed_platform.common.logging import get_logger
from biomed_platform.common.middleware.trace import get_request_id
from biomed_platform.common.utils import compute_body_hash_from_items
from biomed_platform.core.domains.documents import DocumentFetchResponse
from biomed_platform.core.domains.ingestion import IngestBatchCommand
from biomed_platform.core.errors.errors import SystemError, business_error
from biomed_platform.core.use_cases.document_fetch import DocumentFetchUseCase
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


async def _fetch_batch_no_ingest(
    *,
    request_id: str,
    embedding_model_id: str,
    use_case: DocumentFetchUseCase,
    items: list[schemas.DocumentFetchRequest],
) -> None:
    for item in items:
        root = item.root
        doi = getattr(root, "doi", None)
        pmid = getattr(root, "pmid", None)
        try:
            await use_case.fetch_one(
                request_id=request_id,
                embedding_model_id=embedding_model_id,
                doi=doi,
                pmid=pmid,
                ingest_enabled=False,
            )
        except Exception:
            log.exception(
                "Batch fetch failed for item, request_id=%s, doi=%s, pmid=%s",
                request_id,
                doi,
                pmid,
            )


async def _build_ingest_items(
    *,
    request_id: str,
    pubmed_client,
    use_case: DocumentFetchUseCase,
    items: list[schemas.DocumentFetchRequest],
) -> list:
    ingest_items: list = []
    for item in items:
        root = item.root
        doi = getattr(root, "doi", None)
        pmid = getattr(root, "pmid", None)
        try:
            doc = await pubmed_client.fetch_document(doi=doi, pmid=pmid)
            if doc is None:
                raise business_error(
                    code="not_found",
                    message="DOI or PMID not found",
                    details={"doi": doi, "pmid": pmid},
                )
            fetch_resp: DocumentFetchResponse = use_case.build_fetch_response(
                request_id=request_id,
                doc=doc,
                requested_doi=doi,
                requested_pmid=pmid,
            )
            ingest_items.append(
                use_case.build_ingest_item(
                    doc=doc,
                    doi_value=fetch_resp.doi,
                    content_text=fetch_resp.content_text,
                    source_type=fetch_resp.source_type,
                )
            )
        except Exception:
            log.exception(
                "Batch fetch failed for item, request_id=%s, doi=%s, pmid=%s",
                request_id,
                doi,
                pmid,
            )
    return ingest_items


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


@router.post(
    "/fetch",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": schemas.DocumentFetchResponse},
        400: {"model": schemas.ErrorResponse},
        404: {"model": schemas.ErrorResponse},
        409: {"model": schemas.ErrorResponse},
        422: {"model": schemas.ErrorResponse},
        429: {"model": schemas.ErrorResponse},
        500: {"model": schemas.ErrorResponse},
    },
    summary="Fetch a document by DOI or PMID",
)
async def fetch_document(
    request: Request,
    body: schemas.DocumentFetchRequest,
    x_ingest_enabled: bool | None = Header(default=None, alias="X-Ingest-Enabled"),
) -> schemas.DocumentFetchResponse:
    request_id = get_request_id()
    embedding_model_id = _resolve_effective_embedding_model_id(request=request)
    ingest_enabled = True if x_ingest_enabled is None else bool(x_ingest_enabled)

    pubmed_client = getattr(request.app.state, "pubmed_client", None)
    if pubmed_client is None:
        raise SystemError(
            code="service_not_configured",
            message="PubMed client not configured",
            details=None,
            retryable=False,
        )

    ingestion_service = (
        getattr(request.app.state, "ingestion_service", None) if ingest_enabled else None
    )

    use_case = DocumentFetchUseCase(
        pubmed_client=pubmed_client,
        ingestion_service=ingestion_service,
    )

    root = body.root
    doi = getattr(root, "doi", None)
    pmid = getattr(root, "pmid", None)

    result = await use_case.fetch_one(
        request_id=request_id,
        embedding_model_id=embedding_model_id,
        doi=doi,
        pmid=pmid,
        ingest_enabled=ingest_enabled,
    )
    if isinstance(result, schemas.DocumentFetchResponse):
        return result
    return to_api_document_fetch_response(result)


@router.get(
    "",
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
    result = await use_case.list_dois(
        request_id=request_id,
        embedding_model_id=embedding_model_id,
        include_document_info=include_info,
    )
    if isinstance(result, (schemas.DoiListSimpleResponse, schemas.DoiListExpandedResponse)):
        return result
    return to_api_doi_list_response(result)


@router.post(
    "/fetch/batch",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"model": schemas.DocumentFetchBatchAcceptedResponse},
        400: {"model": schemas.ErrorResponse},
        422: {"model": schemas.ErrorResponse},
        429: {"model": schemas.ErrorResponse},
        500: {"model": schemas.ErrorResponse},
    },
    summary="Fetch documents by DOI or PMID (async)",
)
async def fetch_document_batch(
    request: Request,
    body: schemas.DocumentFetchBatchRequest,
    x_ingest_enabled: bool | None = Header(default=None, alias="X-Ingest-Enabled"),
) -> schemas.DocumentFetchBatchAcceptedResponse:
    request_id = get_request_id()
    embedding_model_id = _resolve_effective_embedding_model_id(request=request)
    ingest_enabled = True if x_ingest_enabled is None else bool(x_ingest_enabled)

    pubmed_client = getattr(request.app.state, "pubmed_client", None)
    if pubmed_client is None:
        raise SystemError(
            code="service_not_configured",
            message="PubMed client not configured",
            details=None,
            retryable=False,
        )

    ingestion_service = (
        getattr(request.app.state, "ingestion_service", None) if ingest_enabled else None
    )

    use_case = DocumentFetchUseCase(
        pubmed_client=pubmed_client,
        ingestion_service=None,
    )

    items = list(body.items)
    if not ingest_enabled:
        job_id = uuid.uuid4().hex
        if items:
            asyncio.create_task(
                _fetch_batch_no_ingest(
                    request_id=request_id,
                    embedding_model_id=embedding_model_id,
                    use_case=use_case,
                    items=items,
                )
            )
        return schemas.DocumentFetchBatchAcceptedResponse(
            request_id=request_id,
            job_id=job_id,
            state=schemas.JobState.queued,
        )

    if ingestion_service is None:
        raise SystemError(
            code="service_not_configured",
            message="Ingestion service not configured",
            details=None,
            retryable=False,
        )

    ingest_items = await _build_ingest_items(
        request_id=request_id,
        pubmed_client=pubmed_client,
        use_case=use_case,
        items=items,
    )

    if not ingest_items:
        raise business_error(
            code="not_found",
            message="No documents fetched for ingestion",
            details=None,
        )

    body_hash = compute_body_hash_from_items(
        effective_embedding_model_id=embedding_model_id,
        items=ingest_items,
    )
    cmd = IngestBatchCommand(
        effective_embedding_model_id=embedding_model_id,
        items=tuple(ingest_items),
        idempotency_key=None,
        body_hash=body_hash,
        correlation_id=request_id,
    )
    accepted = await ingestion_service.ingest_batch(cmd)

    return schemas.DocumentFetchBatchAcceptedResponse(
        request_id=request_id,
        job_id=accepted.job_id,
        state=schemas.JobState(accepted.state.value),
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
    result = await use_case.get_by_doi(
        request_id=request_id,
        embedding_model_id=embedding_model_id,
        doi=doi,
    )
    if isinstance(result, schemas.DocumentResponse):
        return result
    return to_api_document_response(result)
