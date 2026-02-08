from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Path, Query, Request, status

from scheduler_pubmed.src.api.contracts import contracts
from scheduler_pubmed.src.api.mappers.pubmed_query_mapper import (
    to_api_pubmed_query,
    to_api_pubmed_query_list,
    to_domain_create_pubmed_query,
    to_domain_update_pubmed_query,
)
from scheduler_pubmed.src.api.models import schemas
from scheduler_pubmed.src.core.errors.errors import SystemError
from scheduler_pubmed.src.core.use_cases.pubmed_queries import PubMedQueryUseCase
from scheduler_pubmed.src.db.repositories.pubmed_query_repository import (
    SqlAlchemyPubMedQueryRepository,
)

router = APIRouter(prefix="/v1/pubmed/queries", tags=["PubMed Queries"])


def _get_use_case(request: Request) -> PubMedQueryUseCase:
    session_maker = getattr(request.app.state, "db_sessionmaker", None)
    if session_maker is None:
        raise SystemError(
            code="service_not_configured",
            message="Database session is not configured",
            details=None,
            retryable=False,
        )
    repository = SqlAlchemyPubMedQueryRepository(session_maker=session_maker)
    return PubMedQueryUseCase(repository=repository)


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=schemas.PubMedQuery,
    operation_id="createPubMedQuery",
    responses={
        400: {"model": schemas.ErrorResponse},
        422: {"model": schemas.ErrorResponse},
        500: {"model": schemas.ErrorResponse},
    },
    summary="Create a PubMed query",
)
async def create_pubmed_query(
    request: Request,
    body: contracts.CreatePubMedQueryRequest,
) -> contracts.CreatePubMedQueryResponse:
    use_case = _get_use_case(request)
    created = await use_case.create(command=to_domain_create_pubmed_query(body))
    return to_api_pubmed_query(created)


@router.get(
    "",
    response_model=list[schemas.PubMedQuery],
    operation_id="listPubMedQueries",
    responses={
        400: {"model": schemas.ErrorResponse},
        422: {"model": schemas.ErrorResponse},
        500: {"model": schemas.ErrorResponse},
    },
    summary="List PubMed queries",
)
async def list_pubmed_queries(
    request: Request,
    enabled: bool | None = Query(default=None),
) -> contracts.ListPubMedQueriesResponse:
    use_case = _get_use_case(request)
    queries = await use_case.list(enabled=enabled)
    return to_api_pubmed_query_list(queries)


@router.get(
    "/{queryId}",
    response_model=schemas.PubMedQuery,
    operation_id="getPubMedQuery",
    responses={
        404: {"model": schemas.ErrorResponse},
        422: {"model": schemas.ErrorResponse},
        500: {"model": schemas.ErrorResponse},
    },
    summary="Get PubMed query by id",
)
async def get_pubmed_query(
    request: Request,
    query_id: UUID = Path(alias="queryId"),
) -> contracts.GetPubMedQueryResponse:
    use_case = _get_use_case(request)
    query = await use_case.get_by_id(query_id=query_id)
    return to_api_pubmed_query(query)


@router.patch(
    "/{queryId}",
    response_model=schemas.PubMedQuery,
    operation_id="updatePubMedQuery",
    responses={
        400: {"model": schemas.ErrorResponse},
        404: {"model": schemas.ErrorResponse},
        422: {"model": schemas.ErrorResponse},
        500: {"model": schemas.ErrorResponse},
    },
    summary="Partially Update PubMed query",
)
async def update_pubmed_query(
    request: Request,
    body: contracts.UpdatePubMedQueryRequest,
    query_id: UUID = Path(alias="queryId"),
) -> contracts.UpdatePubMedQueryResponse:
    use_case = _get_use_case(request)
    updated = await use_case.update(
        query_id=query_id,
        command=to_domain_update_pubmed_query(body),
    )
    return to_api_pubmed_query(updated)


@router.patch(
    "/{queryId}/enable",
    response_model=schemas.PubMedQuery,
    operation_id="enablePubMedQuery",
    responses={
        404: {"model": schemas.ErrorResponse},
        422: {"model": schemas.ErrorResponse},
        500: {"model": schemas.ErrorResponse},
    },
    summary="Enable PubMed query",
)
async def enable_pubmed_query(
    request: Request,
    query_id: UUID = Path(alias="queryId"),
) -> contracts.EnablePubMedQueryResponse:
    use_case = _get_use_case(request)
    query = await use_case.enable(query_id=query_id)
    return to_api_pubmed_query(query)


@router.patch(
    "/{queryId}/disable",
    response_model=schemas.PubMedQuery,
    operation_id="disablePubMedQuery",
    responses={
        404: {"model": schemas.ErrorResponse},
        422: {"model": schemas.ErrorResponse},
        500: {"model": schemas.ErrorResponse},
    },
    summary="Disable PubMed query",
)
async def disable_pubmed_query(
    request: Request,
    query_id: UUID = Path(alias="queryId"),
) -> contracts.DisablePubMedQueryResponse:
    use_case = _get_use_case(request)
    query = await use_case.disable(query_id=query_id)
    return to_api_pubmed_query(query)
