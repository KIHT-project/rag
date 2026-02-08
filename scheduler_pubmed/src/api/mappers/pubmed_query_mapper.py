from __future__ import annotations

from scheduler_pubmed.src.api.contracts import contracts
from scheduler_pubmed.src.api.models import schemas
from scheduler_pubmed.src.core.domains.pubmed_query import (
    CreatePubMedQueryCommand,
    PubMedQuery,
    UpdatePubMedQueryCommand,
)


def to_domain_create_pubmed_query(
    payload: contracts.CreatePubMedQueryRequest,
) -> CreatePubMedQueryCommand:
    return CreatePubMedQueryCommand(
        pubmed_query=payload.pubmed_query,
        description=payload.description,
        enabled=payload.enabled,
    )


def to_domain_update_pubmed_query(
    payload: contracts.UpdatePubMedQueryRequest,
) -> UpdatePubMedQueryCommand:
    return UpdatePubMedQueryCommand(
        pubmed_query=payload.pubmed_query,
        description=payload.description,
    )


def to_api_pubmed_query(model: PubMedQuery) -> schemas.PubMedQuery:
    return schemas.PubMedQuery(
        id=model.id,
        pubmed_query=model.pubmed_query,
        description=model.description,
        enabled=model.enabled,
        last_successful_run_at=model.last_successful_run_at,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def to_api_pubmed_query_list(models: list[PubMedQuery]) -> list[schemas.PubMedQuery]:
    return [to_api_pubmed_query(model) for model in models]
