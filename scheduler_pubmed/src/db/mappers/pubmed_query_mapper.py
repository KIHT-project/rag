from __future__ import annotations

from scheduler_pubmed.src.core.domains.pubmed_query import PubMedQuery as DomainPubMedQuery
from scheduler_pubmed.src.db.models.scheduler import PubMedQuery as DbPubMedQuery


def to_domain_pubmed_query(model: DbPubMedQuery) -> DomainPubMedQuery:
    return DomainPubMedQuery(
        id=model.id,
        pubmed_query=model.pubmed_query,
        description=model.description,
        enabled=model.enabled,
        last_successful_run_at=model.last_successful_run_at,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )
