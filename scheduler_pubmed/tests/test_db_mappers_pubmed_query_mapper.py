from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from scheduler_pubmed.src.db.mappers.pubmed_query_mapper import to_domain_pubmed_query
from scheduler_pubmed.src.db.models.scheduler import PubMedQuery


def test_to_domain_pubmed_query_maps_model() -> None:
    now = datetime(2026, 2, 8, 14, 0, 0, tzinfo=timezone.utc)
    db_model = PubMedQuery(
        id=uuid4(),
        pubmed_query="oncology",
        description="Oncology papers",
        enabled=True,
        last_successful_run_at=now,
        created_at=now,
        updated_at=now,
    )

    domain_model = to_domain_pubmed_query(db_model)

    assert domain_model.id == db_model.id
    assert domain_model.pubmed_query == "oncology"
    assert domain_model.description == "Oncology papers"
    assert domain_model.enabled is True
    assert domain_model.last_successful_run_at == now
    assert domain_model.created_at == now
    assert domain_model.updated_at == now
