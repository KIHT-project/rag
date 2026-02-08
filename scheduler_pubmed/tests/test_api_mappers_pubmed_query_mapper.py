from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from scheduler_pubmed.src.api.mappers.pubmed_query_mapper import (
    to_api_pubmed_query,
    to_api_pubmed_query_list,
    to_domain_create_pubmed_query,
    to_domain_update_pubmed_query,
)
from scheduler_pubmed.src.api.models import schemas
from scheduler_pubmed.src.core.domains.pubmed_query import PubMedQuery


def test_to_domain_create_pubmed_query_maps_fields() -> None:
    payload = schemas.PubMedQueryCreate(
        pubmed_query="cancer[Title]",
        description="Cancer papers",
        enabled=False,
    )

    command = to_domain_create_pubmed_query(payload)

    assert command.pubmed_query == "cancer[Title]"
    assert command.description == "Cancer papers"
    assert command.enabled is False


def test_to_domain_update_pubmed_query_maps_fields() -> None:
    payload = schemas.PubMedQueryUpdate(
        pubmed_query="heart disease[Title]",
        description="Heart disease papers",
    )

    command = to_domain_update_pubmed_query(payload)

    assert command.pubmed_query == "heart disease[Title]"
    assert command.description == "Heart disease papers"


def test_to_api_pubmed_query_maps_domain_model() -> None:
    now = datetime(2026, 2, 8, 13, 0, 0, tzinfo=timezone.utc)
    model = PubMedQuery(
        id=uuid4(),
        pubmed_query="diabetes[Title]",
        description="Diabetes papers",
        enabled=True,
        last_successful_run_at=None,
        created_at=now,
        updated_at=now,
    )

    response = to_api_pubmed_query(model)

    assert response.id == model.id
    assert response.pubmed_query == model.pubmed_query
    assert response.description == model.description
    assert response.enabled is True
    assert response.last_successful_run_at is None
    assert response.created_at == now
    assert response.updated_at == now


def test_to_api_pubmed_query_list_maps_all_items() -> None:
    now = datetime(2026, 2, 8, 13, 0, 0, tzinfo=timezone.utc)
    models = [
        PubMedQuery(
            id=uuid4(),
            pubmed_query="query-a",
            description="desc-a",
            enabled=True,
            last_successful_run_at=None,
            created_at=now,
            updated_at=now,
        ),
        PubMedQuery(
            id=uuid4(),
            pubmed_query="query-b",
            description="desc-b",
            enabled=False,
            last_successful_run_at=None,
            created_at=now,
            updated_at=now,
        ),
    ]

    response = to_api_pubmed_query_list(models)

    assert len(response) == 2
    assert response[0].pubmed_query == "query-a"
    assert response[1].enabled is False
