from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from scheduler_pubmed.src.api.endpoints import pubmed_queries as endpoint_mod
from scheduler_pubmed.src.api.error_handlers import install_error_handlers
from scheduler_pubmed.src.core.domains.pubmed_query import (
    CreatePubMedQueryCommand,
    PubMedQuery,
    UpdatePubMedQueryCommand,
)
from scheduler_pubmed.src.core.errors.errors import business_error


class _FakeUseCase:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self._query = PubMedQuery(
            id=uuid4(),
            pubmed_query="seed",
            description="seed description",
            enabled=True,
            last_successful_run_at=None,
            created_at=now,
            updated_at=now,
        )
        self.last_create: CreatePubMedQueryCommand | None = None
        self.last_update: tuple[UUID, UpdatePubMedQueryCommand] | None = None
        self.last_enabled: tuple[UUID, bool] | None = None

    async def create(self, *, command: CreatePubMedQueryCommand) -> PubMedQuery:
        self.last_create = command
        return self._query

    async def list(self, *, enabled: bool | None = None) -> list[PubMedQuery]:
        if enabled is None:
            return [self._query]
        return [self._query] if self._query.enabled is enabled else []

    async def get_by_id(self, *, query_id: UUID) -> PubMedQuery:
        if query_id != self._query.id:
            raise business_error(
                code="not_found",
                message="PubMed query not found",
                details={"query_id": str(query_id)},
            )
        return self._query

    async def update(self, *, query_id: UUID, command: UpdatePubMedQueryCommand) -> PubMedQuery:
        if query_id != self._query.id:
            raise business_error(
                code="not_found",
                message="PubMed query not found",
                details={"query_id": str(query_id)},
            )
        self.last_update = (query_id, command)
        return self._query

    async def enable(self, *, query_id: UUID) -> PubMedQuery:
        if query_id != self._query.id:
            raise business_error(
                code="not_found",
                message="PubMed query not found",
                details={"query_id": str(query_id)},
            )
        self.last_enabled = (query_id, True)
        return self._query

    async def disable(self, *, query_id: UUID) -> PubMedQuery:
        if query_id != self._query.id:
            raise business_error(
                code="not_found",
                message="PubMed query not found",
                details={"query_id": str(query_id)},
            )
        self.last_enabled = (query_id, False)
        return self._query


@pytest.fixture
def query_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, _FakeUseCase]:
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(endpoint_mod.router)

    fake_use_case = _FakeUseCase()
    monkeypatch.setattr(endpoint_mod, "_get_use_case", lambda _request: fake_use_case)

    with TestClient(app) as client:
        yield client, fake_use_case


def test_create_pubmed_query_returns_201(query_client: tuple[TestClient, _FakeUseCase]) -> None:
    client, use_case = query_client

    response = client.post(
        "/v1/pubmed/queries",
        json={
            "pubmed_query": "oncology",
            "description": "oncology papers",
            "enabled": True,
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == str(use_case._query.id)
    assert body["pubmed_query"] == "seed"
    assert use_case.last_create is not None
    assert use_case.last_create.pubmed_query == "oncology"


def test_list_pubmed_queries_returns_200(query_client: tuple[TestClient, _FakeUseCase]) -> None:
    client, _ = query_client

    response = client.get("/v1/pubmed/queries")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 1


def test_get_pubmed_query_not_found(query_client: tuple[TestClient, _FakeUseCase]) -> None:
    client, _ = query_client

    response = client.get(f"/v1/pubmed/queries/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_update_pubmed_query_returns_200(query_client: tuple[TestClient, _FakeUseCase]) -> None:
    client, use_case = query_client

    response = client.patch(
        f"/v1/pubmed/queries/{use_case._query.id}",
        json={"pubmed_query": "updated", "description": "updated description"},
    )

    assert response.status_code == 200
    assert use_case.last_update is not None
    assert use_case.last_update[1].pubmed_query == "updated"


def test_disable_enable_pubmed_query(query_client: tuple[TestClient, _FakeUseCase]) -> None:
    client, use_case = query_client

    disable_response = client.patch(f"/v1/pubmed/queries/{use_case._query.id}/disable")
    enable_response = client.patch(f"/v1/pubmed/queries/{use_case._query.id}/enable")

    assert disable_response.status_code == 200
    assert enable_response.status_code == 200
    assert use_case.last_enabled == (use_case._query.id, True)


def test_openapi_includes_pubmed_queries_tag() -> None:
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(endpoint_mod.router)

    schema = app.openapi()
    assert schema["paths"]["/v1/pubmed/queries"]["post"]["tags"] == ["PubMed Queries"]
