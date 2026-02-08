from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from pytest_bdd import given, scenario, then, when

from scheduler_pubmed.src.api.endpoints import pubmed_queries as endpoint_mod
from scheduler_pubmed.src.core.domains.pubmed_query import (
    CreatePubMedQueryCommand,
    PubMedQuery,
    UpdatePubMedQueryCommand,
)
from scheduler_pubmed.src.core.errors.errors import business_error
from tests.bdd.helpers.scheduler_queries_api import (
    create_pubmed_query,
    disable_pubmed_query,
    enable_pubmed_query,
    get_pubmed_query,
    list_pubmed_queries,
    update_pubmed_query,
)


class _FakeSchedulerPubMedQueryUseCase:
    def __init__(self) -> None:
        self._queries: dict[UUID, PubMedQuery] = {}

    async def create(self, *, command: CreatePubMedQueryCommand) -> PubMedQuery:
        now = datetime.now(timezone.utc)
        model = PubMedQuery(
            id=uuid4(),
            pubmed_query=command.pubmed_query,
            description=command.description,
            enabled=command.enabled,
            last_successful_run_at=None,
            created_at=now,
            updated_at=now,
        )
        self._queries[model.id] = model
        return model

    async def list(self, *, enabled: bool | None = None) -> list[PubMedQuery]:
        items = list(self._queries.values())
        if enabled is None:
            return items
        return [item for item in items if item.enabled is enabled]

    async def get_by_id(self, *, query_id: UUID) -> PubMedQuery:
        query = self._queries.get(query_id)
        if query is None:
            raise business_error(
                code="not_found",
                message="PubMed query not found",
                details={"query_id": str(query_id)},
            )
        return query

    async def update(self, *, query_id: UUID, command: UpdatePubMedQueryCommand) -> PubMedQuery:
        query = await self.get_by_id(query_id=query_id)
        updated = PubMedQuery(
            id=query.id,
            pubmed_query=command.pubmed_query,
            description=command.description,
            enabled=query.enabled,
            last_successful_run_at=query.last_successful_run_at,
            created_at=query.created_at,
            updated_at=datetime.now(timezone.utc),
        )
        self._queries[query_id] = updated
        return updated

    async def disable(self, *, query_id: UUID) -> PubMedQuery:
        return await self._set_enabled(query_id=query_id, enabled=False)

    async def enable(self, *, query_id: UUID) -> PubMedQuery:
        return await self._set_enabled(query_id=query_id, enabled=True)

    async def _set_enabled(self, *, query_id: UUID, enabled: bool) -> PubMedQuery:
        query = await self.get_by_id(query_id=query_id)
        updated = PubMedQuery(
            id=query.id,
            pubmed_query=query.pubmed_query,
            description=query.description,
            enabled=enabled,
            last_successful_run_at=query.last_successful_run_at,
            created_at=query.created_at,
            updated_at=datetime.now(timezone.utc),
        )
        self._queries[query_id] = updated
        return updated


@pytest.fixture(autouse=True)
def fake_scheduler_query_use_case(
    monkeypatch: pytest.MonkeyPatch,
    bdd_target: str,
) -> _FakeSchedulerPubMedQueryUseCase:
    if bdd_target != "scheduler":
        pytest.skip("Scheduler query BDD runs only on scheduler target")
    fake = _FakeSchedulerPubMedQueryUseCase()
    monkeypatch.setattr(endpoint_mod, "_get_use_case", lambda _request: fake)
    return fake


@scenario("../features/pubmed_queries.feature", "Create and get PubMed query")
def test_create_and_get_pubmed_query() -> None:
    pass


@scenario("../features/pubmed_queries.feature", "Update query and toggle enabled status")
def test_update_toggle_pubmed_query() -> None:
    pass


@scenario("../features/pubmed_queries.feature", "List queries filtered by enabled")
def test_list_pubmed_queries_filtered_by_enabled() -> None:
    pass


@given("the scheduler query API is ready")
def given_scheduler_query_api_ready() -> None:
    return


@given("a scheduler PubMed query exists")
def given_scheduler_pubmed_query_exists(client, ctx: dict) -> None:
    response = create_pubmed_query(
        client,
        payload={
            "pubmed_query": "existing query",
            "description": "existing description",
            "enabled": True,
        },
    )
    assert response.status_code == 201
    body = response.json()
    ctx["query_id"] = body["id"]
    ctx["res"] = response


@given("I have both enabled and disabled scheduler queries")
def given_enabled_and_disabled_queries(client) -> None:
    enabled_response = create_pubmed_query(
        client,
        payload={
            "pubmed_query": "enabled query",
            "description": "enabled description",
            "enabled": True,
        },
    )
    assert enabled_response.status_code == 201

    disabled_response = create_pubmed_query(
        client,
        payload={
            "pubmed_query": "disabled query",
            "description": "disabled description",
            "enabled": True,
        },
    )
    assert disabled_response.status_code == 201
    disabled_id = disabled_response.json()["id"]
    toggle = disable_pubmed_query(client, query_id=disabled_id)
    assert toggle.status_code == 200


@when("I create a scheduler PubMed query")
def when_create_scheduler_pubmed_query(client, ctx: dict) -> None:
    response = create_pubmed_query(
        client,
        payload={
            "pubmed_query": "diabetes[Title]",
            "description": "Diabetes papers",
            "enabled": True,
        },
    )
    ctx["res"] = response
    if response.status_code == 201:
        ctx["query_id"] = response.json()["id"]


@when("I get the created scheduler PubMed query by id")
def when_get_created_scheduler_pubmed_query(client, ctx: dict) -> None:
    ctx["res"] = get_pubmed_query(client, query_id=ctx["query_id"])


@when("I update the scheduler PubMed query")
def when_update_scheduler_pubmed_query(client, ctx: dict) -> None:
    ctx["res"] = update_pubmed_query(
        client,
        query_id=ctx["query_id"],
        payload={
            "pubmed_query": "updated query",
            "description": "updated description",
        },
    )


@when("I disable the scheduler PubMed query")
def when_disable_scheduler_pubmed_query(client, ctx: dict) -> None:
    ctx["res"] = disable_pubmed_query(client, query_id=ctx["query_id"])


@when("I enable the scheduler PubMed query")
def when_enable_scheduler_pubmed_query(client, ctx: dict) -> None:
    ctx["res"] = enable_pubmed_query(client, query_id=ctx["query_id"])


@when("I list scheduler PubMed queries with enabled true")
def when_list_scheduler_pubmed_queries_enabled_true(client, ctx: dict) -> None:
    ctx["res"] = list_pubmed_queries(client, enabled=True)


@when("I list scheduler PubMed queries with enabled false")
def when_list_scheduler_pubmed_queries_enabled_false(client, ctx: dict) -> None:
    ctx["res"] = list_pubmed_queries(client, enabled=False)


@then("the scheduler query response status is 201")
def then_scheduler_query_response_status_201(ctx: dict) -> None:
    assert ctx["res"].status_code == 201


@then("the scheduler query response status is 200")
def then_scheduler_query_response_status_200(ctx: dict) -> None:
    assert ctx["res"].status_code == 200


@then("the created scheduler query has an id")
def then_created_scheduler_query_has_id(ctx: dict) -> None:
    query_id = ctx["res"].json().get("id")
    UUID(query_id)


@then("the fetched scheduler query id matches the created id")
def then_fetched_scheduler_query_id_matches(ctx: dict) -> None:
    body = ctx["res"].json()
    assert body["id"] == ctx["query_id"]


@then("the scheduler query description is updated")
def then_scheduler_query_description_is_updated(ctx: dict) -> None:
    assert ctx["res"].json()["description"] == "updated description"


@then("the scheduler query is disabled")
def then_scheduler_query_is_disabled(ctx: dict) -> None:
    assert ctx["res"].json()["enabled"] is False


@then("the scheduler query is enabled")
def then_scheduler_query_is_enabled(ctx: dict) -> None:
    assert ctx["res"].json()["enabled"] is True


@then("all listed scheduler queries are enabled")
def then_all_listed_scheduler_queries_are_enabled(ctx: dict) -> None:
    body = ctx["res"].json()
    assert isinstance(body, list)
    assert len(body) >= 1
    assert all(item["enabled"] is True for item in body)


@then("all listed scheduler queries are disabled")
def then_all_listed_scheduler_queries_are_disabled(ctx: dict) -> None:
    body = ctx["res"].json()
    assert isinstance(body, list)
    assert len(body) >= 1
    assert all(item["enabled"] is False for item in body)
