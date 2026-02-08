from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from scheduler_pubmed.src.core.domains.pubmed_query import (
    CreatePubMedQueryCommand,
    PubMedQuery,
    UpdatePubMedQueryCommand,
)
from scheduler_pubmed.src.core.errors.errors import AppError
from scheduler_pubmed.src.core.use_cases.pubmed_queries import PubMedQueryUseCase


class FakePubMedQueryRepository:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self._items: dict[UUID, PubMedQuery] = {}
        self.seed = PubMedQuery(
            id=uuid4(),
            pubmed_query="seed",
            description="seed description",
            enabled=True,
            last_successful_run_at=None,
            created_at=now,
            updated_at=now,
        )
        self._items[self.seed.id] = self.seed

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
        self._items[model.id] = model
        return model

    async def list(self, *, enabled: bool | None = None) -> list[PubMedQuery]:
        items = list(self._items.values())
        if enabled is None:
            return items
        return [item for item in items if item.enabled is enabled]

    async def get_by_id(self, *, query_id: UUID) -> PubMedQuery | None:
        return self._items.get(query_id)

    async def update(
        self,
        *,
        query_id: UUID,
        command: UpdatePubMedQueryCommand,
    ) -> PubMedQuery | None:
        current = self._items.get(query_id)
        if current is None:
            return None
        updated = replace(
            current,
            pubmed_query=command.pubmed_query,
            description=command.description,
            updated_at=datetime.now(timezone.utc),
        )
        self._items[query_id] = updated
        return updated

    async def set_enabled(self, *, query_id: UUID, enabled: bool) -> PubMedQuery | None:
        current = self._items.get(query_id)
        if current is None:
            return None
        updated = replace(current, enabled=enabled, updated_at=datetime.now(timezone.utc))
        self._items[query_id] = updated
        return updated


@pytest.mark.asyncio
async def test_create_normalizes_inputs() -> None:
    repo = FakePubMedQueryRepository()
    use_case = PubMedQueryUseCase(repository=repo)

    created = await use_case.create(
        command=CreatePubMedQueryCommand(
            pubmed_query="  lung cancer[Title]  ",
            description="  Lung papers  ",
            enabled=True,
        )
    )

    assert created.pubmed_query == "lung cancer[Title]"
    assert created.description == "Lung papers"
    assert created.enabled is True


@pytest.mark.asyncio
async def test_create_rejects_empty_query() -> None:
    repo = FakePubMedQueryRepository()
    use_case = PubMedQueryUseCase(repository=repo)

    with pytest.raises(AppError) as exc_info:
        await use_case.create(
            command=CreatePubMedQueryCommand(
                pubmed_query="   ",
                description="desc",
                enabled=True,
            )
        )

    assert exc_info.value.code == "validation_error"


@pytest.mark.asyncio
async def test_get_by_id_not_found_raises() -> None:
    repo = FakePubMedQueryRepository()
    use_case = PubMedQueryUseCase(repository=repo)

    with pytest.raises(AppError) as exc_info:
        await use_case.get_by_id(query_id=uuid4())

    assert exc_info.value.code == "not_found"


@pytest.mark.asyncio
async def test_update_updates_existing_query() -> None:
    repo = FakePubMedQueryRepository()
    use_case = PubMedQueryUseCase(repository=repo)

    updated = await use_case.update(
        query_id=repo.seed.id,
        command=UpdatePubMedQueryCommand(
            pubmed_query="updated query",
            description="updated description",
        ),
    )

    assert updated.id == repo.seed.id
    assert updated.pubmed_query == "updated query"
    assert updated.description == "updated description"


@pytest.mark.asyncio
async def test_enable_disable_toggle_query() -> None:
    repo = FakePubMedQueryRepository()
    use_case = PubMedQueryUseCase(repository=repo)

    disabled = await use_case.disable(query_id=repo.seed.id)
    enabled = await use_case.enable(query_id=repo.seed.id)

    assert disabled.enabled is False
    assert enabled.enabled is True


@pytest.mark.asyncio
async def test_list_filters_by_enabled() -> None:
    repo = FakePubMedQueryRepository()
    use_case = PubMedQueryUseCase(repository=repo)
    await use_case.disable(query_id=repo.seed.id)

    enabled_items = await use_case.list(enabled=True)
    disabled_items = await use_case.list(enabled=False)

    assert enabled_items == []
    assert len(disabled_items) == 1
    assert disabled_items[0].id == repo.seed.id
