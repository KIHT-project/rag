from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from scheduler_pubmed.src.core.domains.pubmed_query import (
    CreatePubMedQueryCommand,
    UpdatePubMedQueryCommand,
)
from scheduler_pubmed.src.db.models.scheduler import PubMedQuery as DbPubMedQuery
from scheduler_pubmed.src.db.repositories.pubmed_query_repository import (
    SqlAlchemyPubMedQueryRepository,
)


class _FakeScalarResult:
    def __init__(self, items: list[DbPubMedQuery]) -> None:
        self._items = items

    def all(self) -> list[DbPubMedQuery]:
        return self._items


class _FakeResult:
    def __init__(self, items: list[DbPubMedQuery]) -> None:
        self._items = items

    def scalars(self) -> _FakeScalarResult:
        return _FakeScalarResult(self._items)


class _FakeSession:
    def __init__(
        self,
        *,
        execute_items: list[DbPubMedQuery] | None = None,
        get_items: dict[UUID, DbPubMedQuery] | None = None,
    ) -> None:
        self.execute_items = execute_items or []
        self.get_items = get_items or {}
        self.added: list[DbPubMedQuery] = []
        self.last_stmt = None
        self.commits = 0
        self.flushes = 0
        self.refreshes = 0

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def add(self, model: DbPubMedQuery) -> None:
        self.added.append(model)

    async def flush(self) -> None:
        self.flushes += 1

    async def refresh(self, model: DbPubMedQuery) -> None:
        self.refreshes += 1
        now = datetime.now(timezone.utc)
        if getattr(model, "id", None) is None:
            model.id = uuid4()
        if getattr(model, "created_at", None) is None:
            model.created_at = now
        if getattr(model, "updated_at", None) is None:
            model.updated_at = now
        if getattr(model, "enabled", None) is None:
            model.enabled = True

    async def commit(self) -> None:
        self.commits += 1

    async def execute(self, stmt):
        self.last_stmt = stmt
        return _FakeResult(self.execute_items)

    async def get(self, _model_cls, query_id: UUID):
        return self.get_items.get(query_id)


class _FakeSessionMaker:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session

    def __call__(self) -> _FakeSession:
        return self.session


@pytest.mark.asyncio
async def test_create_persists_query() -> None:
    session = _FakeSession()
    repository = SqlAlchemyPubMedQueryRepository(session_maker=_FakeSessionMaker(session))  # type: ignore[arg-type]

    created = await repository.create(
        command=CreatePubMedQueryCommand(
            pubmed_query="new query",
            description="new description",
            enabled=False,
        )
    )

    assert len(session.added) == 1
    assert session.commits == 1
    assert created.pubmed_query == "new query"
    assert created.enabled is False


@pytest.mark.asyncio
async def test_list_returns_domain_queries() -> None:
    now = datetime.now(timezone.utc)
    db_item = DbPubMedQuery(
        id=uuid4(),
        pubmed_query="listed",
        description="listed description",
        enabled=True,
        last_successful_run_at=None,
        created_at=now,
        updated_at=now,
    )
    session = _FakeSession(execute_items=[db_item])
    repository = SqlAlchemyPubMedQueryRepository(session_maker=_FakeSessionMaker(session))  # type: ignore[arg-type]

    result = await repository.list(enabled=True)

    assert len(result) == 1
    assert result[0].id == db_item.id
    assert "WHERE" in str(session.last_stmt)


@pytest.mark.asyncio
async def test_get_by_id_returns_none_when_missing() -> None:
    repository = SqlAlchemyPubMedQueryRepository(
        session_maker=_FakeSessionMaker(_FakeSession())  # type: ignore[arg-type]
    )

    result = await repository.get_by_id(query_id=uuid4())

    assert result is None


@pytest.mark.asyncio
async def test_update_updates_existing_row() -> None:
    now = datetime.now(timezone.utc)
    query_id = uuid4()
    db_item = DbPubMedQuery(
        id=query_id,
        pubmed_query="before",
        description="before description",
        enabled=True,
        last_successful_run_at=None,
        created_at=now,
        updated_at=now,
    )
    session = _FakeSession(get_items={query_id: db_item})
    repository = SqlAlchemyPubMedQueryRepository(session_maker=_FakeSessionMaker(session))  # type: ignore[arg-type]

    result = await repository.update(
        query_id=query_id,
        command=UpdatePubMedQueryCommand(
            pubmed_query="after",
            description="after description",
        ),
    )

    assert result is not None
    assert result.pubmed_query == "after"
    assert result.description == "after description"
    assert session.commits == 1


@pytest.mark.asyncio
async def test_set_enabled_returns_none_when_missing() -> None:
    repository = SqlAlchemyPubMedQueryRepository(
        session_maker=_FakeSessionMaker(_FakeSession())  # type: ignore[arg-type]
    )

    result = await repository.set_enabled(query_id=uuid4(), enabled=False)

    assert result is None


@pytest.mark.asyncio
async def test_set_enabled_updates_existing_row() -> None:
    now = datetime.now(timezone.utc)
    query_id = uuid4()
    db_item = DbPubMedQuery(
        id=query_id,
        pubmed_query="query",
        description="description",
        enabled=True,
        last_successful_run_at=None,
        created_at=now,
        updated_at=now,
    )
    session = _FakeSession(get_items={query_id: db_item})
    repository = SqlAlchemyPubMedQueryRepository(session_maker=_FakeSessionMaker(session))  # type: ignore[arg-type]

    result = await repository.set_enabled(query_id=query_id, enabled=False)

    assert result is not None
    assert result.enabled is False
    assert session.commits == 1
