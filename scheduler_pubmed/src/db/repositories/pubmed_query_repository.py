from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scheduler_pubmed.src.core.domains.pubmed_query import (
    CreatePubMedQueryCommand,
    PubMedQuery,
    UpdatePubMedQueryCommand,
)
from scheduler_pubmed.src.db.mappers.pubmed_query_mapper import to_domain_pubmed_query
from scheduler_pubmed.src.db.models.scheduler import PubMedQuery as DbPubMedQuery


class SqlAlchemyPubMedQueryRepository:
    def __init__(self, *, session_maker: async_sessionmaker[AsyncSession]) -> None:
        self._session_maker = session_maker

    async def create(self, *, command: CreatePubMedQueryCommand) -> PubMedQuery:
        async with self._session_maker() as session:
            model = DbPubMedQuery(
                pubmed_query=command.pubmed_query,
                description=command.description,
                enabled=command.enabled,
            )
            session.add(model)
            await session.flush()
            await session.refresh(model)
            await session.commit()
            return to_domain_pubmed_query(model)

    async def list(self, *, enabled: bool | None = None) -> list[PubMedQuery]:
        async with self._session_maker() as session:
            stmt = select(DbPubMedQuery).order_by(DbPubMedQuery.created_at.asc())
            if enabled is not None:
                stmt = stmt.where(DbPubMedQuery.enabled.is_(enabled))
            result = await session.execute(stmt)
            models = list(result.scalars().all())
            return [to_domain_pubmed_query(model) for model in models]

    async def get_by_id(self, *, query_id: UUID) -> PubMedQuery | None:
        async with self._session_maker() as session:
            model = await session.get(DbPubMedQuery, query_id)
            if model is None:
                return None
            return to_domain_pubmed_query(model)

    async def update(
        self,
        *,
        query_id: UUID,
        command: UpdatePubMedQueryCommand,
    ) -> PubMedQuery | None:
        async with self._session_maker() as session:
            model = await session.get(DbPubMedQuery, query_id)
            if model is None:
                return None

            model.pubmed_query = command.pubmed_query
            model.description = command.description
            model.updated_at = datetime.now(timezone.utc)

            await session.flush()
            await session.refresh(model)
            await session.commit()
            return to_domain_pubmed_query(model)

    async def set_enabled(self, *, query_id: UUID, enabled: bool) -> PubMedQuery | None:
        async with self._session_maker() as session:
            model = await session.get(DbPubMedQuery, query_id)
            if model is None:
                return None

            model.enabled = enabled
            model.updated_at = datetime.now(timezone.utc)

            await session.flush()
            await session.refresh(model)
            await session.commit()
            return to_domain_pubmed_query(model)
