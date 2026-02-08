from __future__ import annotations

from typing import Protocol
from uuid import UUID

from scheduler_pubmed.src.core.domains.pubmed_query import (
    CreatePubMedQueryCommand,
    PubMedQuery,
    UpdatePubMedQueryCommand,
)


class PubMedQueryRepository(Protocol):
    async def create(self, *, command: CreatePubMedQueryCommand) -> PubMedQuery:
        raise NotImplementedError

    async def list(self, *, enabled: bool | None = None) -> list[PubMedQuery]:
        raise NotImplementedError

    async def get_by_id(self, *, query_id: UUID) -> PubMedQuery | None:
        raise NotImplementedError

    async def update(
        self,
        *,
        query_id: UUID,
        command: UpdatePubMedQueryCommand,
    ) -> PubMedQuery | None:
        raise NotImplementedError

    async def set_enabled(self, *, query_id: UUID, enabled: bool) -> PubMedQuery | None:
        raise NotImplementedError
