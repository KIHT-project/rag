from __future__ import annotations

from uuid import UUID

from scheduler_pubmed.src.core.domains.pubmed_query import (
    CreatePubMedQueryCommand,
    PubMedQuery,
    UpdatePubMedQueryCommand,
)
from scheduler_pubmed.src.core.errors.errors import business_error
from scheduler_pubmed.src.core.ports.pubmed_query_repository import PubMedQueryRepository


def _normalize_non_empty(*, field_name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise business_error(
            code="validation_error",
            message=f"{field_name} must not be empty",
            details={field_name: value},
        )
    return normalized


class PubMedQueryUseCase:
    def __init__(self, *, repository: PubMedQueryRepository) -> None:
        self._repository = repository

    async def create(self, *, command: CreatePubMedQueryCommand) -> PubMedQuery:
        normalized = CreatePubMedQueryCommand(
            pubmed_query=_normalize_non_empty(
                field_name="pubmed_query", value=command.pubmed_query
            ),
            description=_normalize_non_empty(field_name="description", value=command.description),
            enabled=bool(command.enabled),
        )
        return await self._repository.create(command=normalized)

    async def list(self, *, enabled: bool | None = None) -> list[PubMedQuery]:
        return await self._repository.list(enabled=enabled)

    async def get_by_id(self, *, query_id: UUID) -> PubMedQuery:
        query = await self._repository.get_by_id(query_id=query_id)
        if query is None:
            raise business_error(
                code="not_found",
                message="PubMed query not found",
                details={"query_id": str(query_id)},
            )
        return query

    async def update(self, *, query_id: UUID, command: UpdatePubMedQueryCommand) -> PubMedQuery:
        normalized = UpdatePubMedQueryCommand(
            pubmed_query=_normalize_non_empty(
                field_name="pubmed_query", value=command.pubmed_query
            ),
            description=_normalize_non_empty(field_name="description", value=command.description),
        )
        query = await self._repository.update(query_id=query_id, command=normalized)
        if query is None:
            raise business_error(
                code="not_found",
                message="PubMed query not found",
                details={"query_id": str(query_id)},
            )
        return query

    async def enable(self, *, query_id: UUID) -> PubMedQuery:
        query = await self._repository.set_enabled(query_id=query_id, enabled=True)
        if query is None:
            raise business_error(
                code="not_found",
                message="PubMed query not found",
                details={"query_id": str(query_id)},
            )
        return query

    async def disable(self, *, query_id: UUID) -> PubMedQuery:
        query = await self._repository.set_enabled(query_id=query_id, enabled=False)
        if query is None:
            raise business_error(
                code="not_found",
                message="PubMed query not found",
                details={"query_id": str(query_id)},
            )
        return query
