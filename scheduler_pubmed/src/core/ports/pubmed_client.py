from __future__ import annotations

from typing import Protocol

from scheduler_pubmed.src.core.domains.scheduler import PubMedSearchResult


class PubMedClient(Protocol):
    async def search(
        self,
        *,
        query: str,
        reldate_days: int | None = None,
    ) -> list[PubMedSearchResult]:
        raise NotImplementedError
