from __future__ import annotations

from typing import Protocol

from biomed_platform.core.domains.pubmed import PubMedDocument


class PubMedClient(Protocol):
    async def fetch_document(
        self,
        *,
        doi: str | None,
        pmid: str | None,
    ) -> PubMedDocument | None: ...
