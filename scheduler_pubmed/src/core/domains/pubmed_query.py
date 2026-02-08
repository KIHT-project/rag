from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True)
class PubMedQuery:
    id: UUID
    pubmed_query: str
    description: str
    enabled: bool
    last_successful_run_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class CreatePubMedQueryCommand:
    pubmed_query: str
    description: str
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class UpdatePubMedQueryCommand:
    pubmed_query: str
    description: str
