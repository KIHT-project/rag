from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Sequence


class JobState(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    partial = "partial"


class IngestItemState(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    skipped_duplicate = "skipped_duplicate"


@dataclass(frozen=True, slots=True)
class IngestItem:
    doi_original: str
    doi_normalized: str
    disease: str
    source_type: str
    content_text: str
    year: int | None = None
    title: str | None = None
    journal: str | None = None
    authors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IngestItemStatus:
    doi_original: str
    doc_id: str
    state: IngestItemState
    message: str | None = None


@dataclass(frozen=True, slots=True)
class JobCounts:
    total: int
    succeeded: int
    failed: int
    skipped_duplicate: int


@dataclass(slots=True)
class IngestionJob:
    job_id: str
    state: JobState
    created_at: datetime
    updated_at: datetime
    effective_embedding_model_id: str
    correlation_id: str | None = None

    items: list[IngestItemStatus] = field(default_factory=list)
    counts: JobCounts | None = None


@dataclass(frozen=True, slots=True)
class IngestBatchCommand:
    effective_embedding_model_id: str
    items: tuple[IngestItem, ...]
    idempotency_key: str | None
    body_hash: str
    correlation_id: str | None


@dataclass(frozen=True, slots=True)
class IngestBatchAccepted:
    job_id: str
    state: JobState  # always queued


@dataclass(frozen=True, slots=True)
class RetryAfterHint:
    seconds: int


@dataclass(frozen=True, slots=True)
class TextChunk:
    index: int
    text: str
    start: int
    end: int
    section: str | None = None
    subsection: str | None = None


@dataclass(frozen=True, slots=True)
class VectorPoint:
    point_id: str  # must be UUID or uint for Qdrant
    vector: Sequence[float]
    payload: dict[str, object]


@dataclass(frozen=True)
class ReservedDocs:
    items_with_doc_id: list[tuple[IngestItem, str]]
    reserved_doc_ids: list[str]


@dataclass(frozen=True)
class SplitItems:
    valid: list[IngestItem]
    invalid: list[IngestItem]


@dataclass(frozen=True, slots=True)
class SkippedDuplicate:
    item: IngestItem
    doc_id: str
    message: str


@dataclass(frozen=True)
class ReserveResult:
    reserved: ReservedDocs
    skipped_duplicates: list[SkippedDuplicate]


@dataclass(slots=True)
class JobStats:
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
