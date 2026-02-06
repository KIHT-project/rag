from dataclasses import dataclass
from enum import StrEnum

from biomed_platform.core.domains.ingestion import IngestBatchAccepted
from biomed_platform.core.domains.retrieval import SourceType


class ContentTextSource(StrEnum):
    pmc = "pmc"
    abstract = "abstract"


@dataclass(frozen=True, slots=True)
class DocumentFetchResponse:
    request_id: str
    doi: str | None
    pmid: str | None
    title: str | None
    journal: str | None
    year: int | None
    authors: list[str] | None
    source_type: SourceType | None
    content_text: str
    content_text_source: ContentTextSource
    full_text_available: bool
    ingest: IngestBatchAccepted | None = None
