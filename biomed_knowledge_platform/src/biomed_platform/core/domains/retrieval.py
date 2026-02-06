from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Mapping


class Disease(StrEnum):
    thrombosis = "thrombosis"
    unknown = "unknown"
    cancer = "cancer"


class SourceType(StrEnum):
    pubmed_abstract = "pubmed_abstract"
    other_abstract = "other_abstract"
    full_text = "full_text"


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    point_id: str
    score: float
    payload: Mapping[str, object]


@dataclass(slots=True)
class DocBest:
    doc_id: str
    doi: str
    score: float
    year: int | None
    source_type: SourceType | None
    title: str | None
    journal: str | None
    authors: list[str] | None


@dataclass(slots=True)
class ChunkPart:
    chunk_id: str
    chunk_index: int
    start: int
    end: int
    text: str


@dataclass(slots=True)
class ChunkCandidate:
    chunk_id: str
    doc_id: str
    doi: str
    title: str | None
    year: int | None
    section: str | None
    source_type: SourceType | None
    score: float
    chunk_text: str | None


@dataclass(frozen=True, slots=True)
class SearchFilters:
    disease: Disease | None = None
    year_min: int | None = None
    year_max: int | None = None
    source_type: SourceType | None = None


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    top_k: int | None = 20
    cursor: str | None = None
    filters: SearchFilters | None = None


@dataclass(frozen=True, slots=True)
class ChunkSection:
    chunk_id: str
    section: str | None


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_ids: list[str]
    sections: list[ChunkSection]
    doc_id: str
    doi: str
    authors: list[str] | None
    journal: str | None
    score: float
    year: int | None
    disease: Disease | None
    source_type: SourceType | None
    title: str | None
    content_text: str | None


@dataclass(frozen=True, slots=True)
class SearchResponse:
    request_id: str
    next_cursor: str | None
    effective_embedding_model_id: str
    hits: list[SearchHit]


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    doc_id: str
    doi: str
    chunk_ids: list[str]
    sections: list[ChunkSection]
    chunk_total: int
    authors: list[str] | None
    journal: str | None
    year: int | None
    disease: Disease | None
    source_type: SourceType | None
    title: str | None
    content_text: str
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class DocumentResponse(DocumentInfo):
    request_id: str


@dataclass(frozen=True, slots=True)
class DoiListSimpleResponse:
    request_id: str
    dois: list[str]


@dataclass(frozen=True, slots=True)
class DoiListExpandedResponse:
    request_id: str
    documents: list[DocumentInfo]
