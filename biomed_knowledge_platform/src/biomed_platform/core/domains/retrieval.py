from dataclasses import dataclass
from typing import Mapping

from biomed_platform.api.models.generated import schemas


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
    source_type: schemas.SourceType | None
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
    source_type: schemas.SourceType | None
    score: float
    chunk_text: str | None
