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
