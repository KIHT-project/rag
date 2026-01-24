from dataclasses import dataclass
from enum import Enum
from typing import Any


class RetrievalOrigin(str, Enum):
    QUESTION = "question"
    HYDE = "hyde"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class HybridChunkCandidate:
    chunk_id: str
    doc_id: str
    doi: str
    title: str | None
    year: int | None
    section: str | None
    source_type: Any
    score: float
    chunk_text: str | None
    origin: RetrievalOrigin
