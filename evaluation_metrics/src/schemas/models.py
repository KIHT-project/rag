from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel, Field


class QueryItem(BaseModel):
    id: str = Field(alias="_id")
    text: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    filters: Optional[dict[str, Any]] = None


class SearchHit(BaseModel):
    doc_id: str
    doi: str
    score: float
    title: str | None = None
    authors: list[str] | str | None = None
    journal: str | None = None
    year: int | None = None
    content_text: str | None = None


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class AskRequest(BaseModel):
    question: str
    filters: Optional[dict[str, Any]] = None


class AskCitation(BaseModel):
    chunk_id: str
    snippet: str


class AskResponse(BaseModel):
    answer: dict[str, Any] | None = None
    citations: list[AskCitation] = []
    raw: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    run_dir: str
