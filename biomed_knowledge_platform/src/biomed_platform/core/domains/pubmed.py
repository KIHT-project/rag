from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PubMedDocument:
    doi: str | None
    pmid: str | None
    pmcid: str | None
    title: str | None
    journal: str | None
    year: int | None
    authors: list[str] | None
    mesh_terms: list[str] | None
    abstract: str | None
    full_text: str | None
