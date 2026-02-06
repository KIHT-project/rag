from __future__ import annotations

from biomed_platform.api.models.generated import schemas
from biomed_platform.core.domains.retrieval import (
    ChunkSection,
    Disease,
    SearchFilters,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SourceType,
)


def _to_domain_disease(value: schemas.Disease | None) -> Disease | None:
    if value is None:
        return None
    return Disease(value.value)


def _to_domain_source_type(value: schemas.SourceType | None) -> SourceType | None:
    if value is None:
        return None
    return SourceType(value.value)


def _to_api_disease(value: Disease | None) -> schemas.Disease | None:
    if value is None:
        return None
    return schemas.Disease(value.value)


def _to_api_source_type(value: SourceType | None) -> schemas.SourceType | None:
    if value is None:
        return None
    return schemas.SourceType(value.value)


def to_domain_search_filters(filters: schemas.SearchFilters | None) -> SearchFilters | None:
    if filters is None:
        return None
    return SearchFilters(
        disease=_to_domain_disease(filters.disease),
        year_min=filters.year_min,
        year_max=filters.year_max,
        source_type=_to_domain_source_type(filters.source_type),
    )


def to_domain_search_request(request: schemas.SearchRequest) -> SearchRequest:
    return SearchRequest(
        query=request.query,
        top_k=request.top_k,
        cursor=request.cursor,
        filters=to_domain_search_filters(request.filters),
    )


def _to_api_chunk_section(section: ChunkSection) -> schemas.ChunkSection:
    return schemas.ChunkSection(chunk_id=section.chunk_id, section=section.section)


def _to_api_search_hit(hit: SearchHit) -> schemas.SearchHit:
    return schemas.SearchHit(
        chunk_ids=hit.chunk_ids,
        sections=[_to_api_chunk_section(s) for s in hit.sections],
        doc_id=hit.doc_id,
        doi=hit.doi,
        authors=hit.authors,
        journal=hit.journal,
        score=float(hit.score),
        year=hit.year,
        disease=_to_api_disease(hit.disease),
        source_type=_to_api_source_type(hit.source_type),
        title=hit.title,
        content_text=hit.content_text or "",
    )


def to_api_search_response(response: SearchResponse) -> schemas.SearchResponse:
    return schemas.SearchResponse(
        request_id=response.request_id,
        next_cursor=response.next_cursor,
        effective_embedding_model_id=response.effective_embedding_model_id,
        hits=[_to_api_search_hit(hit) for hit in response.hits],
    )
