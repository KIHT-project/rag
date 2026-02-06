from __future__ import annotations

from biomed_platform.api.models.generated import schemas
from biomed_platform.core.domains.documents import ContentTextSource, DocumentFetchResponse
from biomed_platform.core.domains.ingestion import IngestBatchAccepted
from biomed_platform.core.domains.retrieval import (
    DocumentInfo,
    DocumentResponse,
    DoiListExpandedResponse,
    DoiListSimpleResponse,
    SourceType,
)


def _to_api_source_type(value: SourceType | None) -> schemas.SourceType | None:
    if value is None:
        return None
    return schemas.SourceType(value.value)


def _to_api_content_text_source(value: ContentTextSource) -> schemas.ContentTextSource:
    return schemas.ContentTextSource(value.value)


def _to_api_ingest(result: IngestBatchAccepted | None) -> schemas.IngestJobAcceptedResponse | None:
    if result is None:
        return None
    return schemas.IngestJobAcceptedResponse(
        job_id=result.job_id,
        state=schemas.State(result.state.value),
    )


def to_api_document_fetch_response(
    response: DocumentFetchResponse,
) -> schemas.DocumentFetchResponse:
    return schemas.DocumentFetchResponse(
        request_id=response.request_id,
        doi=response.doi,
        pmid=response.pmid,
        title=response.title,
        journal=response.journal,
        year=response.year,
        authors=response.authors,
        source_type=_to_api_source_type(response.source_type),
        content_text=response.content_text,
        content_text_source=_to_api_content_text_source(response.content_text_source),
        full_text_available=response.full_text_available,
        ingest=_to_api_ingest(response.ingest),
    )


def _to_api_chunk_sections(sections) -> list[schemas.ChunkSection]:
    return [schemas.ChunkSection(chunk_id=s.chunk_id, section=s.section) for s in sections]


def _to_api_document_info(info: DocumentInfo) -> schemas.DocumentInfo:
    return schemas.DocumentInfo(
        doc_id=info.doc_id,
        doi=info.doi,
        chunk_ids=info.chunk_ids,
        sections=_to_api_chunk_sections(info.sections),
        chunk_total=info.chunk_total,
        authors=info.authors,
        journal=info.journal,
        year=info.year,
        disease=schemas.Disease(info.disease.value) if info.disease else None,
        source_type=_to_api_source_type(info.source_type),
        title=info.title,
        content_text=info.content_text,
        updated_at=info.updated_at,
    )


def to_api_document_response(response: DocumentResponse) -> schemas.DocumentResponse:
    return schemas.DocumentResponse(
        request_id=response.request_id,
        doc_id=response.doc_id,
        doi=response.doi,
        chunk_ids=response.chunk_ids,
        sections=_to_api_chunk_sections(response.sections),
        chunk_total=response.chunk_total,
        authors=response.authors,
        journal=response.journal,
        year=response.year,
        disease=schemas.Disease(response.disease.value) if response.disease else None,
        source_type=_to_api_source_type(response.source_type),
        title=response.title,
        content_text=response.content_text,
        updated_at=response.updated_at,
    )


def to_api_doi_list_response(
    response: DoiListSimpleResponse | DoiListExpandedResponse,
) -> schemas.DoiListSimpleResponse | schemas.DoiListExpandedResponse:
    if isinstance(response, DoiListSimpleResponse):
        return schemas.DoiListSimpleResponse(
            request_id=response.request_id,
            dois=response.dois,
        )
    return schemas.DoiListExpandedResponse(
        request_id=response.request_id,
        documents=[_to_api_document_info(d) for d in response.documents],
    )
