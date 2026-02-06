from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Sequence

from biomed_platform.common.logging import get_logger
from biomed_platform.common.utils import compute_doc_id, normalize_doi
from biomed_platform.core.domains.retrieval import (
    ChunkSection,
    Disease,
    DocumentInfo,
    DocumentResponse,
    DoiListExpandedResponse,
    DoiListSimpleResponse,
    SourceType,
    VectorSearchHit,
)
from biomed_platform.core.errors.errors import business_error
from biomed_platform.core.ports.ingestion import VectorWriter

log = get_logger(__name__)


def _parse_iso_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _select_updated_at(payloads: Iterable[dict[str, object]]) -> datetime:
    best: datetime | None = None
    for payload in payloads:
        ts = _parse_iso_ts(payload.get("updated_at") or payload.get("created_at"))
        if ts is not None and (best is None or ts > best):
            best = ts
    return best or datetime.now(timezone.utc)


def _extract_document_info(
    *,
    doc_id: str,
    hits: Sequence[VectorSearchHit],
) -> DocumentInfo:
    payloads = [dict(h.payload) for h in hits if isinstance(h.payload, dict)]
    payload_first = payloads[0] if payloads else {}

    doi = (
        str(payload_first.get("doi_original") or "")
        or str(payload_first.get("doi") or "")
        or str(payload_first.get("doi_normalized") or "")
    )

    chunk_parts: list[tuple[int, str, str]] = []
    section_parts: list[tuple[int, str, str | None]] = []
    section_fallback: list[tuple[str, str | None]] = []
    chunk_ids: list[str] = []
    for hit in hits:
        payload = dict(hit.payload) if isinstance(hit.payload, dict) else {}
        chunk_id = str(hit.point_id)
        text = payload.get("text")
        idx = payload.get("chunk_index")
        section_val = payload.get("section")
        section = str(section_val) if isinstance(section_val, str) else None
        if isinstance(idx, int) and isinstance(text, str):
            chunk_parts.append((idx, chunk_id, text))
            section_parts.append((idx, chunk_id, section))
        else:
            chunk_ids.append(chunk_id)
            section_fallback.append((chunk_id, section))

    chunk_parts.sort(key=lambda item: item[0])
    section_parts.sort(key=lambda item: item[0])
    chunk_ids = [cid for _, cid, _ in chunk_parts] + chunk_ids
    content_text = "\n".join(text for _, _, text in chunk_parts)
    sections = [ChunkSection(chunk_id=cid, section=section) for _, cid, section in section_parts]
    sections.extend(
        ChunkSection(chunk_id=cid, section=section) for cid, section in section_fallback
    )

    authors_val = payload_first.get("authors")
    authors = [str(a) for a in authors_val] if isinstance(authors_val, list) else None

    disease = None
    disease_raw = payload_first.get("disease")
    if isinstance(disease_raw, str) and disease_raw:
        try:
            disease = Disease(disease_raw)
        except Exception:
            disease = None

    source_type = None
    source_type_raw = payload_first.get("source_type")
    if isinstance(source_type_raw, str) and source_type_raw:
        try:
            source_type = SourceType(source_type_raw)
        except Exception:
            source_type = None

    updated_at = _select_updated_at(payloads)

    return DocumentInfo(
        doc_id=doc_id,
        doi=doi,
        chunk_ids=chunk_ids,
        sections=sections,
        chunk_total=len(chunk_ids),
        authors=authors,
        journal=payload_first.get("journal"),
        year=payload_first.get("year"),
        disease=disease,
        source_type=source_type,
        title=payload_first.get("title"),
        content_text=content_text,
        updated_at=updated_at,
    )


def _group_hits_by_doc_id(hits: Sequence[VectorSearchHit]) -> dict[str, list[VectorSearchHit]]:
    grouped: dict[str, list[VectorSearchHit]] = {}
    for hit in hits:
        payload = dict(hit.payload) if isinstance(hit.payload, dict) else {}
        doc_id = str(payload.get("doc_id") or "")
        if not doc_id:
            continue
        grouped.setdefault(doc_id, []).append(hit)
    return grouped


class DocumentLookupUseCase:
    def __init__(self, *, vector_index: VectorWriter) -> None:
        self._vector_index = vector_index

    async def get_by_doi(
        self,
        *,
        request_id: str,
        embedding_model_id: str,
        doi: str,
    ) -> DocumentResponse:
        doi_normalized = normalize_doi(doi)
        if not doi_normalized:
            raise business_error(
                code="validation_error",
                message="Invalid DOI",
                details={"doi": doi},
            )

        doc_id = compute_doc_id(doi_normalized=doi_normalized)
        hits = await self._vector_index.fetch_by_doc_ids(
            embedding_model_id=embedding_model_id,
            doc_ids=[doc_id],
            base_filter=None,
        )
        if not hits:
            raise business_error(
                code="not_found",
                message="DOI not found",
                details={"doi_normalized": doi_normalized},
            )

        info = _extract_document_info(doc_id=doc_id, hits=hits)
        return DocumentResponse(
            request_id=request_id,
            doc_id=info.doc_id,
            doi=info.doi,
            chunk_ids=info.chunk_ids,
            sections=info.sections,
            chunk_total=info.chunk_total,
            authors=info.authors,
            journal=info.journal,
            year=info.year,
            disease=info.disease,
            source_type=info.source_type,
            title=info.title,
            content_text=info.content_text,
            updated_at=info.updated_at,
        )

    async def list_dois(
        self,
        *,
        request_id: str,
        embedding_model_id: str,
        include_document_info: bool,
    ) -> DoiListSimpleResponse | DoiListExpandedResponse:
        hits = await self._vector_index.fetch_all(
            embedding_model_id=embedding_model_id,
        )

        grouped = _group_hits_by_doc_id(hits)

        if not include_document_info:
            dois = []
            for doc_hits in grouped.values():
                payload = (
                    dict(doc_hits[0].payload)
                    if doc_hits and isinstance(doc_hits[0].payload, dict)
                    else {}
                )
                doi = (
                    str(payload.get("doi_original") or "")
                    or str(payload.get("doi") or "")
                    or str(payload.get("doi_normalized") or "")
                )
                if doi:
                    dois.append(doi)

            dois_sorted = sorted(set(dois))
            return DoiListSimpleResponse(
                request_id=request_id,
                dois=dois_sorted,
            )

        documents = [
            _extract_document_info(doc_id=doc_id, hits=doc_hits)
            for doc_id, doc_hits in grouped.items()
        ]

        return DoiListExpandedResponse(
            request_id=request_id,
            documents=documents,
        )
