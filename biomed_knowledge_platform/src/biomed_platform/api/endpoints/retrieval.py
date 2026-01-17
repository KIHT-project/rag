# src/biomed_platform/api/endpoints/retrieval.py
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response, status
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range

from biomed_platform.api.models.generated import schemas
from biomed_platform.common.logging import get_logger
from biomed_platform.common.middleware.trace import get_request_id
from biomed_platform.core.domains.retrieval import DocBest, ChunkPart
from biomed_platform.core.errors.errors import AppError, SystemError

log = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Retrieval"])

_TEXT_TRUNCATE_LIMIT = 2000


def _resolve_effective_embedding_model_id(*, request: Request) -> str:
    cfg = getattr(request.app.state, "settings", None)
    default_model_id: str | None = None

    if cfg is not None:
        rag_cfg = cfg.require_rag()
        emb_cfg = rag_cfg.get("embedding", {})
        if isinstance(emb_cfg, dict):
            default_model_id = emb_cfg.get("provider")

    effective = (default_model_id or "").strip()
    if not effective:
        raise SystemError(
            code="missing_embedding_model_id",
            message="Missing embedding model id, set rag.embedding.provider",
            details=None,
            retryable=False,
        )

    return effective


def _status_for_error_code(code: str) -> int:
    if code == "validation_error":
        return status.HTTP_400_BAD_REQUEST
    if code == "invalid_model_id":
        return status.HTTP_400_BAD_REQUEST
    if code == "too_many_requests":
        return status.HTTP_429_TOO_MANY_REQUESTS
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _to_error_response(*, request_id: str, err: AppError) -> schemas.ErrorResponse:
    try:
        code = schemas.Error(err.code)
    except Exception:
        code = schemas.Error.system_error

    return schemas.ErrorResponse(
        request_id=request_id,
        error=code,
        message=err.message,
        details=err.details,
    )


def _truncate_text(value: str) -> str:
    if len(value) <= _TEXT_TRUNCATE_LIMIT:
        return value
    return value[:_TEXT_TRUNCATE_LIMIT]


def _to_qdrant_filter(filters: schemas.SearchFilters | None) -> Filter | None:
    if filters is None:
        return None

    must: list[Any] = []

    if filters.disease is not None:
        must.append(FieldCondition(key="disease", match=MatchValue(value=filters.disease.value)))

    if filters.source_type is not None:
        must.append(
            FieldCondition(key="source_type", match=MatchValue(value=filters.source_type.value))
        )

    if filters.year_min is not None or filters.year_max is not None:
        must.append(
            FieldCondition(
                key="year",
                range=Range(
                    gte=int(filters.year_min) if filters.year_min is not None else None,
                    lte=int(filters.year_max) if filters.year_max is not None else None,
                ),
            )
        )

    if not must:
        return None

    return Filter(must=must)


def _assemble_full_text(parts: list[ChunkPart]) -> str:
    if not parts:
        return ""

    parts_sorted = sorted(parts, key=lambda p: (p.start, p.end, p.chunk_index))

    out_parts: list[str] = []
    cursor_end = 0

    for p in parts_sorted:
        text = p.text or ""
        if not text:
            continue

        s = int(p.start)
        e = int(p.end) if p.end >= 0 else s + len(text)

        if e <= cursor_end:
            continue

        if s < cursor_end:
            cut = cursor_end - s
            if cut < len(text):
                text = text[cut:]
            else:
                continue

        out_parts.append(text)
        cursor_end = e

    return "".join(out_parts)


@router.post(
    "/search",
    responses={
        200: {"model": schemas.SearchResponse},
        400: {"model": schemas.ErrorResponse},
        429: {"model": schemas.ErrorResponse},
    },
    summary="Vector search over stored chunks",
)
async def search_chunks(
    request: Request,
    response: Response,
    body: schemas.SearchRequest,
) -> schemas.SearchResponse | schemas.ErrorResponse:
    request_id = get_request_id()

    try:
        effective_embedding_model_id = _resolve_effective_embedding_model_id(request=request)

        embedder = getattr(request.app.state, "embedding_provider", None)
        if embedder is None:
            raise SystemError(
                code="service_not_configured",
                message="Embedding provider not configured",
                details=None,
                retryable=False,
            )

        index = getattr(request.app.state, "vector_index", None)
        if index is None:
            raise SystemError(
                code="service_not_configured",
                message="Vector index not configured",
                details=None,
                retryable=False,
            )

        top_k = int(getattr(body, "top_k", None) or 20)

        overfetch_factor = 8
        max_overfetch = 400
        chunk_limit = min(max_overfetch, top_k * overfetch_factor)

        query = (body.query or "").strip()
        if not query:
            raise SystemError(
                code="validation_error",
                message="query must not be empty",
                details=None,
                retryable=False,
            )

        vectors = await embedder.embed_texts(model_id=effective_embedding_model_id, texts=[query])
        if not vectors:
            raise SystemError(
                code="embedding_failed",
                message="Failed to embed query",
                details=None,
                retryable=True,
            )

        qfilter = _to_qdrant_filter(body.filters)

        raw_hits = await index.search(
            embedding_model_id=effective_embedding_model_id,
            query_vector=vectors[0],
            top_k=chunk_limit,
            qfilter=qfilter,
        )

        best_by_doc: dict[str, DocBest] = {}

        for h in raw_hits:
            payload = h.payload or {}
            doc_id = str(payload.get("doc_id") or "")
            if not doc_id:
                continue

            doi = str(payload.get("doi_original") or payload.get("doi") or "")
            title = payload.get("title")
            year = payload.get("year")
            source_type = payload.get("source_type")
            journal = payload.get("journal")
            authors_raw = payload.get("authors")

            parsed_journal = str(journal) if isinstance(journal, str) else None
            parsed_authors = _parse_authors(authors_raw)

            parsed_source_type: schemas.SourceType | None = None
            if isinstance(source_type, str):
                try:
                    parsed_source_type = schemas.SourceType(source_type)
                except Exception:
                    parsed_source_type = None

            candidate = DocBest(
                doc_id=doc_id,
                doi=doi,
                score=float(h.score),
                year=int(year) if isinstance(year, int) else None,
                source_type=parsed_source_type,
                title=str(title) if isinstance(title, str) else None,
                journal=parsed_journal,
                authors=parsed_authors,
            )

            existing = best_by_doc.get(doc_id)
            if existing is None or candidate.score > existing.score:
                best_by_doc[doc_id] = candidate

        selected_docs = sorted(best_by_doc.values(), key=lambda x: x.score, reverse=True)[:top_k]
        doc_ids = [d.doc_id for d in selected_docs if d.doc_id]

        assembled_text_by_doc: dict[str, str] = {}
        chunk_ids_by_doc: dict[str, list[str]] = {}

        fetch_fn = getattr(index, "fetch_by_doc_ids", None)
        if callable(fetch_fn) and doc_ids:
            all_chunks = await fetch_fn(
                embedding_model_id=effective_embedding_model_id,
                doc_ids=doc_ids,
                base_filter=qfilter,
                limit=20000,
            )

            parts_by_doc: dict[str, list[ChunkPart]] = {}

            for ch in all_chunks:
                payload = ch.payload or {}
                did = str(payload.get("doc_id") or "")
                if not did:
                    continue

                chunk_index = payload.get("chunk_index")
                chunk_start = payload.get("chunk_start")
                chunk_end = payload.get("chunk_end")
                text = str(payload.get("text") or "")

                idx = int(chunk_index) if isinstance(chunk_index, int) else 0
                start = int(chunk_start) if isinstance(chunk_start, int) else 0
                end = int(chunk_end) if isinstance(chunk_end, int) else -1

                parts_by_doc.setdefault(did, []).append(
                    ChunkPart(
                        chunk_id=str(ch.point_id),
                        chunk_index=idx,
                        start=start,
                        end=end,
                        text=text,
                    )
                )

            for did, parts in parts_by_doc.items():
                assembled_text_by_doc[did] = _assemble_full_text(parts)
                chunk_ids_by_doc[did] = [
                    p.chunk_id for p in sorted(parts, key=lambda p: p.chunk_index)
                ]

        hits: list[schemas.SearchHit] = []
        for d in selected_docs:
            full_text = assembled_text_by_doc.get(d.doc_id) if d.doc_id else None
            chunk_ids = chunk_ids_by_doc.get(d.doc_id) if d.doc_id else None

            if full_text:
                full_text = _truncate_text(full_text)

            hits.append(
                schemas.SearchHit(
                    chunk_ids=chunk_ids,
                    doc_id=d.doc_id,
                    doi=d.doi,
                    authors=d.authors,
                    journal=d.journal,
                    score=float(d.score),
                    year=d.year,
                    source_type=d.source_type,
                    title=d.title,
                    content_text=full_text,
                )
            )

        return schemas.SearchResponse(
            request_id=request_id,
            effective_embedding_model_id=effective_embedding_model_id,
            next_cursor=None,
            hits=hits,
        )

    except AppError as err:
        http_status = _status_for_error_code(err.code)
        response.status_code = http_status

        if err.code == "too_many_requests":
            details = err.details or {}
            retry_after = details.get("retry_after_seconds")
            if isinstance(retry_after, int):
                response.headers["Retry-After"] = str(retry_after)

        return _to_error_response(request_id=request_id, err=err)


def _parse_authors(value: object) -> list[str] | None:
    if value is None:
        return None

    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
                continue

            if isinstance(item, dict):
                root = item.get("root")
                if isinstance(root, str):
                    s = root.strip()
                    if s:
                        out.append(s)
                continue

            root = getattr(item, "root", None)
            if isinstance(root, str):
                s = root.strip()
                if s:
                    out.append(s)

        return out or None

    if isinstance(value, str):
        s = value.strip()
        return [s] if s else None

    root = getattr(value, "root", None)
    if isinstance(root, str):
        s = root.strip()
        return [s] if s else None

    return None
