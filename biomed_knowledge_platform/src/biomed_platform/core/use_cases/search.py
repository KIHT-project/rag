from __future__ import annotations

from dataclasses import dataclass

from biomed_platform.api.models.generated import schemas
from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.retrieval import ChunkPart, DocBest
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.ports.ingestion import EmbeddingProvider
from biomed_platform.core.ports.retrieval import ChunkStore, VectorSearcher
from biomed_platform.core.services.retrieval.hybrid_ranker import rerank_vector_hits

log = get_logger(__name__)


def _parse_author_item(item: object) -> str | None:
    if isinstance(item, str):
        s = item.strip()
        return s or None

    if isinstance(item, dict):
        root = item.get("root")
        if isinstance(root, str):
            s = root.strip()
            return s or None
        return None

    root = getattr(item, "root", None)
    if isinstance(root, str):
        s = root.strip()
        return s or None

    return None


def _parse_authors(value: object) -> list[str] | None:
    if value is None:
        return None

    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            parsed = _parse_author_item(item)
            if parsed:
                out.append(parsed)
        return out or None

    parsed = _parse_author_item(value)
    return [parsed] if parsed else None


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


def _to_filter_payload(filters: schemas.SearchFilters | None) -> dict[str, object] | None:
    if filters is None:
        return None

    out: dict[str, object] = {}

    if filters.disease is not None:
        out["disease"] = filters.disease.value

    if filters.source_type is not None:
        out["source_type"] = filters.source_type.value

    if filters.year_min is not None:
        out["year_min"] = int(filters.year_min)

    if filters.year_max is not None:
        out["year_max"] = int(filters.year_max)

    return out or None


def _select_best_docs(
    hits: list,
) -> dict[str, DocBest]:
    best_by_doc: dict[str, DocBest] = {}

    for h in hits:
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

    return best_by_doc


def _assemble_chunks(
    chunks: list,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[schemas.ChunkSection]]]:
    assembled_text_by_doc: dict[str, str] = {}
    chunk_ids_by_doc: dict[str, list[str]] = {}
    parts_by_doc: dict[str, list[ChunkPart]] = {}
    section_parts_by_doc: dict[str, list[tuple[int, str, str | None]]] = {}
    section_fallback_by_doc: dict[str, list[tuple[str, str | None]]] = {}

    for ch in chunks:
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
        section_val = payload.get("section")
        section = str(section_val) if isinstance(section_val, str) else None

        parts_by_doc.setdefault(did, []).append(
            ChunkPart(
                chunk_id=str(ch.point_id),
                chunk_index=idx,
                start=start,
                end=end,
                text=text,
            )
        )
        if isinstance(chunk_index, int):
            section_parts_by_doc.setdefault(did, []).append((idx, str(ch.point_id), section))
        else:
            section_fallback_by_doc.setdefault(did, []).append((str(ch.point_id), section))

    for did, parts in parts_by_doc.items():
        assembled_text_by_doc[did] = _assemble_full_text(parts)
        chunk_ids_by_doc[did] = [p.chunk_id for p in sorted(parts, key=lambda p: p.chunk_index)]

    sections_by_doc: dict[str, list[schemas.ChunkSection]] = {}
    for did, parts in section_parts_by_doc.items():
        sorted_parts = sorted(parts, key=lambda p: p[0])
        sections_by_doc[did] = [
            schemas.ChunkSection(chunk_id=cid, section=section) for _, cid, section in sorted_parts
        ]
    for did, fallbacks in section_fallback_by_doc.items():
        sections_by_doc.setdefault(did, []).extend(
            schemas.ChunkSection(chunk_id=cid, section=section) for cid, section in fallbacks
        )

    return assembled_text_by_doc, chunk_ids_by_doc, sections_by_doc


@dataclass(slots=True)
class SearchUseCase:
    embedder: EmbeddingProvider
    searcher: VectorSearcher
    chunks: ChunkStore

    async def execute(
        self,
        *,
        request_id: str,
        embedding_model_id: str,
        req: schemas.SearchRequest,
    ) -> schemas.SearchResponse:
        top_k = int(getattr(req, "top_k", None) or 20)

        overfetch_factor = 8
        max_overfetch = 400
        chunk_limit = min(max_overfetch, top_k * overfetch_factor)

        query = (req.query or "").strip()
        if not query:
            raise SystemError(
                code="validation_error",
                message="query must not be empty",
                details=None,
                retryable=False,
            )

        vectors = await self.embedder.embed_texts(
            model_id=embedding_model_id,
            texts=[query],
        )
        if not vectors:
            raise SystemError(
                code="embedding_failed",
                message="Failed to embed query",
                details=None,
                retryable=True,
            )

        qfilter = _to_filter_payload(req.filters)

        raw_hits = await self.searcher.search(
            embedding_model_id=embedding_model_id,
            query_vector=vectors[0],
            top_k=chunk_limit,
            qfilter=qfilter,
        )
        raw_hits = rerank_vector_hits(query=query, hits=raw_hits)

        best_by_doc = _select_best_docs(raw_hits)

        selected_docs = sorted(
            best_by_doc.values(),
            key=lambda x: x.score,
            reverse=True,
        )[:top_k]

        doc_ids = [d.doc_id for d in selected_docs if d.doc_id]

        assembled_text_by_doc: dict[str, str] = {}
        chunk_ids_by_doc: dict[str, list[str]] = {}
        sections_by_doc: dict[str, list[schemas.ChunkSection]] = {}

        if doc_ids:
            all_chunks = await self.chunks.fetch_by_doc_ids(
                embedding_model_id=embedding_model_id,
                doc_ids=doc_ids,
                base_filter=qfilter,
                limit=20000,
            )

            assembled_text_by_doc, chunk_ids_by_doc, sections_by_doc = _assemble_chunks(all_chunks)

        hits: list[schemas.SearchHit] = []
        for d in selected_docs:
            full_text = assembled_text_by_doc.get(d.doc_id) if d.doc_id else None
            chunk_ids = chunk_ids_by_doc.get(d.doc_id) if d.doc_id else None
            sections = sections_by_doc.get(d.doc_id) if d.doc_id else None

            hits.append(
                schemas.SearchHit(
                    chunk_ids=chunk_ids or [],
                    sections=sections or [],
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
            effective_embedding_model_id=embedding_model_id,
            next_cursor=None,
            hits=hits,
        )
