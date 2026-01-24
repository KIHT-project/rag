from __future__ import annotations

from typing import Any, Mapping, Sequence

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.hyde import HybridChunkCandidate, RetrievalOrigin
from biomed_platform.core.domains.retrieval import ChunkCandidate
from biomed_platform.core.ports.ingestion import EmbeddingProvider
from biomed_platform.core.ports.llm import LlmClientPort
from biomed_platform.core.ports.retrieval import VectorSearcher
from biomed_platform.core.services.hyde.hyde import generate_hypothetical_answer_document

log = get_logger(__name__)


def _merge_origin(*, existing: RetrievalOrigin, incoming: RetrievalOrigin) -> RetrievalOrigin:
    if existing is RetrievalOrigin.BOTH or incoming is RetrievalOrigin.BOTH:
        return RetrievalOrigin.BOTH
    if existing is not incoming:
        return RetrievalOrigin.BOTH
    return existing


def _sort_key(c: HybridChunkCandidate) -> tuple[float, str]:
    return (-float(c.score), str(c.chunk_id))


def union_dedupe_order_candidates(
    *,
    question: Sequence[ChunkCandidate],
    hyde: Sequence[ChunkCandidate],
) -> list[HybridChunkCandidate]:
    """Union, dedupe by chunk_id, keep best score, set origin.

    Deterministic ordering is guaranteed for deterministic inputs.
    """

    by_id: dict[str, HybridChunkCandidate] = {}

    def upsert(*, cand: ChunkCandidate, origin: RetrievalOrigin) -> None:
        cid = str(getattr(cand, "chunk_id", "") or "")
        if not cid:
            return

        incoming = HybridChunkCandidate(
            chunk_id=cid,
            doc_id=str(getattr(cand, "doc_id", "") or ""),
            doi=str(getattr(cand, "doi", "") or ""),
            title=getattr(cand, "title", None),
            year=getattr(cand, "year", None),
            section=getattr(cand, "section", None),
            source_type=getattr(cand, "source_type", None),
            score=float(getattr(cand, "score", 0.0) or 0.0),
            chunk_text=getattr(cand, "chunk_text", None),
            origin=origin,
        )

        existing = by_id.get(cid)
        if existing is None:
            by_id[cid] = incoming
            return

        merged_origin = _merge_origin(existing=existing.origin, incoming=origin)

        if incoming.score > existing.score:
            by_id[cid] = HybridChunkCandidate(
                chunk_id=incoming.chunk_id,
                doc_id=incoming.doc_id,
                doi=incoming.doi,
                title=incoming.title,
                year=incoming.year,
                section=incoming.section,
                source_type=incoming.source_type,
                score=incoming.score,
                chunk_text=incoming.chunk_text,
                origin=merged_origin,
            )
            return

        if incoming.score < existing.score:
            by_id[cid] = HybridChunkCandidate(
                chunk_id=existing.chunk_id,
                doc_id=existing.doc_id,
                doi=existing.doi,
                title=existing.title,
                year=existing.year,
                section=existing.section,
                source_type=existing.source_type,
                score=existing.score,
                chunk_text=existing.chunk_text,
                origin=merged_origin,
            )
            return

        # Equal score, keep existing fields to avoid churn, merge origin.
        if merged_origin is not existing.origin:
            by_id[cid] = HybridChunkCandidate(
                chunk_id=existing.chunk_id,
                doc_id=existing.doc_id,
                doi=existing.doi,
                title=existing.title,
                year=existing.year,
                section=existing.section,
                source_type=existing.source_type,
                score=existing.score,
                chunk_text=existing.chunk_text,
                origin=merged_origin,
            )

    for c in question:
        upsert(cand=c, origin=RetrievalOrigin.QUESTION)

    for c in hyde:
        upsert(cand=c, origin=RetrievalOrigin.HYDE)

    return sorted(by_id.values(), key=_sort_key)


async def embed_text(
    *,
    embedder: EmbeddingProvider,
    embedding_model_id: str,
    text: str,
) -> list[float]:
    texts = [text]
    vectors = await embedder.embed_texts(model_id=embedding_model_id, texts=texts)
    if not isinstance(vectors, list) or not vectors:
        raise ValueError("embedder returned empty response")
    vec = vectors[0]
    if not isinstance(vec, list) or not vec or not all(isinstance(x, (int, float)) for x in vec):
        raise ValueError("embedder returned invalid vector")
    return [float(x) for x in vec]


async def retrieve_hybrid_chunk_candidates(
    *,
    question: str,
    embedding_model_id: str,
    embedder: EmbeddingProvider,
    vector_searcher: VectorSearcher,
    top_k: int,
    qfilter: object | None,
    hyde_enabled: bool | None,
    hyde_llm: LlmClientPort,
    hyde_model_id: str,
    hyde_max_chars: int,
    hyde_llm_options: Mapping[str, Any] | None = None,
) -> list[HybridChunkCandidate]:
    q = (question or "").strip()
    if not q:
        raise ValueError("question must be non empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    log.info(
        "Hybrid retrieval started | embedding_model_id=%s | top_k=%s | hyde_enabled=%s",
        embedding_model_id,
        top_k,
        bool(hyde_enabled),
    )

    q_vec = await embed_text(embedder=embedder, embedding_model_id=embedding_model_id, text=q)
    q_hits = await vector_searcher.search_chunks(
        embedding_model_id=embedding_model_id,
        query_vector=q_vec,
        top_k=top_k,
        qfilter=qfilter,
    )

    hyde_hits: list[ChunkCandidate] = []
    hyde_text = await generate_hypothetical_answer_document(
        llm=hyde_llm,
        model_id=hyde_model_id,
        question=q,
        enabled=hyde_enabled,
        max_chars=hyde_max_chars,
        llm_options=hyde_llm_options,
    )
    if hyde_text is not None:
        h_vec = await embed_text(
            embedder=embedder, embedding_model_id=embedding_model_id, text=hyde_text
        )
        hyde_hits = await vector_searcher.search_chunks(
            embedding_model_id=embedding_model_id,
            query_vector=h_vec,
            top_k=top_k,
            qfilter=qfilter,
        )

    merged = union_dedupe_order_candidates(question=q_hits, hyde=hyde_hits)

    log.info(
        "Hybrid retrieval completed | question_hits=%s | hyde_hits=%s | merged=%s",
        len(q_hits),
        len(hyde_hits),
        len(merged),
    )

    return merged
