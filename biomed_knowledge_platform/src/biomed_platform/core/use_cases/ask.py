from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.hyde import HybridChunkCandidate
from biomed_platform.core.domains.retrieval import ChunkCandidate, SearchFilters
from biomed_platform.core.domains.synthesis import AskResponseEnvelope
from biomed_platform.core.errors.errors import SystemError, business_error
from biomed_platform.core.ports.ingestion import EmbeddingProvider
from biomed_platform.core.ports.llm import LlmClientPort
from biomed_platform.core.ports.retrieval import VectorSearcher
from biomed_platform.core.services.hallucination.synthesis import _build_context
from biomed_platform.core.services.hallucination.synthesis import synthesize_answer
from biomed_platform.core.services.hyde.hyde import generate_hypothetical_answer_document
from biomed_platform.core.services.hyde.hybrid_retrieval import embed_text
from biomed_platform.core.services.hyde.hybrid_retrieval import union_dedupe_order_candidates
from biomed_platform.core.services.retrieval.hybrid_ranker import rerank_hybrid_candidates

log = get_logger(__name__)


_ALLOWED_FILTER_KEYS: set[str] = {
    "disease",
    "year_min",
    "year_max",
    "source_type",
    "doi",
    "doi_normalized",
    "year",
}

_STR_KEYS: set[str] = {"disease", "source_type", "doi", "doi_normalized"}
_INT_KEYS: set[str] = {"year_min", "year_max", "year"}


def _filters_from_domain(filters: SearchFilters) -> dict[str, object] | None:
    out: dict[str, object] = {}

    if filters.disease is not None:
        out["disease"] = str(filters.disease.value)
    if filters.source_type is not None:
        out["source_type"] = str(filters.source_type.value)
    if filters.year_min is not None:
        out["year_min"] = int(filters.year_min)
    if filters.year_max is not None:
        out["year_max"] = int(filters.year_max)

    return out or None


def _reject_unknown_filter_keys(filters: dict) -> None:
    unknown = sorted([str(k) for k in filters.keys() if str(k) not in _ALLOWED_FILTER_KEYS])
    if unknown:
        raise SystemError(
            code="validation_error",
            message="unknown filter keys",
            details={"unknown": unknown},
            retryable=False,
        )


def _coerce_str_filter(*, key: str, value: object) -> str | None:
    if key not in _STR_KEYS:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v if v else None
    return None


def _coerce_int_filter(*, key: str, value: object) -> int | None:
    if key not in _INT_KEYS:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    return None


def _filters_from_dict(filters: dict) -> dict[str, object] | None:
    _reject_unknown_filter_keys(filters)

    out: dict[str, object] = {}
    for k, v in filters.items():
        if v is None:
            continue

        key = str(k)

        s = _coerce_str_filter(key=key, value=v)
        if s is not None:
            out[key] = s
            continue

        i = _coerce_int_filter(key=key, value=v)
        if i is not None:
            out[key] = i
            continue

    return out or None


def _normalize_filters(filters: object | None) -> dict[str, object] | None:
    if filters is None:
        return None

    if isinstance(filters, SearchFilters):
        return _filters_from_domain(filters)

    if all(hasattr(filters, attr) for attr in ("disease", "source_type", "year_min", "year_max")):
        mapped = SearchFilters(
            disease=getattr(filters, "disease", None),
            source_type=getattr(filters, "source_type", None),
            year_min=getattr(filters, "year_min", None),
            year_max=getattr(filters, "year_max", None),
        )
        return _filters_from_domain(mapped)

    if isinstance(filters, dict):
        return _filters_from_dict(filters)

    raise SystemError(
        code="validation_error",
        message="filters must be a mapping or domain SearchFilters",
        details={"type": str(type(filters))},
        retryable=False,
    )


def _validate_question(*, question: str, max_chars: int | None) -> str:
    q = (question or "").strip()
    if not q:
        raise SystemError(
            code="validation_error",
            message="question must not be empty",
            details=None,
            retryable=False,
        )
    if max_chars is not None and int(max_chars) > 0 and len(q) > int(max_chars):
        raise SystemError(
            code="validation_error",
            message="question exceeds max length",
            details={"max_chars": int(max_chars), "length": len(q)},
            retryable=False,
        )
    return q


def _to_chunk_candidate(h: HybridChunkCandidate) -> ChunkCandidate:
    return ChunkCandidate(
        chunk_id=h.chunk_id,
        doc_id=h.doc_id,
        doi=h.doi,
        title=h.title,
        year=h.year,
        section=h.section,
        source_type=h.source_type,
        score=h.score,
        chunk_text=h.chunk_text,
    )


def _serialize_retrieval_candidate(
    *,
    candidate: ChunkCandidate | HybridChunkCandidate,
    selected_chunk_ids: set[str],
) -> dict[str, Any]:
    origin = getattr(candidate, "origin", None)
    origin_value = str(origin.value) if origin is not None and hasattr(origin, "value") else None
    chunk_id = str(getattr(candidate, "chunk_id", "") or "")
    return {
        "chunk_id": chunk_id,
        "doc_id": str(getattr(candidate, "doc_id", "") or ""),
        "doi": str(getattr(candidate, "doi", "") or ""),
        "pmid": str(getattr(candidate, "pmid", "") or ""),
        "title": getattr(candidate, "title", None),
        "year": getattr(candidate, "year", None),
        "section": getattr(candidate, "section", None),
        "source_type": getattr(candidate, "source_type", None),
        "score": float(getattr(candidate, "score", 0.0) or 0.0),
        "origin": origin_value,
        "selected_for_context": chunk_id in selected_chunk_ids,
    }


def _is_usable_for_context(c: HybridChunkCandidate) -> bool:
    return bool((c.chunk_text or "").strip())


def _select_chunks_for_context(
    *,
    ranked: Sequence[HybridChunkCandidate],
    max_chunks_final: int,
    max_context_chars: int,
) -> tuple[list[HybridChunkCandidate], set[str]]:
    selected: list[HybridChunkCandidate] = []

    for cand in ranked:
        if len(selected) >= int(max_chunks_final):
            break

        trial = [*selected, cand]
        _, included = _build_context(chunks=trial, max_chars=max_context_chars)

        if cand.chunk_id in included:
            selected = trial
            continue

    _, included_final = _build_context(chunks=selected, max_chars=max_context_chars)
    return selected, included_final


@dataclass(slots=True)
class AskUseCase:
    embedder: EmbeddingProvider
    vector_index: VectorSearcher
    llm: LlmClientPort
    hyde_generator: Callable[..., Any] = generate_hypothetical_answer_document
    synthesizer: Callable[..., Any] = synthesize_answer

    async def execute(
        self,
        *,
        request_id: str,
        question: str,
        filters: object | None,
        embedding_model_id: str,
        generator_model_id: str,
        hyde_model_id: str,
        hyde_enabled: bool,
        hyde_max_chars: int,
        ask_max_question_chars: int | None,
        ask_max_chunks_candidate: int,
        ask_max_chunks_final: int,
        ask_max_context_chars: int,
        ask_llm_max_retries: int,
        debug_enabled: bool = False,
    ) -> AskResponseEnvelope:
        start_ts = time.perf_counter()

        question_normalized = _validate_question(
            question=question,
            max_chars=ask_max_question_chars,
        )
        filters_normalized = _normalize_filters(filters)

        top_k = max(1, int(ask_max_chunks_candidate))
        log.info(
            "Ask retrieval started | embedding_model_id=%s | top_k=%s | hyde_enabled=%s",
            embedding_model_id,
            top_k,
            bool(hyde_enabled),
        )

        q_vec = await embed_text(
            embedder=self.embedder,
            embedding_model_id=embedding_model_id,
            text=question_normalized,
        )
        question_candidates = await self.vector_index.search_chunks(
            embedding_model_id=embedding_model_id,
            query_vector=q_vec,
            top_k=top_k,
            qfilter=filters_normalized,
        )

        hyde_candidates: list[ChunkCandidate] = []
        hyde_text = None
        if hyde_enabled:
            hyde_text = await self.hyde_generator(
                llm=self.llm,
                model_id=hyde_model_id,
                question=question_normalized,
                enabled=True,
                max_chars=int(hyde_max_chars),
                llm_options=None,
            )

        if hyde_text is not None:
            h_vec = await embed_text(
                embedder=self.embedder,
                embedding_model_id=embedding_model_id,
                text=hyde_text,
            )
            hyde_candidates = await self.vector_index.search_chunks(
                embedding_model_id=embedding_model_id,
                query_vector=h_vec,
                top_k=top_k,
                qfilter=filters_normalized,
            )

        merged_all = union_dedupe_order_candidates(
            question=question_candidates, hyde=hyde_candidates
        )
        merged_usable = [c for c in merged_all if _is_usable_for_context(c)]
        dropped_no_text = len(merged_all) - len(merged_usable)

        log.info(
            "Ask merge completed | question_hits=%s | hyde_hits=%s | merged=%s "
            "| usable=%s | dropped_no_text=%s",
            len(question_candidates),
            len(hyde_candidates),
            len(merged_all),
            len(merged_usable),
            dropped_no_text,
        )

        ranked_candidates = rerank_hybrid_candidates(
            question=question_normalized,
            question_candidates=question_candidates,
            hyde_candidates=hyde_candidates,
        )
        ranked_candidates = [c for c in ranked_candidates if _is_usable_for_context(c)]

        selected_hybrid_chunks, included_chunk_ids = _select_chunks_for_context(
            ranked=ranked_candidates,
            max_chunks_final=int(ask_max_chunks_final),
            max_context_chars=int(ask_max_context_chars),
        )

        log.info(
            "Ask selection completed | selected=%s | included=%s | max_context_chars=%s",
            len(selected_hybrid_chunks),
            len(included_chunk_ids),
            int(ask_max_context_chars),
        )

        if not selected_hybrid_chunks or not included_chunk_ids:
            raise business_error(
                code="validation_error",
                message="no_context_available",
                details=None,
            )

        selected_chunks: list[ChunkCandidate] = [
            _to_chunk_candidate(c) for c in selected_hybrid_chunks
        ]

        synthesis = await self.synthesizer(
            llm=self.llm,
            model_id=generator_model_id,
            question=question_normalized,
            selected_chunks=selected_chunks,
            max_context_chars=int(ask_max_context_chars),
            max_json_retries=int(ask_llm_max_retries),
            llm_options=None,
        )

        duration_ms = (time.perf_counter() - start_ts) * 1000.0

        debug: dict[str, Any] | None = None
        if debug_enabled:
            selected_chunk_ids = {c.chunk_id for c in selected_hybrid_chunks if c.chunk_id}
            debug = {
                "duration_ms": round(duration_ms, 2),
                "question_candidates": [
                    _serialize_retrieval_candidate(
                        candidate=c,
                        selected_chunk_ids=selected_chunk_ids,
                    )
                    for c in question_candidates
                ],
                "hyde_candidates": [
                    _serialize_retrieval_candidate(
                        candidate=c,
                        selected_chunk_ids=selected_chunk_ids,
                    )
                    for c in hyde_candidates
                ],
                "merged_candidates_all": [
                    _serialize_retrieval_candidate(
                        candidate=c,
                        selected_chunk_ids=selected_chunk_ids,
                    )
                    for c in merged_all
                ],
                "merged_candidates_usable": [
                    _serialize_retrieval_candidate(
                        candidate=c,
                        selected_chunk_ids=selected_chunk_ids,
                    )
                    for c in ranked_candidates
                ],
                "selected_candidates": [
                    _serialize_retrieval_candidate(
                        candidate=c,
                        selected_chunk_ids=selected_chunk_ids,
                    )
                    for c in selected_hybrid_chunks
                ],
                "selected_chunk_ids": [c.chunk_id for c in selected_hybrid_chunks],
                "included_chunk_ids": sorted(included_chunk_ids),
                "hyde_text": hyde_text,
            }

        return AskResponseEnvelope(
            request_id=request_id,
            effective_hyde_enabled=bool(hyde_enabled),
            answer=synthesis.answer,
            citations=synthesis.citations,
            debug=debug,
        )
