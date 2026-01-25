from __future__ import annotations

import time
from typing import Any, Mapping, Optional

from fastapi import APIRouter, Header, Request

from biomed_platform.api.models.generated import schemas
from biomed_platform.common.logging import get_logger
from biomed_platform.common.middleware.trace import get_request_id
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.services.hallucination.synthesis import synthesize_answer
from biomed_platform.core.services.hyde.hybrid_retrieval import retrieve_hybrid_chunk_candidates


log = get_logger(__name__)

router = APIRouter(prefix="/v1/ask", tags=["Dummy"])


def _require_embedding_model_id(*, settings: Any) -> str:
    rag_cfg = settings.require_rag()
    emb_cfg = rag_cfg.get("embedding", {})
    provider = emb_cfg.get("provider") if isinstance(emb_cfg, dict) else None
    model_id = str(provider or "").strip()
    if not model_id:
        raise SystemError(
            code="missing_embedding_model_id",
            message="Missing embedding model id, set rag.embedding.provider",
            details=None,
            retryable=False,
        )
    return model_id


def _require_int_from_llm_cfg(*, llm_cfg: Mapping[str, Any], key: str, default: int) -> int:
    raw = llm_cfg.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


@router.post("")
async def ask(
    request: Request,
    payload: schemas.AskRequest,
    x_hyde_enabled: Optional[bool] = Header(default=None, alias="X-HyDE-Enabled"),
):
    start_ts = time.perf_counter()
    app = request.app
    settings = app.state.settings

    llm = app.state.llm_client
    embedder = app.state.embedding_provider
    vector_index = app.state.vector_index

    llm_cfg = settings.require_llm()

    embedding_model_id = _require_embedding_model_id(settings=settings)
    generator_model_id = str(llm_cfg.get("generator_model_id", "")).strip()
    hyde_model_id = str(llm_cfg.get("hyde_model_id", "")).strip()

    hyde_max_chars = _require_int_from_llm_cfg(llm_cfg=llm_cfg, key="hyde_max_chars", default=1024)
    default_top_k = _require_int_from_llm_cfg(
        llm_cfg=llm_cfg, key="ask_max_chunks_candidate", default=8
    )

    hyde_enabled = (
        (x_hyde_enabled is True) if x_hyde_enabled is not None else (payload.hyde_enabled is True)
    )
    raw_top_k = payload.retrieval_top_k
    top_k = int(raw_top_k) if raw_top_k is not None else int(default_top_k)
    log.info(
        f"Ask top_k resolved | payload_retrieval_top_k={payload.retrieval_top_k} | "
        f"default_top_k={default_top_k} | effective_top_k={top_k}"
    )

    candidates = await retrieve_hybrid_chunk_candidates(
        question=payload.question,
        embedding_model_id=embedding_model_id,
        embedder=embedder,
        vector_searcher=vector_index,
        top_k=top_k,
        qfilter=payload.filters,
        hyde_enabled=hyde_enabled,
        hyde_llm=llm,
        hyde_model_id=hyde_model_id,
        hyde_max_chars=hyde_max_chars,
        hyde_llm_options=None,
    )

    synthesis_max_context_chars = _require_int_from_llm_cfg(
        llm_cfg=llm_cfg, key="ask_max_context_chars", default=24000
    )
    synthesis_max_retries = _require_int_from_llm_cfg(
        llm_cfg=llm_cfg, key="ask_llm_max_retries", default=1
    )

    synthesis = await synthesize_answer(
        llm=llm,
        model_id=generator_model_id,
        question=payload.question,
        selected_chunks=candidates,
        max_context_chars=synthesis_max_context_chars,
        max_json_retries=synthesis_max_retries,
        llm_options=None,
    )

    duration_ms = (time.perf_counter() - start_ts) * 1000.0

    return schemas.AskResponseEnvelope(
        request_id=get_request_id(),
        effective_embedding_model_id=embedding_model_id,
        effective_generator_model_id=generator_model_id,
        effective_hyde_enabled=hyde_enabled,
        effective_reranker_mode=schemas.RerankerMode.off,
        answer=synthesis.answer,
        citations=synthesis.citations,
        debug={
            "duration_ms": round(duration_ms, 2),
            "retrieved_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "doi": c.doi,
                    "score": c.score,
                    "origin": c.origin,
                    "chunk_text": c.chunk_text or "",
                }
                for c in candidates
            ],
        },
    )
