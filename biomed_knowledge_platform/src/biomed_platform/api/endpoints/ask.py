from __future__ import annotations

import time
from typing import Any, Mapping, Optional

from fastapi import APIRouter, Header, Request

from biomed_platform.api.models.generated.schemas import AskRequest
from biomed_platform.common.middleware.trace import get_request_id
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.ports.llm import LlmChatMessage
from biomed_platform.core.services.hyde.hybrid_retrieval import retrieve_hybrid_chunk_candidates

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
    payload: AskRequest,
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
        llm_cfg=llm_cfg, key="ask_max_chunks_candidate", default=30
    )

    hyde_enabled = (
        (x_hyde_enabled is True) if x_hyde_enabled is not None else (payload.hyde_enabled is True)
    )
    top_k = int(payload.retrieval_top_k or default_top_k)

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

    context = "\n\n".join(c.chunk_text or "" for c in candidates if c.chunk_text)

    answer = await llm.chat(
        model_id=generator_model_id,
        messages=[
            LlmChatMessage(role="system", content="Answer concisely using the provided context."),
            LlmChatMessage(
                role="user",
                content=f"Question:\n{payload.question}\n\nContext:\n{context}",
            ),
        ],
        options={"temperature": 0},
    )

    duration_ms = (time.perf_counter() - start_ts) * 1000.0

    return {
        "request_id": get_request_id(),
        "duration_ms": round(duration_ms, 2),
        "effective_hyde_enabled": hyde_enabled,
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
        "answer": {
            "summary": answer,
            "risk_factors": [],
            "limitations": [],
        },
        "citations": [],
    }
