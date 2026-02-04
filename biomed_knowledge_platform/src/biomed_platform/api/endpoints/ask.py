from __future__ import annotations

from typing import Any, Mapping, Optional

from fastapi import APIRouter, Header, Request

from biomed_platform.api.models.generated import schemas
from biomed_platform.common.logging import get_logger
from biomed_platform.common.middleware.trace import get_request_id
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.use_cases.ask import AskUseCase


log = get_logger(__name__)

router = APIRouter(prefix="/v1/ask", tags=["Retrieval"])


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


def _get_optional_attr(obj: Any, name: str) -> Any:
    return getattr(obj, name, None)


@router.post("", response_model=schemas.AskResponseEnvelope, response_model_exclude_none=True)
async def ask(
    request: Request,
    payload: schemas.AskRequest,
    x_hyde_enabled: Optional[bool] = Header(default=None, alias="X-HyDE-Enabled"),
    x_debug_enabled: Optional[bool] = Header(default=None, alias="X-Debug-Enabled"),
):
    app = request.app
    settings = app.state.settings

    use_case = AskUseCase(
        embedder=app.state.embedding_provider,
        vector_index=app.state.vector_index,
        llm=app.state.llm_client,
    )

    llm_cfg = settings.require_llm()

    embedding_model_id = _require_embedding_model_id(settings=settings)
    generator_model_id = str(llm_cfg.get("generator_model_id", "")).strip()
    hyde_model_id = str(llm_cfg.get("hyde_model_id", "")).strip()

    hyde_max_chars = _require_int_from_llm_cfg(llm_cfg=llm_cfg, key="hyde_max_chars", default=1024)

    ask_max_chunks_candidate = _require_int_from_llm_cfg(
        llm_cfg=llm_cfg,
        key="ask_max_chunks_candidate",
        default=8,
    )
    ask_max_chunks_final = _require_int_from_llm_cfg(
        llm_cfg=llm_cfg,
        key="ask_max_chunks_final",
        default=8,
    )

    cfg_hyde_enabled = bool(llm_cfg.get("hyde_enabled", False))
    hyde_enabled = bool(x_hyde_enabled) if x_hyde_enabled is not None else cfg_hyde_enabled
    debug_enabled = bool(x_debug_enabled) if x_debug_enabled is not None else False

    log.info(
        "Ask chunk limits resolved | candidate_k=%s | final_k=%s | hyde_enabled=%s",
        int(ask_max_chunks_candidate),
        int(ask_max_chunks_final),
        bool(hyde_enabled),
    )

    synthesis_max_context_chars = _require_int_from_llm_cfg(
        llm_cfg=llm_cfg, key="ask_max_context_chars", default=24000
    )
    synthesis_max_retries = _require_int_from_llm_cfg(
        llm_cfg=llm_cfg, key="ask_llm_max_retries", default=1
    )

    ask_max_question_chars_raw = llm_cfg.get("ask_max_question_chars")
    if ask_max_question_chars_raw is None:
        ask_max_question_chars = None
    else:
        ask_max_question_chars = _require_int_from_llm_cfg(
            llm_cfg=llm_cfg,
            key="ask_max_question_chars",
            default=0,
        )
        if ask_max_question_chars <= 0:
            ask_max_question_chars = None

    filters = _get_optional_attr(payload, "filters")

    return await use_case.execute(
        request_id=get_request_id(),
        question=payload.question,
        filters=filters,
        embedding_model_id=embedding_model_id,
        generator_model_id=generator_model_id,
        hyde_model_id=hyde_model_id,
        hyde_enabled=bool(hyde_enabled),
        hyde_max_chars=int(hyde_max_chars),
        ask_max_question_chars=ask_max_question_chars,
        ask_max_chunks_candidate=int(ask_max_chunks_candidate),
        ask_max_chunks_final=int(ask_max_chunks_final),
        ask_max_context_chars=int(synthesis_max_context_chars),
        ask_llm_max_retries=int(synthesis_max_retries),
        debug_enabled=debug_enabled,
    )
