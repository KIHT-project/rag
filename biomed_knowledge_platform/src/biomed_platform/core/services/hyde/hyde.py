from __future__ import annotations

from typing import Any, Mapping

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.llm import LlmChatMessage
from biomed_platform.core.ports.llm import LlmCallError, LlmClientPort
from biomed_platform.core.services.hyde.prompt_templates import HYDE_PROMPT_TEMPLATE

log = get_logger(__name__)

DEFAULT_MAX_CHARS = 2000


def _cap_text(text: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


async def generate_hypothetical_answer_document(
    *,
    llm: LlmClientPort,
    model_id: str,
    question: str,
    enabled: bool | None,
    max_chars: int = DEFAULT_MAX_CHARS,
    llm_options: Mapping[str, Any] | None = None,
) -> str | None:
    if enabled is not True:
        log.info("HyDE disabled, skipping generation")
        return None

    q = (question or "").strip()
    if not q:
        log.error("HyDE invoked with empty question")
        raise ValueError("question must be non empty")

    # Log original prompt
    log.info(f"HyDE original prompt from request: {q}")

    prompt = HYDE_PROMPT_TEMPLATE.format(question=q)

    # Log HyDE prompt
    log.info(f"HyDE generated a prompt: {prompt}")

    options: dict[str, Any] = {
        "temperature": 0,
        "num_predict": 256,
        "stop": ["\n\n", "\nUser question", "\nRequired JSON"],
    }
    if llm_options:
        options.update(dict(llm_options))

    log.info(
        f"HyDE generation started | model_id={model_id} | "
        f"question_length={len(q)} | max_chars={max_chars} | "
        f"options={list(options.keys())}"
    )

    try:
        text = await llm.chat(
            model_id=model_id,
            messages=[LlmChatMessage(role="user", content=prompt)],
            options=options,
        )
    except LlmCallError as e:
        log.error(
            f"HyDE LLM call failed | model_id={model_id} | "
            f"retryable={e.retryable} | details={e.details}"
        )
        raise
    except Exception as e:
        log.error(
            f"HyDE LLM call failed with unexpected exception | "
            f"model_id={model_id} | error_type={type(e).__name__}",
            exc_info=True,
        )
        raise LlmCallError(
            message="HyDE LLM call failed",
            details={"error_type": type(e).__name__},
            retryable=False,
        ) from e

    raw_length = len(text) if isinstance(text, str) else 0
    cleaned = (text or "").strip()
    cleaned = _cap_text(cleaned, max_chars=max_chars)

    if not cleaned:
        log.error(
            f"HyDE returned empty output after cleanup | "
            f"model_id={model_id} | raw_length={raw_length}"
        )
        raise LlmCallError(
            message="HyDE returned empty text",
            details={"model_id": model_id},
            retryable=False,
        )

    if raw_length > len(cleaned):
        log.warning(
            f"HyDE output truncated | model_id={model_id} | "
            f"raw_length={raw_length} | final_length={len(cleaned)} | "
            f"max_chars={max_chars}"
        )

    log.info(f"HyDE generation completed | model_id={model_id} | " f"output_length={len(cleaned)}")

    return cleaned
