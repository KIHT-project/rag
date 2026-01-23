from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, Request

from biomed_platform.core.ports.llm import LlmChatMessage
from biomed_platform.api.models.generated.schemas import AskRequest


router = APIRouter(prefix="/v1/ask", tags=["Dummy"])


@router.post("")
async def ask_dummy(
    request: Request,
    payload: AskRequest,
    x_hyde_enabled: Optional[bool] = Header(default=False),
):
    llm = request.app.state.llm_client

    llm_cfg = request.app.state.settings.require_llm()
    model_id = str(llm_cfg.get("generator_model_id", "")).strip()

    text = await llm.chat(
        model_id=model_id,
        messages=[
            LlmChatMessage(
                role="user",
                content=f"Answer briefly in one sentence: {payload.question}",
            )
        ],
        options={"temperature": 0},
    )

    return {
        "request_id": request.headers.get("X-Request-Id", "dummy"),
        "effective_hyde_enabled": bool(x_hyde_enabled),
        "answer": {
            "summary": text,
            "risk_factors": [],
            "limitations": ["Dummy endpoint, no retrieval performed"],
        },
        "citations": [],
    }
