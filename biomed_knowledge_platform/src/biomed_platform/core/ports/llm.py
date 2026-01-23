from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence
from biomed_platform.core.domains.llm import LlmChatMessage


class LlmCallError(Exception):
    def __init__(
        self,
        *,
        message: str,
        details: Mapping[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = dict(details) if details is not None else None
        self.retryable = retryable


class LlmClientPort(Protocol):
    async def chat(
        self,
        *,
        model_id: str,
        messages: Sequence[LlmChatMessage],
        options: Mapping[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError

    async def generate_text(
        self,
        *,
        model_id: str,
        prompt: str,
        options: Mapping[str, Any] | None = None,
        system: str | None = None,
    ) -> str:
        raise NotImplementedError
