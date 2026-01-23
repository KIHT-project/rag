from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import httpx

from biomed_platform.core.ports.llm import LlmCallError, LlmChatMessage, LlmClientPort


def _strip_trailing_slash(url: str) -> str:
    return url[:-1] if url.endswith("/") else url


def _ollama_chat_url(base_url: str) -> str:
    base = _strip_trailing_slash(base_url.strip())
    return f"{base}/api/chat"


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599


@dataclass(frozen=True)
class OllamaChatOptions:
    stream: bool = False


class OllamaLlmClient(LlmClientPort):
    def __init__(
        self,
        *,
        base_url: str,
        client: httpx.AsyncClient,
        max_retries: int,
        semaphore: asyncio.Semaphore,
    ) -> None:
        self._base_url = base_url
        self._client = client
        self._max_retries = max(0, int(max_retries))
        self._semaphore = semaphore

    async def chat(
        self,
        *,
        model_id: str,
        messages: Sequence[LlmChatMessage],
        options: Mapping[str, Any] | None = None,
    ) -> str:
        url = _ollama_chat_url(self._base_url)
        payload = self._build_chat_payload(model_id=model_id, messages=messages, options=options)
        return await self._chat_with_retries(url=url, payload=payload)

    def _build_chat_payload(
        self,
        *,
        model_id: str,
        messages: Sequence[LlmChatMessage],
        options: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }
        if options:
            payload["options"] = dict(options)
        return payload

    async def _chat_with_retries(self, *, url: str, payload: Mapping[str, Any]) -> str:
        attempt = 0
        last_error: Exception | None = None

        while attempt <= self._max_retries:
            attempt += 1
            try:
                response = await self._post(url=url, payload=payload)
                return self._handle_response(response=response, attempt=attempt)
            except (httpx.TimeoutException, httpx.RequestError) as e:
                last_error = e
                if self._should_retry(attempt=attempt):
                    continue
                raise self._map_transport_error(url=url, err=e) from e
            except LlmCallError as e:
                last_error = e
                if e.retryable and self._should_retry(attempt=attempt):
                    continue
                raise

        raise self._exhausted_retries_error(last_error=last_error)

    async def _post(self, *, url: str, payload: Mapping[str, Any]) -> httpx.Response:
        async with self._semaphore:
            return await self._client.post(url, json=dict(payload))

    def _handle_response(self, *, response: httpx.Response, attempt: int) -> str:
        if 200 <= response.status_code < 300:
            body = self._safe_json(response=response)
            return self._extract_message_content(body=body)

        if _is_retryable_status(response.status_code) and self._should_retry(attempt=attempt):
            raise LlmCallError(
                message="Retryable Ollama HTTP status",
                details={"status_code": response.status_code},
                retryable=True,
            )

        raise LlmCallError(
            message="Ollama chat request failed",
            details={"status_code": response.status_code, "body": response.text[:1000]},
            retryable=_is_retryable_status(response.status_code),
        )

    def _safe_json(self, *, response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except json.JSONDecodeError as e:
            raise LlmCallError(
                message="Invalid JSON from Ollama",
                details={"status_code": response.status_code},
                retryable=False,
            ) from e

        if not isinstance(body, dict):
            raise LlmCallError(
                message="Unexpected Ollama chat response type",
                details={"type": type(body).__name__},
                retryable=False,
            )
        return body

    def _extract_message_content(self, *, body: Mapping[str, Any]) -> str:
        msg = body.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]

        raise LlmCallError(
            message="Unexpected Ollama chat response shape",
            details={"keys": sorted([str(k) for k in body.keys()])},
            retryable=False,
        )

    def _should_retry(self, *, attempt: int) -> bool:
        return attempt <= self._max_retries

    def _map_transport_error(self, *, url: str, err: Exception) -> LlmCallError:
        if isinstance(err, httpx.TimeoutException):
            return LlmCallError(
                message="Ollama chat timeout",
                details={"url": url},
                retryable=True,
            )

        return LlmCallError(
            message="Ollama chat request error",
            details={"url": url, "type": type(err).__name__},
            retryable=True,
        )

    def _exhausted_retries_error(self, *, last_error: Exception | None) -> LlmCallError:
        return LlmCallError(
            message="Ollama chat failed after retries",
            details={"type": type(last_error).__name__ if last_error else "unknown"},
            retryable=True,
        )

    async def generate_text(
        self,
        *,
        model_id: str,
        prompt: str,
        options: Mapping[str, Any] | None = None,
        system: str | None = None,
    ) -> str:
        msgs: list[LlmChatMessage] = []
        if system:
            msgs.append(LlmChatMessage(role="system", content=system))
        msgs.append(LlmChatMessage(role="user", content=prompt))
        return await self.chat(model_id=model_id, messages=msgs, options=options)
