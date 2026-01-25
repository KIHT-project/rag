from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import httpx

from biomed_platform.common.logging import get_logger
from biomed_platform.core.ports.llm import LlmCallError, LlmChatMessage, LlmClientPort

log = get_logger(__name__)


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
        max_retries: int,
        semaphore: asyncio.Semaphore,
        timeout_seconds: float = 3600.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("base_url must be non empty")

        self._base_url = base_url.strip()
        self._max_retries = max(0, int(max_retries))
        self._semaphore = semaphore

        timeout_s = float(timeout_seconds)
        if timeout_s <= 0:
            raise ValueError("timeout_seconds must be > 0")

        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))

        log.info(
            f"OllamaLlmClient initialized | "
            f"base_url={self._base_url} | "
            f"max_retries={self._max_retries} | "
            f"timeout_seconds={timeout_s} | "
            f"owns_client={self._owns_client}"
        )

    async def aclose(self) -> None:
        if self._owns_client:
            log.info("Closing Ollama http client")
            await self._client.aclose()

    async def chat(
        self,
        *,
        model_id: str,
        messages: Sequence[LlmChatMessage],
        options: Mapping[str, Any] | None = None,
    ) -> str:
        url = _ollama_chat_url(self._base_url)
        payload = self._build_chat_payload(
            model_id=model_id,
            messages=messages,
            options=options,
        )

        log.info(
            f"Ollama chat started | model_id={model_id} | "
            f"messages={len(messages)} | options={options}"
        )

        return await self._chat_with_retries(url=url, payload=payload)

    def _build_chat_payload(
        self,
        *,
        model_id: str,
        messages: Sequence[LlmChatMessage],
        options: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        mid = (model_id or "").strip()
        if not mid:
            raise ValueError("model_id must be non empty")
        if not messages:
            raise ValueError("messages must be non empty")

        payload: dict[str, Any] = {
            "model": mid,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }

        if options:
            opts = dict(options)

            fmt = opts.pop("format", None)
            if isinstance(fmt, str) and fmt.strip():
                payload["format"] = fmt.strip()

            payload["options"] = opts

        log.info(f"Ollama payload built | model_id={mid} | payload_keys={list(payload.keys())}")
        return payload

    async def _chat_with_retries(self, *, url: str, payload: Mapping[str, Any]) -> str:
        attempt = 0
        last_error: Exception | None = None

        while attempt <= self._max_retries:
            attempt += 1
            log.info(f"Ollama request attempt {attempt}/{self._max_retries + 1}")

            try:
                response = await self._post(url=url, payload=payload)
                return self._handle_response(response=response, attempt=attempt)

            except (httpx.TimeoutException, httpx.RequestError) as e:
                last_error = e
                log.warning(
                    f"Ollama transport error | attempt={attempt} | "
                    f"error_type={type(e).__name__}"
                )
                if self._should_retry(attempt=attempt):
                    continue
                raise self._map_transport_error(url=url, err=e) from e

            except LlmCallError as e:
                last_error = e
                log.warning(
                    f"Ollama call error | attempt={attempt} | "
                    f"retryable={e.retryable} | message={e.message}"
                )
                if e.retryable and self._should_retry(attempt=attempt):
                    continue
                raise

        log.error("Ollama retries exhausted")
        raise self._exhausted_retries_error(last_error=last_error)

    async def _post(self, *, url: str, payload: Mapping[str, Any]) -> httpx.Response:
        async with self._semaphore:
            log.info(f"Ollama HTTP POST {url}")
            return await self._client.post(url, json=dict(payload))

    def _handle_response(self, *, response: httpx.Response, attempt: int) -> str:
        status = int(response.status_code)

        log.info(f"Ollama HTTP response | status={status} | attempt={attempt}")

        if 200 <= status < 300:
            body = self._safe_json(response=response)
            return self._extract_message_content(body=body)

        if _is_retryable_status(status) and self._should_retry(attempt=attempt):
            log.warning(f"Ollama retryable HTTP status {status}")
            raise LlmCallError(
                message="Retryable Ollama HTTP status",
                details={"status_code": status},
                retryable=True,
            )

        log.error(f"Ollama non retryable HTTP status {status}")
        raise LlmCallError(
            message="Ollama chat request failed",
            details={"status_code": status, "body": response.text[:1000]},
            retryable=_is_retryable_status(status),
        )

    def _safe_json(self, *, response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
            log.info("Ollama response JSON decoded successfully")
        except json.JSONDecodeError as e:
            log.error("Ollama returned invalid JSON")
            raise LlmCallError(
                message="Invalid JSON from Ollama",
                details={"status_code": response.status_code},
                retryable=False,
            ) from e

        if not isinstance(body, dict):
            log.error(f"Ollama response JSON is not an object | type={type(body).__name__}")
            raise LlmCallError(
                message="Unexpected Ollama chat response type",
                details={"type": type(body).__name__},
                retryable=False,
            )
        return body

    def _extract_message_content(self, *, body: Mapping[str, Any]) -> str:
        msg = body.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            log.info(f"Ollama extracted message content | length={len(msg['content'])}")
            return msg["content"]

        log.error(f"Ollama unexpected response shape | keys={list(body.keys())}")
        raise LlmCallError(
            message="Unexpected Ollama chat response shape",
            details={"keys": sorted([str(k) for k in body.keys()])},
            retryable=False,
        )

    def _should_retry(self, *, attempt: int) -> bool:
        return attempt <= self._max_retries

    def _map_transport_error(self, *, url: str, err: Exception) -> LlmCallError:
        if isinstance(err, httpx.TimeoutException):
            log.error(f"Ollama timeout | url={url}")
            return LlmCallError(
                message="Ollama chat timeout",
                details={"url": url, "error_type": type(err).__name__},
                retryable=True,
            )

        log.error(f"Ollama transport error | url={url} | error_type={type(err).__name__}")
        return LlmCallError(
            message="Ollama chat request error",
            details={"url": url, "error_type": type(err).__name__},
            retryable=True,
        )

    def _exhausted_retries_error(self, *, last_error: Exception | None) -> LlmCallError:
        return LlmCallError(
            message="Ollama chat failed after retries",
            details={"error_type": type(last_error).__name__ if last_error else "unknown"},
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
        log.info(f"Ollama generate_text called | model_id={model_id}")

        msgs: list[LlmChatMessage] = []
        if system:
            log.info("Ollama generate_text using system prompt")
            msgs.append(LlmChatMessage(role="system", content=system))
        msgs.append(LlmChatMessage(role="user", content=prompt))

        return await self.chat(model_id=model_id, messages=msgs, options=options)
