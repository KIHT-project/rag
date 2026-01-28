from __future__ import annotations

import asyncio
import json
import time
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


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except Exception:
            return None
    return None


def _as_str(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _ns_to_s(ns: int | None) -> float | None:
    if ns is None:
        return None
    if ns <= 0:
        return None
    return float(ns) / 1_000_000_000.0


def _safe_rate(tokens: int | None, seconds: float | None) -> float | None:
    if tokens is None or seconds is None:
        return None
    if tokens <= 0 or seconds <= 0:
        return None
    return float(tokens) / float(seconds)


def _pct(part: float | None, total: float | None) -> float | None:
    if part is None or total is None:
        return None
    if part <= 0 or total <= 0:
        return None
    return 100.0 * (part / total)


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

        timeout_s = float(timeout_seconds)
        if timeout_s <= 0:
            raise ValueError("timeout_seconds must be > 0")

        self._base_url = base_url.strip()
        self._max_retries = max(0, int(max_retries))
        self._semaphore = semaphore

        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))

        log.info(
            "OllamaLlmClient initialized | base_url=%s | max_retries=%s"
            "| timeout_seconds=%s | owns_client=%s",
            self._base_url,
            self._max_retries,
            timeout_s,
            bool(self._owns_client),
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
        payload = self._build_chat_payload(model_id=model_id, messages=messages, options=options)

        option_keys = sorted([str(k) for k in (dict(options or {}).keys())])
        log.info(
            "Ollama chat started | model_id=%s | messages=%s | option_keys=%s",
            model_id,
            len(messages),
            option_keys,
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

        opts = dict(options or {})
        fmt = opts.pop("format", None)

        payload: dict[str, Any] = {
            "model": mid,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": opts,
        }

        if isinstance(fmt, str) and fmt.strip():
            payload["format"] = fmt.strip()

        log.info(
            "Ollama payload built | model_id=%s | payload_keys=%s | options_keys=%s",
            mid,
            list(payload.keys()),
            sorted([str(k) for k in opts.keys()]),
        )
        return payload

    async def _chat_with_retries(self, *, url: str, payload: Mapping[str, Any]) -> str:
        last_error: Exception | None = None
        total_attempts = self._max_retries + 1

        for attempt_no in range(1, total_attempts + 1):
            log.info("Ollama request attempt %s/%s", attempt_no, total_attempts)

            try:
                response = await self._post(url=url, payload=payload)
                return self._handle_response(
                    response=response,
                    attempt_no=attempt_no,
                    total_attempts=total_attempts,
                    url=url,
                    model_id=_as_str(payload.get("model")),
                )

            except (httpx.TimeoutException, httpx.RequestError) as e:
                last_error = e
                log.warning(
                    "Ollama transport error | attempt=%s/%s | error_type=%s",
                    attempt_no,
                    total_attempts,
                    type(e).__name__,
                )
                if attempt_no < total_attempts:
                    continue
                raise self._map_transport_error(url=url, err=e) from e

            except LlmCallError as e:
                last_error = e
                log.warning(
                    "Ollama call error | attempt=%s/%s | retryable=%s | message=%s",
                    attempt_no,
                    total_attempts,
                    bool(e.retryable),
                    e.message,
                )
                if e.retryable and attempt_no < total_attempts:
                    continue
                raise

        log.error("Ollama retries exhausted")
        raise self._exhausted_retries_error(last_error=last_error)

    async def _post(self, *, url: str, payload: Mapping[str, Any]) -> httpx.Response:
        async with self._semaphore:
            log.info("Ollama HTTP POST %s", url)
            started = time.perf_counter()
            resp = await self._client.post(url, json=dict(payload))
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            log.info(
                "Ollama HTTP POST completed | status=%s | elapsed_ms=%s",
                resp.status_code,
                elapsed_ms,
            )
            return resp

    def _handle_response(
        self,
        *,
        response: httpx.Response,
        attempt_no: int,
        total_attempts: int,
        url: str,
        model_id: str,
    ) -> str:
        status = int(response.status_code)

        log.info(
            "Ollama HTTP response | status=%s | attempt=%s/%s",
            status,
            attempt_no,
            total_attempts,
        )

        if 200 <= status < 300:
            body = self._safe_json(response=response)
            self._log_perf(body=body, url=url, model_id=model_id)
            return self._extract_message_content(body=body)

        retryable = _is_retryable_status(status)
        if retryable and attempt_no < total_attempts:
            log.warning("Ollama retryable HTTP status %s", status)
            raise LlmCallError(
                message="Retryable Ollama HTTP status",
                details={"status_code": status},
                retryable=True,
            )

        log.error("Ollama non retryable HTTP status %s", status)
        raise LlmCallError(
            message="Ollama chat request failed",
            details={"status_code": status, "body": (response.text or "")[:1000]},
            retryable=retryable,
        )

    def _safe_json(self, *, response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
            log.info("Ollama response JSON decoded successfully")
        except json.JSONDecodeError as e:
            log.error("Ollama returned invalid JSON")
            raise LlmCallError(
                message="Invalid JSON from Ollama",
                details={"status_code": int(response.status_code)},
                retryable=False,
            ) from e

        if not isinstance(body, dict):
            log.error("Ollama response JSON is not an object | type=%s", type(body).__name__)
            raise LlmCallError(
                message="Unexpected Ollama chat response type",
                details={"type": type(body).__name__},
                retryable=False,
            )
        return body

    def _log_perf(self, *, body: Mapping[str, Any], url: str, model_id: str) -> None:
        total_ns = _as_int(body.get("total_duration"))
        load_ns = _as_int(body.get("load_duration"))
        prompt_ns = _as_int(body.get("prompt_eval_duration"))
        gen_ns = _as_int(body.get("eval_duration"))

        prompt_tokens = _as_int(body.get("prompt_eval_count"))
        gen_tokens = _as_int(body.get("eval_count"))

        total_s = _ns_to_s(total_ns)
        load_s = _ns_to_s(load_ns)
        prompt_s = _ns_to_s(prompt_ns)
        gen_s = _ns_to_s(gen_ns)

        prompt_tps = _safe_rate(prompt_tokens, prompt_s)
        gen_tps = _safe_rate(gen_tokens, gen_s)

        if all(
            v is None
            for v in (
                total_ns,
                load_ns,
                prompt_ns,
                gen_ns,
                prompt_tokens,
                gen_tokens,
            )
        ):
            return

        log.info(
            "Ollama perf | model_id=%s | url=%s | total_s=%s | load_s=%s |"
            "prompt_s=%s | gen_s=%s |prompt_tokens=%s | gen_tokens=%s |"
            "prompt_tps=%s | gen_tps=%s | load_pct=%s | prompt_pct=%s | gen_pct=%s",
            model_id or "na",
            url,
            f"{total_s:.3f}" if isinstance(total_s, float) else "na",
            f"{load_s:.3f}" if isinstance(load_s, float) else "na",
            f"{prompt_s:.3f}" if isinstance(prompt_s, float) else "na",
            f"{gen_s:.3f}" if isinstance(gen_s, float) else "na",
            prompt_tokens if prompt_tokens is not None else "na",
            gen_tokens if gen_tokens is not None else "na",
            f"{prompt_tps:.2f}" if isinstance(prompt_tps, float) else "na",
            f"{gen_tps:.2f}" if isinstance(gen_tps, float) else "na",
            f"{_pct(load_s, total_s):.1f}" if _pct(load_s, total_s) is not None else "na",
            f"{_pct(prompt_s, total_s):.1f}" if _pct(prompt_s, total_s) is not None else "na",
            f"{_pct(gen_s, total_s):.1f}" if _pct(gen_s, total_s) is not None else "na",
        )

    def _extract_message_content(self, *, body: Mapping[str, Any]) -> str:
        msg = body.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            content = msg["content"]
            log.info("Ollama extracted message content | length=%s", len(content))
            return content

        log.error(
            "Ollama unexpected response shape | keys=%s", sorted([str(k) for k in body.keys()])
        )
        raise LlmCallError(
            message="Unexpected Ollama chat response shape",
            details={"keys": sorted([str(k) for k in body.keys()])},
            retryable=False,
        )

    def _map_transport_error(self, *, url: str, err: Exception) -> LlmCallError:
        if isinstance(err, httpx.TimeoutException):
            log.error("Ollama timeout | url=%s", url)
            return LlmCallError(
                message="Ollama chat timeout",
                details={"url": url, "error_type": type(err).__name__},
                retryable=True,
            )

        log.error("Ollama transport error | url=%s | error_type=%s", url, type(err).__name__)
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
        log.info("Ollama generate_text called | model_id=%s", model_id)

        msgs: list[LlmChatMessage] = []
        if system:
            log.info("Ollama generate_text using system prompt")
            msgs.append(LlmChatMessage(role="system", content=system))
        msgs.append(LlmChatMessage(role="user", content=prompt))

        return await self.chat(model_id=model_id, messages=msgs, options=options)
