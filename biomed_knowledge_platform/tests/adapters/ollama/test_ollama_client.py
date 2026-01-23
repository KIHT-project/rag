from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import httpx
import pytest

from biomed_platform.adapters.ollama.ollama_client import OllamaLlmClient
from biomed_platform.core.ports.llm import LlmChatMessage, LlmCallError



@dataclass
class _StubResponse:
    status_code: int
    body: object

    @property
    def text(self) -> str:
        return str(self.body)

    def json(self) -> object:
        return self.body


class _StubHttpxClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._queue: list[object] = []

    def queue(self, item: object) -> None:
        self._queue.append(item)

    async def post(self, url: str, *, json: dict) -> _StubResponse:
        self.calls.append((url, json))
        if not self._queue:
            raise RuntimeError("No queued response")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, _StubResponse)
        return item

class _BadJsonResponse(_StubResponse):
    def json(self) -> object:  # type: ignore[override]
        raise json.JSONDecodeError("bad", doc="x", pos=0)

@pytest.mark.asyncio
async def test_chat_builds_payload_and_returns_content() -> None:
    http = _StubHttpxClient()
    http.queue(
        _StubResponse(
            status_code=200,
            body={"message": {"role": "assistant", "content": "ok"}},
        )
    )

    client = OllamaLlmClient(
        base_url="http://localhost:11434/ollama/",
        client=http,  # type: ignore[arg-type]
        max_retries=0,
        semaphore=asyncio.Semaphore(1),
    )

    out = await client.chat(
        model_id="m",
        messages=[LlmChatMessage(role="user", content="hi")],
        options={"temperature": 0},
    )

    assert out == "ok"
    assert len(http.calls) == 1
    url, payload = http.calls[0]
    assert url.endswith("/api/chat")
    assert payload["model"] == "m"
    assert payload["stream"] is False
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["options"] == {"temperature": 0}


@pytest.mark.asyncio
async def test_chat_retries_on_timeout_then_succeeds() -> None:
    http = _StubHttpxClient()
    http.queue(httpx.TimeoutException("timeout"))
    http.queue(
        _StubResponse(
            status_code=200,
            body={"message": {"role": "assistant", "content": "ok"}},
        )
    )

    client = OllamaLlmClient(
        base_url="http://localhost:11434/ollama",
        client=http,  # type: ignore[arg-type]
        max_retries=1,
        semaphore=asyncio.Semaphore(1),
    )

    out = await client.chat(
        model_id="m",
        messages=[LlmChatMessage(role="user", content="hi")],
        options=None,
    )

    assert out == "ok"
    assert len(http.calls) == 2


@pytest.mark.asyncio
async def test_chat_raises_llm_call_error_after_retries_exhausted() -> None:
    http = _StubHttpxClient()
    http.queue(httpx.TimeoutException("timeout"))
    http.queue(httpx.TimeoutException("timeout"))

    client = OllamaLlmClient(
        base_url="http://localhost:11434/ollama",
        client=http,  # type: ignore[arg-type]
        max_retries=1,
        semaphore=asyncio.Semaphore(1),
    )

    with pytest.raises(LlmCallError) as exc:
        await client.chat(
            model_id="m",
            messages=[LlmChatMessage(role="user", content="hi")],
            options=None,
        )

    assert exc.value.retryable is True
    assert "timeout" in exc.value.message.lower()

@pytest.mark.asyncio
async def test_chat_retries_on_429_then_succeeds() -> None:
    http = _StubHttpxClient()
    http.queue(_StubResponse(status_code=429, body={"error": "rate limit"}))
    http.queue(_StubResponse(status_code=200, body={"message": {"role": "assistant", "content": "ok"}}))

    client = OllamaLlmClient(
        base_url="http://localhost:11434/ollama",
        client=http,  # type: ignore[arg-type]
        max_retries=1,
        semaphore=asyncio.Semaphore(1),
    )

    out = await client.chat(
        model_id="m",
        messages=[LlmChatMessage(role="user", content="hi")],
        options=None,
    )

    assert out == "ok"
    assert len(http.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [500, 502, 503, 504, 599])
async def test_chat_retries_on_5xx_then_succeeds(status_code: int) -> None:
    http = _StubHttpxClient()
    http.queue(_StubResponse(status_code=status_code, body={"error": "server"}))
    http.queue(_StubResponse(status_code=200, body={"message": {"role": "assistant", "content": "ok"}}))

    client = OllamaLlmClient(
        base_url="http://localhost:11434/ollama",
        client=http,  # type: ignore[arg-type]
        max_retries=1,
        semaphore=asyncio.Semaphore(1),
    )

    out = await client.chat(
        model_id="m",
        messages=[LlmChatMessage(role="user", content="hi")],
        options=None,
    )

    assert out == "ok"
    assert len(http.calls) == 2


@pytest.mark.asyncio
async def test_chat_non_retryable_http_status_raises_llm_call_error() -> None:
    http = _StubHttpxClient()
    http.queue(_StubResponse(status_code=400, body={"error": "bad request"}))

    client = OllamaLlmClient(
        base_url="http://localhost:11434/ollama",
        client=http,  # type: ignore[arg-type]
        max_retries=3,  # should not matter, 400 is not retryable
        semaphore=asyncio.Semaphore(1),
    )

    with pytest.raises(LlmCallError) as exc:
        await client.chat(
            model_id="m",
            messages=[LlmChatMessage(role="user", content="hi")],
            options=None,
        )

    assert exc.value.retryable is False
    assert "failed" in exc.value.message.lower()
    assert len(http.calls) == 1


@pytest.mark.asyncio
async def test_chat_invalid_json_raises_non_retryable_llm_call_error() -> None:
    http = _StubHttpxClient()
    http.queue(_BadJsonResponse(status_code=200, body="not json"))

    client = OllamaLlmClient(
        base_url="http://localhost:11434/ollama",
        client=http,  # type: ignore[arg-type]
        max_retries=0,
        semaphore=asyncio.Semaphore(1),
    )

    with pytest.raises(LlmCallError) as exc:
        await client.chat(
            model_id="m",
            messages=[LlmChatMessage(role="user", content="hi")],
            options=None,
        )

    assert exc.value.retryable is False
    assert "invalid json" in exc.value.message.lower()
    assert len(http.calls) == 1


@pytest.mark.asyncio
async def test_chat_unexpected_response_shape_raises_non_retryable_llm_call_error() -> None:
    http = _StubHttpxClient()
    http.queue(_StubResponse(status_code=200, body={"foo": "bar"}))

    client = OllamaLlmClient(
        base_url="http://localhost:11434/ollama",
        client=http,  # type: ignore[arg-type]
        max_retries=0,
        semaphore=asyncio.Semaphore(1),
    )

    with pytest.raises(LlmCallError) as exc:
        await client.chat(
            model_id="m",
            messages=[LlmChatMessage(role="user", content="hi")],
            options=None,
        )

    assert exc.value.retryable is False
    assert "unexpected" in exc.value.message.lower()
    assert len(http.calls) == 1


@pytest.mark.asyncio
async def test_chat_retries_on_request_error_then_succeeds() -> None:
    http = _StubHttpxClient()
    http.queue(httpx.RequestError("boom"))
    http.queue(_StubResponse(status_code=200, body={"message": {"role": "assistant", "content": "ok"}}))

    client = OllamaLlmClient(
        base_url="http://localhost:11434/ollama",
        client=http,  # type: ignore[arg-type]
        max_retries=1,
        semaphore=asyncio.Semaphore(1),
    )

    out = await client.chat(
        model_id="m",
        messages=[LlmChatMessage(role="user", content="hi")],
        options=None,
    )

    assert out == "ok"
    assert len(http.calls) == 2


@pytest.mark.asyncio
async def test_chat_request_error_after_retries_exhausted_raises_llm_call_error() -> None:
    http = _StubHttpxClient()
    http.queue(httpx.RequestError("boom"))
    http.queue(httpx.RequestError("boom"))

    client = OllamaLlmClient(
        base_url="http://localhost:11434/ollama",
        client=http,  # type: ignore[arg-type]
        max_retries=1,
        semaphore=asyncio.Semaphore(1),
    )

    with pytest.raises(LlmCallError) as exc:
        await client.chat(
            model_id="m",
            messages=[LlmChatMessage(role="user", content="hi")],
            options=None,
        )

    assert exc.value.retryable is True
    assert "request error" in exc.value.message.lower()
    assert len(http.calls) == 2


@pytest.mark.asyncio
async def test_chat_timeout_after_retries_exhausted_includes_url_in_details() -> None:
    http = _StubHttpxClient()
    http.queue(httpx.TimeoutException("timeout"))
    http.queue(httpx.TimeoutException("timeout"))

    client = OllamaLlmClient(
        base_url="http://localhost:11434/ollama",
        client=http,  # type: ignore[arg-type]
        max_retries=1,
        semaphore=asyncio.Semaphore(1),
    )

    with pytest.raises(LlmCallError) as exc:
        await client.chat(
            model_id="m",
            messages=[LlmChatMessage(role="user", content="hi")],
            options=None,
        )

    assert exc.value.retryable is True
    assert "timeout" in exc.value.message.lower()
    assert "url" in (exc.value.details or {})
    assert len(http.calls) == 2


@pytest.mark.asyncio
async def test_generate_text_builds_system_and_user_messages_and_delegates_to_chat() -> None:
    http = _StubHttpxClient()
    http.queue(_StubResponse(status_code=200, body={"message": {"role": "assistant", "content": "ok"}}))

    client = OllamaLlmClient(
        base_url="http://localhost:11434/ollama",
        client=http,  # type: ignore[arg-type]
        max_retries=0,
        semaphore=asyncio.Semaphore(1),
    )

    out = await client.generate_text(
        model_id="m",
        prompt="hello",
        system="sys",
        options={"temperature": 0},
    )

    assert out == "ok"
    assert len(http.calls) == 1
    _, payload = http.calls[0]
    assert payload["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    assert payload["options"] == {"temperature": 0}