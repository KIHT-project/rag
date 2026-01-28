from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.llm import LlmChatMessage
from biomed_platform.core.ports.llm import LlmCallError, LlmClientPort
from biomed_platform.core.services.hyde.prompt_templates import HYDE_PROMPT_TEMPLATE

log = get_logger(__name__)

DEFAULT_MAX_CHARS = 512
DEFAULT_CACHE_TTL_SECONDS = 3600
DEFAULT_CACHE_MAX_ENTRIES = 2048

_MIN_MAX_CHARS = 128
_MAX_MAX_CHARS = 2000


def _cap_text(text: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _normalize_question(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return ""
    return " ".join(q.split())


def _qhash(question: str) -> str:
    qn = _normalize_question(question)
    return hashlib.sha256(qn.encode("utf-8")).hexdigest()[:12]


def _stable_options_hash(options: Mapping[str, Any]) -> str:
    allow = {
        "temperature",
        "num_predict",
        "stop",
        "top_k",
        "top_p",
        "seed",
    }
    picked: dict[str, Any] = {}
    for k in allow:
        if k in options:
            picked[k] = options[k]
    raw = json.dumps(picked, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _clamp_max_chars(max_chars: int) -> int:
    mc = int(max_chars)
    if mc < _MIN_MAX_CHARS:
        return _MIN_MAX_CHARS
    if mc > _MAX_MAX_CHARS:
        return _MAX_MAX_CHARS
    return mc


def _build_options(llm_options: Mapping[str, Any] | None) -> dict[str, Any]:
    base: dict[str, Any] = {
        "temperature": 0,
        "num_predict": 96,
        "stop": ["\n\n"],
    }
    if llm_options:
        base.update(dict(llm_options))
    return base


def _make_cache_key(
    *, model_id: str, question_norm: str, max_chars: int, options: Mapping[str, Any]
) -> tuple[str, str, str]:
    qh = _qhash(question_norm)
    oh = _stable_options_hash(options)
    prompt_version = "v1"
    key = f"{model_id}:{prompt_version}:{qh}:{max_chars}:{oh}"
    return key, qh, oh


def _log_start(
    *, model_id: str, qh: str, oh: str, qlen: int, max_chars: int, options: Mapping[str, Any]
) -> None:
    log.info(
        "HyDE generation started | model_id=%s | qh=%s | oh=%s |"
        "qlen=%s | max_chars=%s | option_keys=%s",
        model_id,
        qh,
        oh,
        qlen,
        max_chars,
        sorted([str(k) for k in options.keys()]),
    )


async def _call_hyde_llm(
    *, llm: LlmClientPort, model_id: str, question_norm: str, options: Mapping[str, Any]
) -> str:
    prompt = HYDE_PROMPT_TEMPLATE.format(question=question_norm)
    return await llm.chat(
        model_id=model_id,
        messages=[LlmChatMessage(role="user", content=prompt)],
        options=dict(options),
    )


def _clean_hyde_text(*, text: str | None, max_chars: int, model_id: str, qh: str, oh: str) -> str:
    cleaned = _cap_text((text or "").strip(), max_chars=max_chars)
    if cleaned:
        return cleaned
    raise LlmCallError(
        message="HyDE returned empty text",
        details={"model_id": model_id, "qh": qh, "oh": oh},
        retryable=False,
    )


def _wrap_unknown_error(e: Exception) -> LlmCallError:
    return LlmCallError(
        message="HyDE LLM call failed",
        details={"error_type": type(e).__name__},
        retryable=False,
    )


@dataclass(frozen=True)
class _HydeCacheEntry:
    text: str
    created_at_mono: float


class _HydeCache:
    def __init__(self, *, max_entries: int) -> None:
        self._by_key: OrderedDict[str, _HydeCacheEntry] = OrderedDict()
        self._max_entries = max(1, int(max_entries))
        self._lock = asyncio.Lock()
        self._in_flight: dict[str, asyncio.Future[str]] = {}

    @staticmethod
    def _consume_future_exception(fut: asyncio.Future[str]) -> None:
        if fut.cancelled():
            return
        try:
            fut.exception()
        except Exception:
            return

    def _now(self) -> float:
        return time.monotonic()

    def _is_expired(self, e: _HydeCacheEntry, ttl_seconds: int) -> bool:
        return (self._now() - float(e.created_at_mono)) > float(ttl_seconds)

    async def get(self, *, key: str, ttl_seconds: int) -> str | None:
        async with self._lock:
            e = self._by_key.get(key)
            if e is None:
                return None
            if self._is_expired(e, ttl_seconds):
                self._by_key.pop(key, None)
                return None
            self._by_key.move_to_end(key)
            return e.text

    async def put(self, *, key: str, text: str) -> None:
        async with self._lock:
            self._by_key[key] = _HydeCacheEntry(text=text, created_at_mono=self._now())
            self._by_key.move_to_end(key)
            while len(self._by_key) > self._max_entries:
                self._by_key.popitem(last=False)

    async def register_in_flight(self, *, key: str) -> tuple[asyncio.Future[str], bool]:
        async with self._lock:
            existing = self._in_flight.get(key)
            if existing is not None:
                return existing, False

            loop = asyncio.get_running_loop()
            fut: asyncio.Future[str] = loop.create_future()
            fut.add_done_callback(self._consume_future_exception)
            self._in_flight[key] = fut
            return fut, True

    async def resolve_in_flight(self, *, key: str, value: str) -> None:
        async with self._lock:
            fut = self._in_flight.pop(key, None)
        if fut is not None and not fut.done():
            fut.set_result(value)

    async def reject_in_flight(self, *, key: str, exc: BaseException) -> None:
        async with self._lock:
            fut = self._in_flight.pop(key, None)
        if fut is not None and not fut.done():
            fut.set_exception(exc)


_HYDE_CACHE = _HydeCache(max_entries=DEFAULT_CACHE_MAX_ENTRIES)


async def generate_hypothetical_answer_document(
    *,
    llm: LlmClientPort,
    model_id: str,
    question: str,
    enabled: bool | None,
    max_chars: int = DEFAULT_MAX_CHARS,
    llm_options: Mapping[str, Any] | None = None,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
) -> str | None:
    if enabled is not True:
        log.info("HyDE disabled, skipping generation")
        return None

    qn = _normalize_question(question)
    if not qn:
        raise ValueError("question must be non empty")

    mc = _clamp_max_chars(max_chars)
    options = _build_options(llm_options)

    key, qh, oh = _make_cache_key(
        model_id=model_id,
        question_norm=qn,
        max_chars=mc,
        options=options,
    )

    cached = await _HYDE_CACHE.get(key=key, ttl_seconds=int(cache_ttl_seconds))
    if cached is not None:
        log.info(
            "HyDE cache hit | model_id=%s | qh=%s | oh=%s | len=%s", model_id, qh, oh, len(cached)
        )
        return cached

    fut, is_owner = await _HYDE_CACHE.register_in_flight(key=key)
    if not is_owner:
        log.info("HyDE in flight wait | model_id=%s | qh=%s | oh=%s", model_id, qh, oh)
        return await fut

    started = time.monotonic()
    try:
        _log_start(model_id=model_id, qh=qh, oh=oh, qlen=len(qn), max_chars=mc, options=options)

        raw = await _call_hyde_llm(llm=llm, model_id=model_id, question_norm=qn, options=options)
        cleaned = _clean_hyde_text(text=raw, max_chars=mc, model_id=model_id, qh=qh, oh=oh)

        await _HYDE_CACHE.put(key=key, text=cleaned)
        await _HYDE_CACHE.resolve_in_flight(key=key, value=cleaned)

        elapsed_ms = int((time.monotonic() - started) * 1000)
        log.info(
            "HyDE generation completed | model_id=%s | qh=%s | oh=%s | len=%s | elapsed_ms=%s",
            model_id,
            qh,
            oh,
            len(cleaned),
            elapsed_ms,
        )
        return cleaned

    except asyncio.CancelledError as e:
        await _HYDE_CACHE.reject_in_flight(key=key, exc=e)
        raise

    except LlmCallError as e:
        await _HYDE_CACHE.reject_in_flight(key=key, exc=e)
        raise

    except Exception as e:
        err = _wrap_unknown_error(e)
        await _HYDE_CACHE.reject_in_flight(key=key, exc=err)
        raise err from e
