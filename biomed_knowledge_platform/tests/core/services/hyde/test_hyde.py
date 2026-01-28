from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from biomed_platform.core.ports.llm import LlmCallError
from biomed_platform.core.services.hyde import hyde as hyde_mod
from biomed_platform.core.services.hyde.hyde import generate_hypothetical_answer_document


@pytest.fixture(autouse=True)
def _reset_hyde_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hyde_mod,
        "_HYDE_CACHE",
        hyde_mod._HydeCache(max_entries=128),
        raising=True,
    )


class TestHydeService:
    @pytest.mark.asyncio
    async def test_hyde_enabled_calls_llm_once_and_returns_capped_text(self) -> None:
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value="  " + ("x" * 500) + "  ")
        model_id = "m1"
        question = "What causes neuropathic pain?"
        max_chars = 200

        text = await generate_hypothetical_answer_document(
            llm=llm,
            model_id=model_id,
            question=question,
            enabled=True,
            max_chars=max_chars,
        )

        assert text is not None
        assert text == "x" * max_chars
        assert llm.chat.await_count == 1

        kwargs = llm.chat.await_args.kwargs
        assert kwargs["model_id"] == model_id
        assert kwargs["options"]["temperature"] == 0
        assert len(kwargs["messages"]) == 1
        assert question in kwargs["messages"][0].content

    @pytest.mark.asyncio
    async def test_hyde_disabled_returns_none_and_does_not_call_llm(self) -> None:
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value="should not be called")

        text = await generate_hypothetical_answer_document(
            llm=llm,
            model_id="m1",
            question="q",
            enabled=None,
        )

        assert text is None
        assert llm.chat.await_count == 0

    @pytest.mark.asyncio
    async def test_question_empty_raises_value_error(self) -> None:
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value="irrelevant")

        with pytest.raises(ValueError, match="question must be non empty"):
            await generate_hypothetical_answer_document(
                llm=llm,
                model_id="m1",
                question="   ",
                enabled=True,
            )

        assert llm.chat.await_count == 0

    @pytest.mark.asyncio
    async def test_max_chars_zero_is_clamped_to_min_and_does_not_truncate_short_text(self) -> None:
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value="  abcdef  ")

        text = await generate_hypothetical_answer_document(
            llm=llm,
            model_id="m1",
            question="q",
            enabled=True,
            max_chars=0,
        )

        assert text == "abcdef"
        assert llm.chat.await_count == 1

    @pytest.mark.asyncio
    async def test_len_text_less_or_equal_max_chars_returns_text_without_truncation(self) -> None:
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value="  abc  ")

        text = await generate_hypothetical_answer_document(
            llm=llm,
            model_id="m1",
            question="q",
            enabled=True,
            max_chars=200,
        )

        assert text == "abc"
        assert llm.chat.await_count == 1

    @pytest.mark.asyncio
    async def test_llm_options_are_merged_into_options(self) -> None:
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value="ok")

        llm_options = {"temperature": 0.7, "num_ctx": 2048}

        _ = await generate_hypothetical_answer_document(
            llm=llm,
            model_id="m1",
            question="q",
            enabled=True,
            llm_options=llm_options,
        )

        kwargs = llm.chat.await_args.kwargs
        assert kwargs["options"]["temperature"] == 0.7
        assert kwargs["options"]["num_ctx"] == 2048

    @pytest.mark.asyncio
    async def test_llm_raises_llm_call_error_is_re_raised(self) -> None:
        llm = AsyncMock()
        llm.chat = AsyncMock(
            side_effect=LlmCallError(
                message="upstream",
                details={"x": "y"},
                retryable=True,
            )
        )

        with pytest.raises(LlmCallError) as excinfo:
            await generate_hypothetical_answer_document(
                llm=llm,
                model_id="m1",
                question="q",
                enabled=True,
            )

        err = excinfo.value
        assert err.message == "upstream"
        assert err.details == {"x": "y"}
        assert err.retryable is True
        assert llm.chat.await_count == 1

    @pytest.mark.asyncio
    async def test_llm_raises_unexpected_exception_is_wrapped_in_llm_call_error(self) -> None:
        llm = AsyncMock()
        llm.chat = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(LlmCallError) as excinfo:
            await generate_hypothetical_answer_document(
                llm=llm,
                model_id="m1",
                question="q",
                enabled=True,
            )

        err = excinfo.value
        assert err.message == "HyDE LLM call failed"
        assert err.details["error_type"] == "RuntimeError"
        assert err.retryable is False
        assert llm.chat.await_count == 1

    @pytest.mark.asyncio
    async def test_empty_llm_text_raises_llm_call_error(self) -> None:
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value="   ")

        with pytest.raises(LlmCallError) as excinfo:
            await generate_hypothetical_answer_document(
                llm=llm,
                model_id="m1",
                question="q",
                enabled=True,
            )

        err = excinfo.value
        assert err.message == "HyDE returned empty text"
        assert err.details["model_id"] == "m1"
        assert err.retryable is False
        assert llm.chat.await_count == 1
