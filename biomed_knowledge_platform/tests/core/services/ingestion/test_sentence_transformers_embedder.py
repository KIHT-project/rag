# tests/core/services/ingestion/test_sentence_transformers_embedder.py
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Sequence

import pytest

from biomed_platform.core.services.ingestion.sentence_transformers_embedder import (
    SentenceTransformersEmbeddingProvider,
)


pytestmark = pytest.mark.asyncio


@dataclass
class _EncodeCall:
    clean: list[str]
    batch_size: int
    show_progress_bar: bool
    convert_to_numpy: bool
    normalize_embeddings: bool


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[_EncodeCall] = []
        self.return_value: Any = [[0.1, 0.2], [0.3, 0.4]]

    def encode(
        self,
        clean: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> Any:
        self.calls.append(
            _EncodeCall(
                clean=list(clean),
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
            )
        )
        return self.return_value


class _FakeNDArray:
    def __init__(self, data: list[list[float]]) -> None:
        self._data = data

    def astype(self, _: str) -> "_FakeNDArray":
        return self

    def tolist(self) -> list[list[float]]:
        return [list(row) for row in self._data]


def _install_fake_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_np = ModuleType("numpy")
    fake_np.ndarray = _FakeNDArray  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "numpy", fake_np)


async def _inline_to_thread(fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
    return fn(*args, **kwargs)


class TestSentenceTransformersEmbeddingProviderEmbedTexts:
    async def test_given_empty_model_id_when_embed_texts_then_raises_value_error(self) -> None:
        # Given
        provider = SentenceTransformersEmbeddingProvider()

        # When, Then
        with pytest.raises(ValueError, match="empty_model_id"):
            await provider.embed_texts(model_id="   ", texts=["a"])

    async def test_given_no_texts_when_embed_texts_then_returns_empty_list(self) -> None:
        # Given
        provider = SentenceTransformersEmbeddingProvider()

        # When
        got = await provider.embed_texts(model_id="m1", texts=[])

        # Then
        assert got == []

    async def test_given_texts_when_embed_texts_then_strips_texts_and_calls_encode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        provider = SentenceTransformersEmbeddingProvider(
            device="cpu",
            normalize_embeddings=True,
            batch_size=16,
        )
        model = _FakeModel()

        async def fake_get_or_load(mid: str) -> object:
            assert mid == "m1"
            return model

        monkeypatch.setattr(provider, "_get_or_load_model", fake_get_or_load)
        monkeypatch.setattr(asyncio, "to_thread", _inline_to_thread)

        # When
        got = await provider.embed_texts(model_id="  m1  ", texts=["  a  ", "", "  b"])

        # Then
        assert got == [[0.1, 0.2], [0.3, 0.4]]
        assert len(model.calls) == 1

        call = model.calls[0]
        assert call.clean == ["a", "", "b"]
        assert call.batch_size == 16
        assert call.show_progress_bar is False
        assert call.convert_to_numpy is True
        assert call.normalize_embeddings is True

    async def test_given_model_returns_numpy_ndarray_when_embed_texts_then_converts_to_lists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        _install_fake_numpy(monkeypatch)

        provider = SentenceTransformersEmbeddingProvider(batch_size=8, normalize_embeddings=False)
        model = _FakeModel()
        model.return_value = _FakeNDArray([[1.0, 2.0], [3.0, 4.0]])

        async def fake_get_or_load(_: str) -> object:
            return model

        monkeypatch.setattr(provider, "_get_or_load_model", fake_get_or_load)
        monkeypatch.setattr(asyncio, "to_thread", _inline_to_thread)

        # When
        got = await provider.embed_texts(model_id="m1", texts=["x", "y"])

        # Then
        assert got == [[1.0, 2.0], [3.0, 4.0]]
        assert len(model.calls) == 1

    async def test_given_embedding_thread_raises_when_embed_texts_then_reraises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        provider = SentenceTransformersEmbeddingProvider()
        model = _FakeModel()

        async def fake_get_or_load(_: str) -> object:
            return model

        async def failing_to_thread(fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

        monkeypatch.setattr(provider, "_get_or_load_model", fake_get_or_load)
        monkeypatch.setattr(asyncio, "to_thread", failing_to_thread)

        # When, Then
        with pytest.raises(RuntimeError, match="boom"):
            await provider.embed_texts(model_id="m1", texts=["x"])

    async def test_given_embedding_count_mismatch_when_embed_texts_then_returns_result_anyway(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        provider = SentenceTransformersEmbeddingProvider()
        model = _FakeModel()
        model.return_value = [[0.1, 0.2]]  # mismatch, 2 texts, 1 vector

        async def fake_get_or_load(_: str) -> object:
            return model

        monkeypatch.setattr(provider, "_get_or_load_model", fake_get_or_load)
        monkeypatch.setattr(asyncio, "to_thread", _inline_to_thread)

        # When
        got = await provider.embed_texts(model_id="m1", texts=["a", "b"])

        # Then
        assert got == [[0.1, 0.2]]


class TestSentenceTransformersEmbeddingProviderModelLoading:
    async def test_given_cached_model_when_get_or_load_model_then_returns_without_loading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        provider = SentenceTransformersEmbeddingProvider()
        sentinel = object()
        provider._models["m1"] = sentinel  # type: ignore[attr-defined]

        async def failing_to_thread(fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("should not load")

        monkeypatch.setattr(asyncio, "to_thread", failing_to_thread)

        # When
        got = await provider._get_or_load_model("m1")

        # Then
        assert got is sentinel

    async def test_given_two_concurrent_requests_when_get_or_load_model_then_loads_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        provider = SentenceTransformersEmbeddingProvider(device="cpu")

        created: list[tuple[str, str]] = []

        class _FakeSentenceTransformer:
            def __init__(self, model_id: str, device: str) -> None:
                created.append((model_id, device))

        fake_st = ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

        load_gate = asyncio.Event()

        async def gated_to_thread(fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
            if fn.__name__ == "_load":
                await load_gate.wait()
            return fn(*args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", gated_to_thread)

        t1 = asyncio.create_task(provider._get_or_load_model("m1"))
        t2 = asyncio.create_task(provider._get_or_load_model("m1"))

        await asyncio.sleep(0)
        load_gate.set()

        m1, m2 = await asyncio.gather(t1, t2)

        # Then
        assert m1 is m2
        assert created == [("m1", "cpu")]
        assert provider._models["m1"] is m1  # type: ignore[attr-defined]
