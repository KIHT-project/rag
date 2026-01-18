from __future__ import annotations

import types

import pytest

from biomed_platform.core.services.ingestion.sentence_transformers_embedder import (
    SentenceTransformersEmbeddingProvider,
)


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        return [[1.0, 2.0] for _ in texts]


@pytest.mark.anyio
async def test_embed_texts_validates_model_id_and_handles_empty_texts() -> None:
    # Given, an embedder
    emb = SentenceTransformersEmbeddingProvider(device="cpu", normalize_embeddings=False, batch_size=2)

    # When, model id is empty
    with pytest.raises(ValueError):
        await emb.embed_texts(model_id=" ", texts=["x"])

    # Then, it rejects empty model id

    # When, texts are empty
    out = await emb.embed_texts(model_id="m", texts=[])

    # Then, it returns empty result
    assert out == []


@pytest.mark.anyio
async def test_embed_texts_caches_model_and_calls_encode(monkeypatch) -> None:
    # Given, a fake sentence transformers module
    fake = _FakeModel()

    st_mod = types.ModuleType("sentence_transformers")

    class _SentenceTransformer:
        def __init__(self, model_id: str, device: str = "cpu") -> None:
            self.model_id = model_id
            self.device = device

        def __call__(self, *args, **kwargs):
            return self

    def _factory(model_id: str, device: str = "cpu"):
        return fake

    st_mod.SentenceTransformer = _factory  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", st_mod)

    async def _to_thread(fn):
        return fn()

    monkeypatch.setattr(
        "biomed_platform.core.services.ingestion.sentence_transformers_embedder.asyncio.to_thread",
        _to_thread,
        raising=True,
    )

    emb = SentenceTransformersEmbeddingProvider(device="cpu", normalize_embeddings=True, batch_size=3)

    # When, embedding is called twice with same model id
    out1 = await emb.embed_texts(model_id="m1", texts=[" a ", "b"])
    out2 = await emb.embed_texts(model_id="m1", texts=["c"])

    # Then, model is loaded once, and encode was called with cleaned texts
    assert out1 == [[1.0, 2.0], [1.0, 2.0]]
    assert out2 == [[1.0, 2.0]]
    assert len(fake.calls) == 2
    assert fake.calls[0]["texts"] == ["a", "b"]
    assert fake.calls[0]["normalize_embeddings"] is True
