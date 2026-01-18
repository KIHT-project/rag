from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence

from biomed_platform.common.logging import get_logger
from biomed_platform.core.ports.ingestion import EmbeddingProvider

log = get_logger(__name__)


@dataclass(slots=True)
class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """
    In process embedding provider using sentence transformers.

    Properties
    - model cache per model_id
    - per model load lock to avoid duplicate downloads and excessive memory
    - embedding runs in a thread to avoid blocking the event loop
    """

    device: str = "cpu"
    normalize_embeddings: bool = False
    batch_size: int = 32

    def __post_init__(self) -> None:
        self._models: dict[str, object] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

        log.info(
            "SentenceTransformersEmbeddingProvider initialized, device=%s,"
            "normalize=%s, batch_size=%s",
            self.device,
            self.normalize_embeddings,
            self.batch_size,
        )

    async def embed_texts(self, *, model_id: str, texts: Sequence[str]) -> list[list[float]]:
        mid = (model_id or "").strip()
        if not mid:
            log.error("Embedding requested with empty model_id")
            raise ValueError("empty_model_id")

        if not texts:
            log.debug(
                "Embedding skipped, no texts provided, model_id=%s",
                mid,
            )
            return []

        log.debug(
            "Embedding request started, model_id=%s, text_count=%s",
            mid,
            len(texts),
        )

        model = await self._get_or_load_model(mid)

        clean: list[str] = []
        for t in texts:
            clean.append((t or "").strip())

        def _encode() -> list[list[float]]:
            # local import avoids import cost during unrelated code paths
            import numpy as np  # type: ignore

            vectors = model.encode(  # type: ignore[attr-defined]
                clean,
                batch_size=int(self.batch_size),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=bool(self.normalize_embeddings),
            )

            if isinstance(vectors, np.ndarray):
                return vectors.astype("float32").tolist()

            return [list(map(float, v)) for v in vectors]

        try:
            result = await asyncio.to_thread(_encode)
        except Exception:
            log.exception(
                "Embedding execution failed, model_id=%s, text_count=%s",
                mid,
                len(clean),
            )
            raise

        if len(result) != len(clean):
            log.warning(
                "Embedding count mismatch, model_id=%s, texts=%s, vectors=%s",
                mid,
                len(clean),
                len(result),
            )

        vector_size = len(result[0]) if result else 0
        log.debug(
            "Embedding completed, model_id=%s, vectors=%s, vector_size=%s",
            mid,
            len(result),
            vector_size,
        )

        return result

    async def _get_or_load_model(self, model_id: str) -> object:
        existing = self._models.get(model_id)
        if existing is not None:
            log.debug("Embedding model cache hit, model_id=%s", model_id)
            return existing

        log.debug("Embedding model cache miss, model_id=%s", model_id)

        async with self._global_lock:
            existing = self._models.get(model_id)
            if existing is not None:
                log.debug(
                    "Embedding model loaded while waiting on global lock, model_id=%s",
                    model_id,
                )
                return existing

            lock = self._locks.get(model_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[model_id] = lock

        async with lock:
            existing = self._models.get(model_id)
            if existing is not None:
                log.debug(
                    "Embedding model loaded while waiting on model lock, model_id=%s",
                    model_id,
                )
                return existing

            def _load() -> object:
                from sentence_transformers import SentenceTransformer  # type: ignore

                return SentenceTransformer(model_id, device=self.device)

            log.info(
                "Loading embedding model, model_id=%s, device=%s",
                model_id,
                self.device,
            )

            try:
                model = await asyncio.to_thread(_load)
            except Exception:
                log.exception(
                    "Failed to load embedding model, model_id=%s",
                    model_id,
                )
                raise

            self._models[model_id] = model

            log.info(
                "Embedding model loaded and cached, model_id=%s",
                model_id,
            )

            return model
