from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import biomed_platform.api.app as app_mod


class TestApiApp:
    def _settings(self) -> Any:
        class _Settings:
            logging_path = "/tmp/logging.yaml"

            def require_api(self) -> dict[str, str]:
                return {
                    "title": "Biomedical Knowledge Platform",
                    "description": "RAG based biomedical knowledge system",
                    "version": "0.1.0",
                }

            def require_rag(self) -> dict[str, Any]:
                return {
                    "embedding": {
                        "provider": "sentence_transformers",
                        "device": "cpu",
                        "normalize_embeddings": False,
                        "batch_size": 8,
                    },
                    "chunking": {
                        "chunk_size": 1200,
                        "overlap": 150,
                    },
                }

            def require_qdrant(self) -> dict[str, Any]:
                return {
                    "url": "http://qdrant:6333",
                    "api_key": None,
                    "timeout_seconds": 1.0,
                    "collection": {
                        "name_prefix": "docs",
                        "distance": "COSINE",
                    },
                }

        return _Settings()

    def _patch_runtime_deps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(app_mod, "QdrantClient", lambda **kwargs: object())
        monkeypatch.setattr(app_mod, "parse_distance", lambda s: s)

        class _FakeChunker:
            def __init__(self, *, chunk_size: int, overlap: int) -> None:
                self.chunk_size = chunk_size
                self.overlap = overlap

        class _FakeEmbedder:
            def __init__(self, *, device: str, normalize_embeddings: bool, batch_size: int) -> None:
                self.device = device
                self.normalize_embeddings = normalize_embeddings
                self.batch_size = batch_size

        class _FakeIndex:
            def __init__(self, *, client: Any, collection_name_prefix: str, distance: Any) -> None:
                self.client = client
                self.collection_name_prefix = collection_name_prefix
                self.distance = distance

        class _FakePipeline:
            def __init__(self, *, chunker: Any, embedder: Any, index: Any) -> None:
                self.chunker = chunker
                self.embedder = embedder
                self.index = index

        monkeypatch.setattr(app_mod, "SimpleCharChunker", _FakeChunker)
        monkeypatch.setattr(app_mod, "SentenceTransformersEmbeddingProvider", _FakeEmbedder)
        monkeypatch.setattr(app_mod, "QdrantVectorIndex", _FakeIndex)
        monkeypatch.setattr(app_mod, "DefaultIngestionPipeline", _FakePipeline)

    def test_given_settings_when_create_app_then_configures_logging_and_sets_app_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        configured: dict[str, Any] = {}
        installed: dict[str, Any] = {"error_handlers": False}

        def fake_load_settings() -> Any:
            return self._settings()

        def fake_configure_logging(path: str) -> None:
            configured["logging_path"] = path

        def fake_install_error_handlers(app: FastAPI) -> None:
            installed["error_handlers"] = True

        monkeypatch.setattr(app_mod, "load_settings", fake_load_settings)
        monkeypatch.setattr(app_mod, "configure_logging", fake_configure_logging)
        monkeypatch.setattr(app_mod, "install_error_handlers", fake_install_error_handlers)
        monkeypatch.setattr(app_mod, "get_logger", lambda _: type("L", (), {"info": lambda *a, **k: None})())
        self._patch_runtime_deps(monkeypatch)

        app = app_mod.create_app()

        assert isinstance(app, FastAPI)
        assert configured["logging_path"] == "/tmp/logging.yaml"
        assert installed["error_handlers"] is True

        assert app.title == "Biomedical Knowledge Platform"
        assert app.description == "RAG based biomedical knowledge system"
        assert app.version == "0.1.0"

        assert getattr(app.state, "settings") is not None
        assert getattr(app.state, "ingestion_service") is not None

    def test_given_create_app_when_called_then_registers_expected_routers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(app_mod, "load_settings", lambda: self._settings())
        monkeypatch.setattr(app_mod, "configure_logging", lambda _: None)
        monkeypatch.setattr(app_mod, "get_logger", lambda _: type("L", (), {"info": lambda *a, **k: None})())
        self._patch_runtime_deps(monkeypatch)

        app = app_mod.create_app()

        paths = {getattr(r, "path", None) for r in app.router.routes}
        assert "/health" in paths
        assert "/ready" in paths

    def test_given_create_app_when_called_then_adds_middlewares_in_expected_chain_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(app_mod, "load_settings", lambda: self._settings())
        monkeypatch.setattr(app_mod, "configure_logging", lambda _: None)
        monkeypatch.setattr(app_mod, "get_logger", lambda _: type("L", (), {"info": lambda *a, **k: None})())
        self._patch_runtime_deps(monkeypatch)

        app = app_mod.create_app()

        middleware_classes = [m.cls for m in app.user_middleware]

        assert app_mod.RequestContextMiddleware in middleware_classes
        assert app_mod.AccessLogMiddleware in middleware_classes

        assert middleware_classes.index(app_mod.RequestContextMiddleware) < middleware_classes.index(
            app_mod.AccessLogMiddleware
        )

    def test_given_load_settings_raises_when_create_app_then_propagates_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app_mod, "load_settings", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        with pytest.raises(RuntimeError, match="boom"):
            app_mod.create_app()

    def test_given_app_startup_when_testclient_enters_then_starts_workers_and_logs_startup(
            self,
            monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(app_mod, "load_settings", lambda: self._settings())
        monkeypatch.setattr(app_mod, "configure_logging", lambda _: None)
        self._patch_runtime_deps(monkeypatch)

        class _Logger:
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple[Any, ...]]] = []

            def info(self, msg: str, *args: Any) -> None:
                self.calls.append((msg, args))

        logger = _Logger()

        # Critical, app_mod.create_app uses module level app_mod.log, not get_logger
        monkeypatch.setattr(app_mod, "log", logger)

        created_workers: list[dict[str, Any]] = []
        run_calls: int = 0

        class _FakeWorker:
            def __init__(
                    self,
                    *,
                    queue: Any,
                    job_store: Any,
                    document_registry: Any,
                    payload_store: Any,
                    pipeline: Any,
            ) -> None:
                created_workers.append(
                    {
                        "queue": queue,
                        "job_store": job_store,
                        "document_registry": document_registry,
                        "payload_store": payload_store,
                        "pipeline": pipeline,
                    }
                )

            async def run_forever(self) -> None:
                nonlocal run_calls
                run_calls += 1
                await asyncio.sleep(0)

        monkeypatch.setattr(app_mod, "IngestionWorker", _FakeWorker)

        app = app_mod.create_app()
        with TestClient(app) as _:
            pass

        assert len(created_workers) == app_mod.WORKER_COUNT
        assert run_calls == app_mod.WORKER_COUNT

        assert any("API %s | version=%s | Description: %s" in c[0] for c in logger.calls)
        assert any(
            c[1] == ("Biomedical Knowledge Platform", "0.1.0", "RAG based biomedical knowledge system")
            for c in logger.calls
        )

        assert any("RAG runtime configured, qdrant_url=%s, embedding_provider=%s" in c[0] for c in logger.calls)
        assert any(c[1] == ("http://qdrant:6333", "sentence_transformers") for c in logger.calls)
