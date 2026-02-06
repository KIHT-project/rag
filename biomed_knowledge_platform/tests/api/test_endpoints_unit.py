from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request
from starlette.responses import Response

from biomed_platform.api.endpoints.ask import ask
from biomed_platform.api.endpoints.ingestion import _resolve_effective_embedding_model_id as _resolve_ingest_model
from biomed_platform.api.endpoints.ingestion import ingest_items
from biomed_platform.api.endpoints.documents import _resolve_effective_embedding_model_id as _resolve_delete_model
from biomed_platform.api.endpoints.documents import (
    delete_document,
    fetch_document,
    fetch_document_batch,
    get_document,
    list_dois,
)
from biomed_platform.api.endpoints.retrieval import _resolve_effective_embedding_model_id as _resolve_search_model
from biomed_platform.api.endpoints.retrieval import search
from biomed_platform.api.endpoints.system import health_check, readiness_check
from biomed_platform.api.models.generated import schemas
from biomed_platform.core.domains.readiness import (
    CheckStatus,
    ReadinessChecks,
    ReadinessResult,
    ReadinessStatus,
)
from biomed_platform.core.errors.errors import SystemError


class _Settings:
    def __init__(
        self,
        *,
        embedding_provider: str = "e",
        qdrant_url: str = "http://q",
        ollama_url: str = "http://o",
        postgres_url: str = "postgresql://user:pass@localhost:5432/db",
        llm_cfg: dict[str, object] | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._qdrant_url = qdrant_url
        self._ollama_url = ollama_url
        self._postgres_url = postgres_url
        self._llm_cfg = llm_cfg or {}

    def require_rag(self):
        return {"embedding": {"provider": self._embedding_provider}}

    def require_qdrant(self):
        return {"url": self._qdrant_url}

    def require_llm(self):
        cfg = {"ollama_base_url": self._ollama_url}
        cfg.update(self._llm_cfg)
        return cfg

    def require_postgres(self) -> dict[str, object]:
        return {
            "postgres_url": "postgresql+asyncpg://user:pass@localhost:5432/db",
        }


def _make_request(app) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [],
        "app": app,
    }
    return Request(scope)


def _make_ask_response(request_id: str) -> schemas.AskResponseEnvelope:
    return schemas.AskResponseEnvelope(
        request_id=request_id,
        effective_hyde_enabled=False,
        answer=schemas.AnswerPayload(summary="ok", risk_factors=[], limitations=[]),
        citations=[],
        debug=None,
    )

def _any_enum_member(enum_cls):
    return next(iter(enum_cls))

def _make_ingest_item(*, doi: str = "10.1/x") -> schemas.IngestItem:
    return schemas.IngestItem(
        doi=doi,
        disease=_any_enum_member(schemas.Disease),
        source_type=_any_enum_member(schemas.SourceType),
        content_text="some text",
    )

def _make_ingest_batch_request() -> schemas.IngestBatchRequest:
    return schemas.IngestBatchRequest(
        items=[_make_ingest_item()],
    )

def test_health_check() -> None:
    # Given, health endpoint
    # When
    out = health_check()
    # Then
    assert out == {"status": "ok"}


@pytest.mark.anyio
async def test_delete_document_endpoint_resolves_model_and_calls_use_case(monkeypatch) -> None:
    # Given
    app = SimpleNamespace(state=SimpleNamespace(settings=_Settings(embedding_provider="mdef")))
    req = _make_request(app)

    assert _resolve_delete_model(request=req) == "mdef"

    vector_index = AsyncMock()
    document_registry = AsyncMock()
    app.state.vector_index = vector_index
    app.state.document_registry = document_registry

    captured: dict[str, object] = {}

    class _UseCase:
        def __init__(self, *, vector_index, document_registry) -> None:
            self.vector_index = vector_index
            self.document_registry = document_registry

        async def execute(self, *, request_id: str, embedding_model_id: str, doi: str) -> None:
            captured["request_id"] = request_id
            captured["embedding_model_id"] = embedding_model_id
            captured["doi"] = doi

    monkeypatch.setattr(
        "biomed_platform.api.endpoints.documents.DeleteDocumentUseCase",
        _UseCase,
        raising=True,
    )
    monkeypatch.setattr(
        "biomed_platform.api.endpoints.documents.get_request_id",
        lambda: "rid",
        raising=True,
    )

    # When
    res = await delete_document(req, "10.1000/xyz123")

    # Then
    assert res.status_code == 204
    assert captured == {
        "request_id": "rid",
        "embedding_model_id": "mdef",
        "doi": "10.1000/xyz123",
    }


@pytest.mark.anyio
async def test_delete_document_endpoint_errors_when_missing_service() -> None:
    # Given
    app = SimpleNamespace(state=SimpleNamespace(settings=_Settings(embedding_provider="mdef")))
    req = _make_request(app)

    # When
    with pytest.raises(SystemError) as exc:
        await delete_document(req, "10.1000/xyz123")

    # Then
    assert exc.value.code == "service_not_configured"


@pytest.mark.anyio
async def test_get_document_endpoint_calls_use_case(monkeypatch) -> None:
    # Given
    app = SimpleNamespace(state=SimpleNamespace(settings=_Settings(embedding_provider="mdef")))
    req = _make_request(app)

    vector_index = AsyncMock()
    app.state.vector_index = vector_index

    captured: dict[str, object] = {}

    class _UseCase:
        def __init__(self, *, vector_index) -> None:
            self.vector_index = vector_index

        async def get_by_doi(self, *, request_id: str, embedding_model_id: str, doi: str):
            captured["request_id"] = request_id
            captured["embedding_model_id"] = embedding_model_id
            captured["doi"] = doi
            return schemas.DocumentResponse(
                request_id=request_id,
                doc_id="doc",
                doi=doi,
                chunk_ids=["c1"],
                sections=[schemas.ChunkSection(chunk_id="c1", section="Introduction")],
                chunk_total=1,
                authors=None,
                journal=None,
                year=None,
                source_type=None,
                title=None,
                content_text="text",
                updated_at="2026-01-18T14:32:05Z",
            )

    monkeypatch.setattr(
        "biomed_platform.api.endpoints.documents.DocumentLookupUseCase",
        _UseCase,
        raising=True,
    )
    monkeypatch.setattr(
        "biomed_platform.api.endpoints.documents.get_request_id",
        lambda: "rid",
        raising=True,
    )

    # When
    res = await get_document(req, "10.1000/xyz123")

    # Then
    assert res.request_id == "rid"
    assert res.doi == "10.1000/xyz123"
    assert captured == {
        "request_id": "rid",
        "embedding_model_id": "mdef",
        "doi": "10.1000/xyz123",
    }


@pytest.mark.anyio
async def test_list_dois_endpoint_calls_use_case(monkeypatch) -> None:
    # Given
    app = SimpleNamespace(state=SimpleNamespace(settings=_Settings(embedding_provider="mdef")))
    req = _make_request(app)

    vector_index = AsyncMock()
    app.state.vector_index = vector_index

    captured: dict[str, object] = {}

    class _UseCase:
        def __init__(self, *, vector_index) -> None:
            self.vector_index = vector_index

        async def list_dois(
            self,
            *,
            request_id: str,
            embedding_model_id: str,
            include_document_info: bool,
        ):
            captured["request_id"] = request_id
            captured["embedding_model_id"] = embedding_model_id
            captured["include_document_info"] = include_document_info
            return schemas.DoiListSimpleResponse(
                request_id=request_id,
                dois=["10.1/a"],
            )

    monkeypatch.setattr(
        "biomed_platform.api.endpoints.documents.DocumentLookupUseCase",
        _UseCase,
        raising=True,
    )
    monkeypatch.setattr(
        "biomed_platform.api.endpoints.documents.get_request_id",
        lambda: "rid",
        raising=True,
    )

    # When
    res = await list_dois(req, x_include_document_info=True)

    # Then
    assert res.request_id == "rid"
    assert captured == {
        "request_id": "rid",
        "embedding_model_id": "mdef",
        "include_document_info": True,
    }


@pytest.mark.anyio
async def test_fetch_document_endpoint_calls_use_case(monkeypatch) -> None:
    # Given
    app = SimpleNamespace(state=SimpleNamespace(settings=_Settings(embedding_provider="mdef")))
    req = _make_request(app)
    app.state.pubmed_client = object()

    captured: dict[str, object] = {}

    class _UseCase:
        def __init__(self, *, pubmed_client, ingestion_service):
            self.pubmed_client = pubmed_client
            self.ingestion_service = ingestion_service

        async def fetch_one(
            self,
            *,
            request_id: str,
            embedding_model_id: str,
            doi: str | None,
            pmid: str | None,
            ingest_enabled: bool,
        ):
            captured.update(
                {
                    "request_id": request_id,
                    "embedding_model_id": embedding_model_id,
                    "doi": doi,
                    "pmid": pmid,
                    "ingest_enabled": ingest_enabled,
                }
            )
            return schemas.DocumentFetchResponse(
                request_id=request_id,
                doi=doi,
                pmid=pmid,
                title=None,
                journal=None,
                year=None,
                authors=None,
                source_type=None,
                content_text="x",
                content_text_source=schemas.ContentTextSource.abstract,
                full_text_available=False,
                ingest=None,
            )

    monkeypatch.setattr(
        "biomed_platform.api.endpoints.documents.DocumentFetchUseCase",
        _UseCase,
        raising=True,
    )
    monkeypatch.setattr(
        "biomed_platform.api.endpoints.documents.get_request_id",
        lambda: "rid",
        raising=True,
    )

    # When
    body = schemas.DocumentFetchRequest(root=schemas.DocumentFetchByDoiRequest(doi="10.1/a"))
    res = await fetch_document(req, body, x_ingest_enabled=None)

    # Then
    assert res.request_id == "rid"
    assert captured == {
        "request_id": "rid",
        "embedding_model_id": "mdef",
        "doi": "10.1/a",
        "pmid": None,
        "ingest_enabled": True,
    }


@pytest.mark.anyio
async def test_fetch_document_batch_endpoint_returns_job(monkeypatch) -> None:
    # Given
    app = SimpleNamespace(state=SimpleNamespace(settings=_Settings(embedding_provider="mdef")))
    req = _make_request(app)
    pubmed_doc = SimpleNamespace(
        doi="10.1/a",
        pmid="123",
        pmcid=None,
        title="T",
        journal="J",
        year=2024,
        authors=["A"],
        mesh_terms=None,
        abstract="abstract",
        full_text=None,
    )

    class _PubMedClient:
        async def fetch_document(self, *, doi: str | None, pmid: str | None):
            return pubmed_doc

    app.state.pubmed_client = _PubMedClient()
    ingest_service = AsyncMock()
    ingest_service.ingest_batch = AsyncMock(
        return_value=SimpleNamespace(job_id="job1", state=schemas.State.queued)
    )
    app.state.ingestion_service = ingest_service

    monkeypatch.setattr(
        "biomed_platform.api.endpoints.documents.get_request_id",
        lambda: "rid",
        raising=True,
    )

    body = schemas.DocumentFetchBatchRequest(
        items=[schemas.DocumentFetchRequest(root=schemas.DocumentFetchByPmidRequest(pmid="123"))]
    )

    # When
    res = await fetch_document_batch(req, body, x_ingest_enabled=True)

    # Then
    assert res.request_id == "rid"
    assert res.job_id == "job1"


@pytest.mark.anyio
async def test_ingestion_endpoint_resolves_model_and_calls_service(monkeypatch) -> None:
    # Given, a request with settings default model id
    app = SimpleNamespace(state=SimpleNamespace(settings=_Settings(embedding_provider="mdef")))
    req = _make_request(app)

    body = _make_ingest_batch_request()

    # When, resolving effective model
    effective = _resolve_ingest_model(request=req, body=body)

    # Then
    assert effective == "mdef"

    # Given, ingestion service configured
    service = AsyncMock()
    service.ingest_batch = AsyncMock(return_value=SimpleNamespace(job_id="j", state="queued"))
    app.state.ingestion_service = service

    monkeypatch.setattr("biomed_platform.api.endpoints.ingestion.get_request_id", lambda: "rid", raising=True)

    # When
    _ = await ingest_items(req, body, idempotency_key=None)

    # Then
    service.ingest_batch.assert_awaited()



@pytest.mark.anyio
async def test_ingestion_endpoint_errors_when_missing_service() -> None:
    # Given
    app = SimpleNamespace(state=SimpleNamespace(settings=_Settings(embedding_provider="mdef")))
    req = _make_request(app)

    body = schemas.IngestBatchRequest(
        items=[
            schemas.IngestItem(
                doi="10.1/x",
                disease=schemas.Disease.thrombosis,
                source_type=schemas.SourceType.pubmed_abstract,
                content_text="x",
                year=None,
                title=None,
                journal=None,
                authors=None,
            )
        ],
    )

    # When
    with pytest.raises(SystemError) as exc:
        await ingest_items(req, body, idempotency_key=None)

    # Then
    assert exc.value.code == "service_not_configured"



@pytest.mark.anyio
async def test_search_endpoint_resolves_model_and_calls_use_case(monkeypatch) -> None:
    # Given
    app = SimpleNamespace(state=SimpleNamespace(settings=_Settings(embedding_provider="mdef")))
    req = _make_request(app)

    assert _resolve_search_model(request=req) == "mdef"

    use_case = AsyncMock()
    use_case.execute = AsyncMock(return_value=schemas.SearchResponse(request_id="rid", effective_embedding_model_id="mdef", next_cursor=None, hits=[]))
    app.state.search_use_case = use_case

    monkeypatch.setattr("biomed_platform.api.endpoints.retrieval.get_request_id", lambda: "rid", raising=True)

    # When
    out = await search(req, schemas.SearchRequest(query="q", top_k=1, filters=None))

    # Then
    assert out.request_id == "rid"
    use_case.execute.assert_awaited()


@pytest.mark.anyio
async def test_system_readiness_sets_status_code(monkeypatch) -> None:
    # Given
    app = SimpleNamespace(state=SimpleNamespace(settings=_Settings()))
    req = _make_request(app)
    resp = Response()

    ready = ReadinessResult(
        status=ReadinessStatus.ready,
        checks=ReadinessChecks(qdrant=CheckStatus.ok, llm=CheckStatus.ok, rdbms=CheckStatus.ok),
        errors=None,
    )

    monkeypatch.setattr("biomed_platform.api.endpoints.system.compute_readiness", AsyncMock(return_value=ready), raising=True)

    # When
    out = await readiness_check(req, resp)

    # Then
    assert resp.status_code == 200
    assert out.status.value == "ready"

    # Given, not ready
    not_ready = ReadinessResult(
        status=ReadinessStatus.not_ready,
        checks=ReadinessChecks(qdrant=CheckStatus.unhealthy, llm=CheckStatus.ok, rdbms=CheckStatus.ok),
        errors={"qdrant": {"reason": "http_5xx"}},
    )

    monkeypatch.setattr("biomed_platform.api.endpoints.system.compute_readiness", AsyncMock(return_value=not_ready), raising=True)
    resp2 = Response()

    # When
    _ = await readiness_check(req, resp2)

    # Then
    assert resp2.status_code == 503


@pytest.mark.anyio
async def test_ask_endpoint_respects_headers_and_llm_cfg(monkeypatch) -> None:
    # Given
    llm_cfg = {
        "generator_model_id": "gen",
        "hyde_model_id": "hyde",
        "hyde_enabled": True,
        "hyde_max_chars": "512",
        "ask_max_chunks_candidate": "5",
        "ask_max_chunks_final": "3",
        "ask_max_context_chars": "12000",
        "ask_llm_max_retries": "2",
        "ask_max_question_chars": "200",
    }
    app = SimpleNamespace(state=SimpleNamespace(settings=_Settings(embedding_provider="emb", llm_cfg=llm_cfg)))
    app.state.embedding_provider = AsyncMock()
    app.state.vector_index = AsyncMock()
    app.state.llm_client = AsyncMock()
    req = _make_request(app)

    captured: dict[str, object] = {}

    class _UseCase:
        def __init__(self, *, embedder, vector_index, llm) -> None:
            self.embedder = embedder
            self.vector_index = vector_index
            self.llm = llm

        async def execute(self, **kwargs):
            captured.update(kwargs)
            return _make_ask_response(kwargs["request_id"])

    monkeypatch.setattr("biomed_platform.api.endpoints.ask.AskUseCase", _UseCase, raising=True)
    monkeypatch.setattr("biomed_platform.api.endpoints.ask.get_request_id", lambda: "rid", raising=True)

    # When
    res = await ask(
        req,
        schemas.AskRequest(question="What is thrombosis?"),
        x_hyde_enabled=False,
        x_debug_enabled=True,
    )

    # Then
    assert res.request_id == "rid"
    assert captured["embedding_model_id"] == "emb"
    assert captured["generator_model_id"] == "gen"
    assert captured["hyde_model_id"] == "hyde"
    assert captured["hyde_enabled"] is False
    assert captured["debug_enabled"] is True
    assert captured["hyde_max_chars"] == 512
    assert captured["ask_max_chunks_candidate"] == 5
    assert captured["ask_max_chunks_final"] == 3
    assert captured["ask_max_context_chars"] == 12000
    assert captured["ask_llm_max_retries"] == 2
    assert captured["ask_max_question_chars"] == 200


@pytest.mark.anyio
async def test_ask_endpoint_disables_max_question_chars_when_zero(monkeypatch) -> None:
    # Given
    llm_cfg = {"ask_max_question_chars": 0}
    app = SimpleNamespace(state=SimpleNamespace(settings=_Settings(embedding_provider="emb", llm_cfg=llm_cfg)))
    app.state.embedding_provider = AsyncMock()
    app.state.vector_index = AsyncMock()
    app.state.llm_client = AsyncMock()
    req = _make_request(app)

    captured: dict[str, object] = {}

    class _UseCase:
        def __init__(self, *, embedder, vector_index, llm) -> None:
            self.embedder = embedder
            self.vector_index = vector_index
            self.llm = llm

        async def execute(self, **kwargs):
            captured.update(kwargs)
            return _make_ask_response(kwargs["request_id"])

    monkeypatch.setattr("biomed_platform.api.endpoints.ask.AskUseCase", _UseCase, raising=True)
    monkeypatch.setattr("biomed_platform.api.endpoints.ask.get_request_id", lambda: "rid", raising=True)

    # When
    _ = await ask(
        req,
        schemas.AskRequest(question="Q"),
        x_hyde_enabled=None,
        x_debug_enabled=None,
    )

    # Then
    assert captured["ask_max_question_chars"] is None
    assert captured["debug_enabled"] is False


@pytest.mark.anyio
async def test_ask_endpoint_raises_when_missing_embedding_model_id() -> None:
    # Given
    app = SimpleNamespace(state=SimpleNamespace(settings=_Settings(embedding_provider="")))
    app.state.embedding_provider = AsyncMock()
    app.state.vector_index = AsyncMock()
    app.state.llm_client = AsyncMock()
    req = _make_request(app)

    # When / Then
    with pytest.raises(SystemError) as exc:
        await ask(req, schemas.AskRequest(question="Q"))

    assert exc.value.code == "missing_embedding_model_id"
