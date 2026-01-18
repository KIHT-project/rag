from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request
from starlette.responses import Response

from biomed_platform.api.endpoints.ingestion import _resolve_effective_embedding_model_id as _resolve_ingest_model
from biomed_platform.api.endpoints.ingestion import ingest_items
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
    def __init__(self, *, embedding_provider: str = "e", qdrant_url: str = "http://q", ollama_url: str = "http://o") -> None:
        self._embedding_provider = embedding_provider
        self._qdrant_url = qdrant_url
        self._ollama_url = ollama_url

    def require_rag(self):
        return {"embedding": {"provider": self._embedding_provider}}

    def require_qdrant(self):
        return {"url": self._qdrant_url}

    def require_llm(self):
        return {"ollama_base_url": self._ollama_url}


def _make_request(app) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [],
        "app": app,
    }
    return Request(scope)

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
        embedding_model_id=None,
        items=[_make_ingest_item()],
    )

def test_health_check() -> None:
    # Given, health endpoint
    # When
    out = health_check()
    # Then
    assert out == {"status": "ok"}


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
        embedding_model_id="m",
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
        checks=ReadinessChecks(qdrant=CheckStatus.ok, llm=CheckStatus.ok),
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
        checks=ReadinessChecks(qdrant=CheckStatus.unhealthy, llm=CheckStatus.ok),
        errors={"qdrant": {"reason": "http_5xx"}},
    )

    monkeypatch.setattr("biomed_platform.api.endpoints.system.compute_readiness", AsyncMock(return_value=not_ready), raising=True)
    resp2 = Response()

    # When
    _ = await readiness_check(req, resp2)

    # Then
    assert resp2.status_code == 503


