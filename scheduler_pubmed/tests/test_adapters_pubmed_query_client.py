from __future__ import annotations

from datetime import datetime
import json

import httpx
import pytest

from scheduler_pubmed.src.adapters.pubmed.query_client import (
    PubMedQueryClient,
    _extract_doi,
    _parse_pubmed_datetime,
)
from scheduler_pubmed.src.core.errors.errors import AppError


def _json_response(data: dict, *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(data).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


@pytest.mark.asyncio
async def test_search_returns_pubmed_results_with_dois() -> None:
    esearch_params: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/esearch.fcgi"):
            esearch_params.update(dict(request.url.params))
            return _json_response({"esearchresult": {"idlist": ["100", "200"]}})
        if request.url.path.endswith("/esummary.fcgi"):
            return _json_response(
                {
                    "result": {
                        "100": {
                            "articleids": [{"idtype": "doi", "value": "10.1000/abc"}],
                            "sortpubdate": "2026/02/08 10:30",
                        },
                        "200": {
                            "articleids": [{"idtype": "pubmed", "value": "200"}],
                            "pubdate": "2025",
                        },
                    }
                }
            )
        return _json_response({}, status_code=404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = PubMedQueryClient(client=client, base_url="https://example.test")
        results = await adapter.search(query="thrombosis")

    assert len(results) == 2
    assert results[0].pmid == "100"
    assert results[0].doi == "10.1000/abc"
    assert isinstance(results[0].published_at, datetime)
    assert results[1].doi is None
    assert esearch_params["reldate"] == "1"
    assert esearch_params["datetype"] == "pdat"


@pytest.mark.asyncio
async def test_search_accepts_reldate_override() -> None:
    esearch_params: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/esearch.fcgi"):
            esearch_params.update(dict(request.url.params))
            return _json_response({"esearchresult": {"idlist": []}})
        return _json_response({}, status_code=404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = PubMedQueryClient(client=client, base_url="https://example.test")
        results = await adapter.search(query="thrombosis", reldate_days=365)

    assert results == []
    assert esearch_params["reldate"] == "365"


@pytest.mark.asyncio
async def test_search_raises_on_error_status() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({}, status_code=500)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = PubMedQueryClient(client=client, base_url="https://example.test")
        with pytest.raises(AppError) as exc_info:
            await adapter.search(query="thrombosis")

    assert exc_info.value.code == "pubmed_unavailable"


def test_parse_pubmed_datetime_variants() -> None:
    assert _parse_pubmed_datetime(None) is None
    assert _parse_pubmed_datetime("") is None
    assert _parse_pubmed_datetime("not-a-date") is None
    assert _parse_pubmed_datetime("2026/02/08 10:30") is not None
    assert _parse_pubmed_datetime("2026/02/08") is not None
    assert _parse_pubmed_datetime("2026 Feb 08") is not None
    assert _parse_pubmed_datetime("2026 Feb") is not None
    assert _parse_pubmed_datetime("2026") is not None
    assert _parse_pubmed_datetime("2026xxxx") is not None


def test_extract_doi_handles_unexpected_articleids() -> None:
    assert _extract_doi({"articleids": "invalid"}) is None
    assert _extract_doi({"articleids": ["invalid"]}) is None
    assert _extract_doi({"articleids": [{"idtype": "pmid", "value": "1"}]}) is None
    assert (
        _extract_doi({"articleids": [{"idtype": "doi", "value": " 10.1000/xyz "}]})
        == "10.1000/xyz"
    )


@pytest.mark.asyncio
async def test_search_returns_empty_when_search_result_shape_is_invalid() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/esearch.fcgi"):
            return _json_response({"esearchresult": {"idlist": "invalid"}})
        return _json_response({}, status_code=404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = PubMedQueryClient(client=client, base_url="https://example.test")
        results = await adapter.search(query="thrombosis")

    assert results == []


@pytest.mark.asyncio
async def test_search_handles_missing_summary_for_one_pmid() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/esearch.fcgi"):
            return _json_response({"esearchresult": {"idlist": ["100", "200"]}})
        if request.url.path.endswith("/esummary.fcgi"):
            return _json_response({"result": {"100": {"articleids": []}}})
        return _json_response({}, status_code=404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = PubMedQueryClient(client=client, base_url="https://example.test")
        results = await adapter.search(query="thrombosis")

    assert len(results) == 2
    assert results[1].pmid == "200"
    assert results[1].doi is None
    assert results[1].published_at is None


@pytest.mark.asyncio
async def test_get_json_raises_on_invalid_json_payload() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=b"not-json",
            headers={"content-type": "application/json"},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = PubMedQueryClient(client=client, base_url="https://example.test")
        with pytest.raises(AppError) as exc_info:
            await adapter._get_json(path="/esearch.fcgi", params={})  # noqa: SLF001

    assert exc_info.value.code == "pubmed_invalid_response"


@pytest.mark.asyncio
async def test_get_json_raises_on_non_mapping_payload() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=json.dumps([1, 2, 3]).encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = PubMedQueryClient(client=client, base_url="https://example.test")
        with pytest.raises(AppError) as exc_info:
            await adapter._get_json(path="/esearch.fcgi", params={})  # noqa: SLF001

    assert exc_info.value.code == "pubmed_invalid_response"


@pytest.mark.asyncio
async def test_get_json_raises_on_request_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = PubMedQueryClient(client=client, base_url="https://example.test")
        with pytest.raises(AppError) as exc_info:
            await adapter._get_json(path="/esearch.fcgi", params={})  # noqa: SLF001

    assert exc_info.value.code == "pubmed_unavailable"


@pytest.mark.asyncio
async def test_fetch_summaries_returns_empty_when_result_missing() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/esummary.fcgi"):
            return _json_response({"result": "invalid"})
        return _json_response({}, status_code=404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = PubMedQueryClient(client=client, base_url="https://example.test")
        result = await adapter._fetch_summaries(pmids=["1"])  # noqa: SLF001

    assert result == {}


@pytest.mark.asyncio
async def test_search_pmids_returns_empty_when_esearchresult_is_not_mapping() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/esearch.fcgi"):
            return _json_response({"esearchresult": "invalid"})
        return _json_response({}, status_code=404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = PubMedQueryClient(client=client, base_url="https://example.test")
        result = await adapter._search_pmids(  # noqa: SLF001
            query="thrombosis",
            reldate_days=1,
        )

    assert result == []


@pytest.mark.asyncio
async def test_fetch_summaries_returns_empty_when_no_pmids() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: _json_response({}))
    ) as client:
        adapter = PubMedQueryClient(client=client, base_url="https://example.test")
        result = await adapter._fetch_summaries(pmids=[])  # noqa: SLF001

    assert result == {}
