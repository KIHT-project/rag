from __future__ import annotations

import json

import httpx
import pytest

from scheduler_pubmed.src.adapters.rag.documents_client import RagDocumentsClient
from scheduler_pubmed.src.core.errors.errors import AppError


def _json_response(data: dict, *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(data).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def _build_client(handler) -> RagDocumentsClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    return RagDocumentsClient(
        client=http_client,
        base_url="https://rag.example.test",
        documents_get_path="/v1/documents/",
        documents_post_batch_path="/v1/documents/fetch/batch",
        ingest_jobs_get_path="/v1/ingest/jobs/",
    )


def _build_request_error_adapter() -> RagDocumentsClient:
    class _FailingClient:
        async def request(self, *args, **kwargs):
            request = httpx.Request("GET", "https://rag.example.test/fail")
            raise httpx.ConnectError("boom", request=request)

        async def get(self, *args, **kwargs):
            request = httpx.Request("GET", "https://rag.example.test/fail")
            raise httpx.ConnectError("boom", request=request)

        async def aclose(self):
            return None

    return RagDocumentsClient(
        client=_FailingClient(),  # type: ignore[arg-type]
        base_url="https://rag.example.test",
        documents_get_path="/v1/documents/",
        documents_post_batch_path="/v1/documents/fetch/batch",
        ingest_jobs_get_path="/v1/ingest/jobs/",
    )


@pytest.mark.asyncio
async def test_document_exists_true_and_false() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/10.1000/exists"):
            return _json_response({"doc_id": "x"}, status_code=200)
        if request.url.path.endswith("/10.1000/missing"):
            return _json_response({"error": "not_found"}, status_code=404)
        return _json_response({}, status_code=500)

    adapter = _build_client(handler)
    try:
        assert await adapter.document_exists(doi="10.1000/exists") is True
        assert await adapter.document_exists(doi="10.1000/missing") is False
    finally:
        await adapter._client.aclose()  # noqa: SLF001


@pytest.mark.asyncio
async def test_fetch_batch_and_ingest_job_status() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/v1/documents/fetch/batch"):
            payload = json.loads(request.content.decode("utf-8"))
            assert payload == {"items": [{"doi": "10.1000/new"}]}
            return _json_response(
                {"request_id": "r1", "job_id": "job-1", "state": "queued"},
                status_code=202,
            )
        if request.url.path.endswith("/v1/ingest/jobs/job-1"):
            return _json_response(
                {
                    "job_id": "job-1",
                    "state": "partial",
                    "items": [
                        {"doi": "10.1000/new", "state": "succeeded"},
                    ],
                }
            )
        return _json_response({}, status_code=404)

    adapter = _build_client(handler)
    try:
        accepted = await adapter.fetch_batch(dois=["10.1000/new"])
        status = await adapter.get_ingest_job_status(job_id="job-1")
    finally:
        await adapter._client.aclose()  # noqa: SLF001

    assert accepted.job_id == "job-1"
    assert accepted.state == "queued"
    assert status.state == "partial"
    assert len(status.items) == 1
    assert status.items[0].doi == "10.1000/new"


@pytest.mark.asyncio
async def test_fetch_batch_raises_on_non_202() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"error": "boom"}, status_code=500)

    adapter = _build_client(handler)
    try:
        with pytest.raises(AppError) as exc_info:
            await adapter.fetch_batch(dois=["10.1000/new"])
    finally:
        await adapter._client.aclose()  # noqa: SLF001

    assert exc_info.value.code == "documents_api_error"


def test_parse_ingest_items_handles_invalid_entries() -> None:
    parsed_empty = RagDocumentsClient._parse_ingest_items(raw_items="invalid")
    parsed_list = RagDocumentsClient._parse_ingest_items(
        raw_items=[
            "invalid",
            {"state": "succeeded"},
            {"doi": "10.1/a", "state": "succeeded", "message": "  done "},
        ]
    )

    assert parsed_empty == []
    assert len(parsed_list) == 1
    assert parsed_list[0].doi == "10.1/a"
    assert parsed_list[0].message == "done"


@pytest.mark.asyncio
async def test_request_json_raises_on_request_error() -> None:
    adapter = _build_request_error_adapter()
    with pytest.raises(AppError) as exc_info:
        await adapter._request_json(  # noqa: SLF001
            method="GET",
            url="https://rag.example.test/fail",
            unavailable_code="u",
            unavailable_message="um",
            error_code="e",
            error_message="em",
            invalid_code="i",
            invalid_message="im",
        )
    assert exc_info.value.code == "u"


@pytest.mark.asyncio
async def test_request_json_raises_on_invalid_json_or_non_dict() -> None:
    async def invalid_json_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, content=b"not-json")

    async def non_dict_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=json.dumps([1, 2]).encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    invalid_adapter = _build_client(invalid_json_handler)
    try:
        with pytest.raises(AppError) as exc_info_1:
            await invalid_adapter._request_json(  # noqa: SLF001
                method="GET",
                url="https://rag.example.test/ok",
                unavailable_code="u",
                unavailable_message="um",
                error_code="e",
                error_message="em",
                invalid_code="i",
                invalid_message="im",
            )
        assert exc_info_1.value.code == "i"
    finally:
        await invalid_adapter._client.aclose()  # noqa: SLF001

    non_dict_adapter = _build_client(non_dict_handler)
    try:
        with pytest.raises(AppError) as exc_info_2:
            await non_dict_adapter._request_json(  # noqa: SLF001
                method="GET",
                url="https://rag.example.test/ok",
                unavailable_code="u",
                unavailable_message="um",
                error_code="e",
                error_message="em",
                invalid_code="i",
                invalid_message="im",
            )
        assert exc_info_2.value.code == "i"
    finally:
        await non_dict_adapter._client.aclose()  # noqa: SLF001


@pytest.mark.asyncio
async def test_document_exists_raises_for_unavailable_or_unexpected_status() -> None:
    unavailable_adapter = _build_request_error_adapter()
    with pytest.raises(AppError) as exc_unavailable:
        await unavailable_adapter.document_exists(doi="10.1000/a")
    assert exc_unavailable.value.code == "documents_api_unavailable"

    async def status_handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"error": "x"}, status_code=409)

    status_adapter = _build_client(status_handler)
    try:
        with pytest.raises(AppError) as exc_status:
            await status_adapter.document_exists(doi="10.1000/a")
    finally:
        await status_adapter._client.aclose()  # noqa: SLF001
    assert exc_status.value.code == "documents_api_error"


@pytest.mark.asyncio
async def test_document_exists_raises_for_500_errors() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"error": "qdrant_fetch_failed"}, status_code=500)

    adapter = _build_client(handler)
    try:
        with pytest.raises(AppError) as exc_info:
            await adapter.document_exists(doi="10.1000/a")
    finally:
        await adapter._client.aclose()  # noqa: SLF001
    assert exc_info.value.code == "documents_api_error"


@pytest.mark.asyncio
async def test_fetch_batch_raises_when_job_id_missing() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return _json_response({"state": "queued"}, status_code=202)

    adapter = _build_client(handler)
    try:
        with pytest.raises(AppError) as exc_info:
            await adapter.fetch_batch(dois=["10.1000/a"])
    finally:
        await adapter._client.aclose()  # noqa: SLF001
    assert exc_info.value.code == "documents_api_invalid_response"
