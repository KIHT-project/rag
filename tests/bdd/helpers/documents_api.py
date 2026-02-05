from __future__ import annotations

from typing import Any
from urllib.parse import quote

from fastapi.testclient import TestClient

from tests.bdd.helpers.ingestion_api import json_body

__all__ = [
    "delete_document",
    "get_document",
    "list_dois",
    "post_fetch_document",
    "post_fetch_batch",
    "extract_error_code",
]


def delete_document(
    client: TestClient,
    *,
    doi: str,
    request_id: str | None = None,
) -> Any:
    headers: dict[str, str] = {}
    if request_id:
        headers["X-Request-Id"] = request_id

    encoded = quote(doi, safe="")
    return client.delete(f"/v1/documents/{encoded}", headers=headers)


def get_document(
    client: TestClient,
    *,
    doi: str,
    request_id: str | None = None,
) -> Any:
    headers: dict[str, str] = {}
    if request_id:
        headers["X-Request-Id"] = request_id

    encoded = quote(doi, safe="")
    return client.get(f"/v1/documents/{encoded}", headers=headers)


def list_dois(
    client: TestClient,
    *,
    include_document_info: bool | None = None,
    request_id: str | None = None,
) -> Any:
    headers: dict[str, str] = {}
    if request_id:
        headers["X-Request-Id"] = request_id
    if include_document_info is not None:
        headers["X-Include-Document-Info"] = "true" if include_document_info else "false"

    return client.get("/v1/documents", headers=headers)


def post_fetch_document(
    client: TestClient,
    *,
    payload: dict[str, Any],
    ingest_enabled: bool | None = None,
    request_id: str | None = None,
) -> Any:
    headers: dict[str, str] = {}
    if request_id:
        headers["X-Request-Id"] = request_id
    if ingest_enabled is not None:
        headers["X-Ingest-Enabled"] = "true" if ingest_enabled else "false"
    return client.post("/v1/documents/fetch", json=payload, headers=headers)


def post_fetch_batch(
    client: TestClient,
    *,
    payload: dict[str, Any],
    ingest_enabled: bool | None = None,
    request_id: str | None = None,
) -> Any:
    headers: dict[str, str] = {}
    if request_id:
        headers["X-Request-Id"] = request_id
    if ingest_enabled is not None:
        headers["X-Ingest-Enabled"] = "true" if ingest_enabled else "false"
    return client.post("/v1/documents/fetch/batch", json=payload, headers=headers)


def extract_error_code(res: Any) -> str:
    body = json_body(res)

    if isinstance(body.get("error"), str):
        return body["error"]

    if isinstance(body.get("detail"), list):
        return "validation_error"

    raise AssertionError(f"Missing error code, status={res.status_code}, body={body}")
