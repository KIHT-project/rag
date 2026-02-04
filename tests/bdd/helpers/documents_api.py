from __future__ import annotations

from typing import Any
from urllib.parse import quote

from fastapi.testclient import TestClient

from tests.bdd.helpers.ingestion_api import json_body

__all__ = [
    "delete_document",
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


def extract_error_code(res: Any) -> str:
    body = json_body(res)

    if isinstance(body.get("error"), str):
        return body["error"]

    if isinstance(body.get("detail"), list):
        return "validation_error"

    raise AssertionError(f"Missing error code, status={res.status_code}, body={body}")
