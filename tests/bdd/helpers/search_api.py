from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


def post_search(
    client: TestClient,
    *,
    payload: dict[str, Any],
    request_id: str | None = None,
) -> Any:
    headers: dict[str, str] = {}
    if request_id:
        headers["X-Request-Id"] = request_id
    return client.post("/v1/search", json=payload, headers=headers)
