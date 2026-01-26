from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


def post_ask(
    client: TestClient,
    *,
    payload: dict[str, Any],
    request_id: str | None = None,
    hyde_enabled: bool | None = None,
) -> Any:
    headers: dict[str, str] = {}
    if request_id:
        headers["X-Request-Id"] = request_id
    if hyde_enabled is not None:
        headers["X-HyDE-Enabled"] = "true" if hyde_enabled else "false"
    return client.post("/v1/ask", json=payload, headers=headers)
