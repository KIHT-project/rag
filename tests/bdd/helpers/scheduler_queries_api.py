from __future__ import annotations

from typing import Any


def create_pubmed_query(client, *, payload: dict[str, Any]) -> Any:
    return client.post("/v1/pubmed/queries", json=payload)


def list_pubmed_queries(client, *, enabled: bool | None = None) -> Any:
    params: dict[str, Any] = {}
    if enabled is not None:
        params["enabled"] = enabled
    return client.get("/v1/pubmed/queries", params=params)


def get_pubmed_query(client, *, query_id: str) -> Any:
    return client.get(f"/v1/pubmed/queries/{query_id}")


def update_pubmed_query(client, *, query_id: str, payload: dict[str, Any]) -> Any:
    return client.patch(f"/v1/pubmed/queries/{query_id}", json=payload)


def enable_pubmed_query(client, *, query_id: str) -> Any:
    return client.patch(f"/v1/pubmed/queries/{query_id}/enable")


def disable_pubmed_query(client, *, query_id: str) -> Any:
    return client.patch(f"/v1/pubmed/queries/{query_id}/disable")
