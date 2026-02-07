from __future__ import annotations

from typing import Any, Optional

import httpx

from evaluation_metrics.src.schemas.models import SearchResponse, SearchRequest, AskResponse, AskRequest


class RagApiClient:
    def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_seconds)

    async def search(
        self,
        *,
        query: str,
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> SearchResponse:
        payload = SearchRequest(query=query, top_k=top_k, filters=filters).model_dump()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(f"{self._base_url}/v1/search", json=payload)
            r.raise_for_status()
            return SearchResponse.model_validate(r.json())

    async def ask(
        self,
        *,
        question: str,
        filters: Optional[dict[str, Any]] = None,
        hyde_enabled: bool = False,
        hyde_header_name: str = "X-HyDE-Enabled",
        hyde_header_value: str = "true",
    ) -> AskResponse:
        payload = AskRequest(question=question, filters=filters).model_dump()
        headers: dict[str, str] = {}
        if hyde_enabled:
            headers[hyde_header_name] = hyde_header_value
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(f"{self._base_url}/v1/ask", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            # Ask response schema can evolve, keep raw
            try:
                parsed = AskResponse.model_validate({"raw": data, **data})
            except Exception:
                parsed = AskResponse(answer=None, citations=[], raw=data)
            return parsed

    async def probe(self) -> None:
        """
        Lightweight connectivity check for the RAG API.
        Raises on connection/auth/server issues.
        """
        payload = SearchRequest(query="connectivity probe", top_k=1, filters=None).model_dump()
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            r = await client.post(f"{self._base_url}/v1/search", json=payload)
            r.raise_for_status()
