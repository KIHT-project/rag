from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from scheduler_pubmed.src.core.domains.scheduler import PubMedSearchResult
from scheduler_pubmed.src.core.errors.errors import system_error

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _parse_pubmed_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None

    for fmt in (
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%Y %b %d",
        "%Y %b",
        "%Y",
    ):
        try:
            parsed = datetime.strptime(raw, fmt)
            return parsed.replace(tzinfo=UTC)
        except ValueError:
            continue

    year = raw[:4]
    if len(year) == 4 and year.isdigit():
        return datetime(int(year), 1, 1, tzinfo=UTC)
    return None


def _extract_doi(summary: dict[str, Any]) -> str | None:
    article_ids = summary.get("articleids")
    if not isinstance(article_ids, list):
        return None

    for item in article_ids:
        if not isinstance(item, dict):
            continue
        id_type = str(item.get("idtype", "")).strip().lower()
        if id_type != "doi":
            continue
        value = str(item.get("value", "")).strip()
        if value:
            return value
    return None


class PubMedQueryClient:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        base_url: str = _EUTILS_BASE,
        retmax: int = 200,
        reldate_days: int = 1,
        datetype: str = "pdat",
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._retmax = max(1, int(retmax))
        self._reldate_days = max(1, int(reldate_days))
        self._datetype = datetype.strip().lower() or "pdat"

    async def _get_json(self, *, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/{path.lstrip('/')}"
        try:
            response = await self._client.get(url, params=params)
        except httpx.RequestError as exc:
            raise system_error(
                code="pubmed_unavailable",
                message="Failed to reach PubMed API",
                details={"error": str(exc), "url": url},
                retryable=True,
            ) from exc

        if response.status_code >= 400:
            raise system_error(
                code="pubmed_unavailable",
                message="PubMed API request failed",
                details={"status_code": response.status_code, "url": url},
                retryable=True,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise system_error(
                code="pubmed_invalid_response",
                message="PubMed API returned invalid JSON",
                details={"url": url},
                retryable=True,
            ) from exc
        if not isinstance(data, dict):
            raise system_error(
                code="pubmed_invalid_response",
                message="PubMed API returned unexpected payload",
                details={"url": url},
                retryable=True,
            )
        return data

    async def _search_pmids(self, *, query: str, reldate_days: int) -> list[str]:
        payload = await self._get_json(
            path="/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": self._retmax,
                "sort": "pub+date",
                "reldate": max(1, int(reldate_days)),
                "datetype": self._datetype,
            },
        )
        result = payload.get("esearchresult")
        if not isinstance(result, dict):
            return []
        idlist = result.get("idlist")
        if not isinstance(idlist, list):
            return []
        return [str(item).strip() for item in idlist if str(item).strip()]

    async def _fetch_summaries(self, *, pmids: list[str]) -> dict[str, dict[str, Any]]:
        if not pmids:
            return {}

        payload = await self._get_json(
            path="/esummary.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json",
            },
        )
        result = payload.get("result")
        if not isinstance(result, dict):
            return {}

        entries: dict[str, dict[str, Any]] = {}
        for pmid in pmids:
            item = result.get(pmid)
            if isinstance(item, dict):
                entries[pmid] = item
        return entries

    async def search(
        self,
        *,
        query: str,
        reldate_days: int | None = None,
    ) -> list[PubMedSearchResult]:
        effective_reldate_days = (
            self._reldate_days if reldate_days is None else max(1, int(reldate_days))
        )
        pmids = await self._search_pmids(query=query, reldate_days=effective_reldate_days)
        if not pmids:
            return []

        summaries = await self._fetch_summaries(pmids=pmids)
        results: list[PubMedSearchResult] = []
        for pmid in pmids:
            summary = summaries.get(pmid)
            if summary is None:
                results.append(PubMedSearchResult(pmid=pmid, doi=None, published_at=None))
                continue

            doi = _extract_doi(summary)
            published_at = _parse_pubmed_datetime(
                summary.get("sortpubdate") or summary.get("pubdate")
            )
            results.append(PubMedSearchResult(pmid=pmid, doi=doi, published_at=published_at))
        return results
