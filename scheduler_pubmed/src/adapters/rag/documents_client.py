from __future__ import annotations

from urllib.parse import quote

import httpx

from scheduler_pubmed.src.core.domains.scheduler import (
    FetchBatchAccepted,
    IngestJobItemStatus,
    IngestJobStatus,
)
from scheduler_pubmed.src.core.errors.errors import system_error


class RagDocumentsClient:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        base_url: str,
        documents_get_path: str,
        documents_post_batch_path: str,
        ingest_jobs_get_path: str,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._documents_get_path = documents_get_path
        self._documents_post_batch_path = documents_post_batch_path
        self._ingest_jobs_get_path = ingest_jobs_get_path

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    async def _request_json(
        self,
        *,
        method: str,
        url: str,
        json_payload: dict | None = None,
        unavailable_code: str,
        unavailable_message: str,
        error_code: str,
        error_message: str,
        invalid_code: str,
        invalid_message: str,
        expected_status: int | None = None,
    ) -> dict:
        try:
            response = await self._client.request(method=method, url=url, json=json_payload)
        except httpx.RequestError as exc:
            raise system_error(
                code=unavailable_code,
                message=unavailable_message,
                details={"error": str(exc), "url": url},
                retryable=True,
            ) from exc

        if expected_status is not None:
            has_error = response.status_code != expected_status
        else:
            has_error = response.status_code >= 400

        if has_error:
            raise system_error(
                code=error_code,
                message=error_message,
                details={"status_code": response.status_code, "url": url},
                retryable=True,
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise system_error(
                code=invalid_code,
                message=invalid_message,
                details={"url": url},
                retryable=True,
            ) from exc

        if not isinstance(body, dict):
            raise system_error(
                code=invalid_code,
                message=invalid_message,
                details={"url": url},
                retryable=True,
            )
        return body

    @staticmethod
    def _parse_ingest_items(*, raw_items: object) -> list[IngestJobItemStatus]:
        if not isinstance(raw_items, list):
            return []

        items: list[IngestJobItemStatus] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            doi = str(raw_item.get("doi", "")).strip()
            if not doi:
                continue
            state = str(raw_item.get("state", "")).strip()
            message = raw_item.get("message")
            message_str = str(message).strip() if message is not None else None
            items.append(
                IngestJobItemStatus(
                    doi=doi,
                    state=state,
                    message=message_str if message_str else None,
                )
            )
        return items

    async def document_exists(self, *, doi: str) -> bool:
        url = self._url(f"{self._documents_get_path}{quote(doi, safe='')}")
        try:
            response = await self._client.get(url)
        except httpx.RequestError as exc:
            raise system_error(
                code="documents_api_unavailable",
                message="Failed to reach documents API",
                details={"error": str(exc), "url": url},
                retryable=True,
            ) from exc

        if response.status_code == 200:
            return True
        if response.status_code == 404:
            return False

        raise system_error(
            code="documents_api_error",
            message="Unexpected response while checking document existence",
            details={"status_code": response.status_code, "url": url},
            retryable=True,
        )

    async def fetch_batch(self, *, dois: list[str]) -> FetchBatchAccepted:
        url = self._url(self._documents_post_batch_path)
        payload = {"items": [{"doi": doi} for doi in dois]}
        body = await self._request_json(
            method="POST",
            url=url,
            json_payload=payload,
            unavailable_code="documents_api_unavailable",
            unavailable_message="Failed to reach documents API",
            error_code="documents_api_error",
            error_message="Unexpected response while enqueueing document batch",
            invalid_code="documents_api_invalid_response",
            invalid_message="Invalid JSON from documents API",
            expected_status=202,
        )

        job_id = str(body.get("job_id", "")).strip()
        state = str(body.get("state", "")).strip()
        if not job_id:
            raise system_error(
                code="documents_api_invalid_response",
                message="Missing job_id from documents API",
                details={"url": url},
                retryable=True,
            )
        return FetchBatchAccepted(job_id=job_id, state=state)

    async def get_ingest_job_status(self, *, job_id: str) -> IngestJobStatus:
        url = self._url(f"{self._ingest_jobs_get_path}{job_id}")
        body = await self._request_json(
            method="GET",
            url=url,
            unavailable_code="documents_api_unavailable",
            unavailable_message="Failed to reach ingest jobs API",
            error_code="documents_api_error",
            error_message="Unexpected response while reading ingest job status",
            invalid_code="documents_api_invalid_response",
            invalid_message="Invalid JSON from ingest jobs API",
            expected_status=None,
        )
        items = self._parse_ingest_items(raw_items=body.get("items"))

        return IngestJobStatus(
            job_id=str(body.get("job_id", "")).strip() or job_id,
            state=str(body.get("state", "")).strip(),
            items=items,
        )
