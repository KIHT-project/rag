from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

__all__ = [
    "PollResult",
    "json_body",
    "post_ingest",
    "get_job",
    "extract_job_id",
    "extract_correlation_id",
    "extract_request_id",
    "is_terminal_job",
    "poll_job_until_terminal",
]


@dataclass(frozen=True, slots=True)
class PollResult:
    job: dict[str, Any]
    polls: int


def json_body(res: Any) -> dict[str, Any]:
    try:
        body = res.json()
        return body if isinstance(body, dict) else {}
    except Exception:
        return {}


def post_ingest(
    client: TestClient,
    *,
    payload: dict[str, Any],
    idempotency_key: str | None = None,
    request_id: str | None = None,
) -> Any:
    headers: dict[str, str] = {}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    if request_id:
        headers["X-Request-Id"] = request_id

    return client.post("/v1/ingest/items", json=payload, headers=headers)


def get_job(client: TestClient, *, job_id: str, request_id: str | None = None) -> Any:
    headers: dict[str, str] = {}
    if request_id:
        headers["X-Request-Id"] = request_id
    return client.get(f"/v1/ingest/jobs/{job_id}", headers=headers)


def extract_job_id(post_res: Any) -> str:
    body = json_body(post_res)
    jid = body.get("job_id") or body.get("id") or body.get("jobId")
    if not jid or not isinstance(jid, str):
        raise AssertionError(f"POST did not return job id, body={body}")
    return jid


def extract_correlation_id(post_res: Any) -> str | None:
    body = json_body(post_res)
    cid = body.get("correlation_id") or body.get("correlationId")
    if isinstance(cid, str) and cid:
        return cid
    return None


def extract_request_id(res: Any) -> str | None:
    rid = res.headers.get("X-Request-Id") or res.headers.get("x-request-id")
    if rid:
        return rid
    body = json_body(res)
    rid2 = body.get("request_id") or body.get("requestId")
    if isinstance(rid2, str) and rid2:
        return rid2
    return None


def is_terminal_job(job: dict[str, Any]) -> bool:
    status = job.get("status") or job.get("state")
    if not isinstance(status, str):
        return False
    return status.lower() in {"succeeded", "failed", "canceled", "cancelled"}


def poll_job_until_terminal(
    client: TestClient,
    *,
    job_id: str,
    timeout_seconds: int = 120,
    interval_seconds: float = 0.5,
    request_id: str | None = None,
) -> PollResult:
    deadline = time.time() + timeout_seconds
    polls = 0
    last_body: dict[str, Any] | None = None

    while time.time() < deadline:
        polls += 1
        res = get_job(client, job_id=job_id, request_id=request_id)
        if res.status_code != 200:
            raise AssertionError(
                f"GET job failed, status={res.status_code}, body={json_body(res)}"
            )
        body = json_body(res)
        last_body = body
        if is_terminal_job(body):
            return PollResult(job=body, polls=polls)
        time.sleep(interval_seconds)

    raise AssertionError(f"Job did not reach terminal state, job_id={job_id}, last={last_body}")
