from __future__ import annotations

import re

import httpx

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.readiness import (
    CheckStatus,
    ReadinessChecks,
    ReadinessResult,
    ReadinessStatus,
)

log = get_logger(__name__)


def normalize_ollama_base_url(url: str) -> str:
    if not url:
        return ""
    return re.sub(r"/version/?$", "", url)


def evaluate_readiness_status(checks: ReadinessChecks) -> ReadinessStatus:
    all_ok = (checks.qdrant == CheckStatus.ok) and (checks.llm == CheckStatus.ok)
    return ReadinessStatus.ready if all_ok else ReadinessStatus.not_ready


async def compute_readiness(
    *,
    qdrant_url: str,
    ollama_url: str,
    timeout: httpx.Timeout,
    client: httpx.AsyncClient | None = None,
) -> ReadinessResult:
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout)

    try:
        qdrant = await check_qdrant(client, qdrant_url)
        llm = await check_ollama(client, ollama_url)
        checks = ReadinessChecks(qdrant=qdrant, llm=llm)
        return ReadinessResult(status=evaluate_readiness_status(checks), checks=checks)
    finally:
        if owns_client:
            await client.aclose()


async def check_qdrant(client: httpx.AsyncClient, base_url: str) -> CheckStatus:
    if not base_url:
        log.warning("Readiness dependency missing config, dep=qdrant")
        return CheckStatus.missing_config

    try:
        r = await client.get(f"{base_url}/collections")
    except httpx.TimeoutException:
        log.warning("Readiness dependency timeout, dep=qdrant")
        return CheckStatus.error
    except httpx.RequestError as e:
        log.warning(
            "Readiness dependency request error, dep=qdrant, type=%s", type(e).__name__
        )
        return CheckStatus.error

    if 200 <= r.status_code < 300:
        return CheckStatus.ok

    if 400 <= r.status_code < 500:
        log.warning(
            "Readiness dependency degraded, dep=qdrant, status_code=%s", r.status_code
        )
        return CheckStatus.degraded

    log.warning(
        "Readiness dependency unhealthy, dep=qdrant, status_code=%s", r.status_code
    )
    return CheckStatus.unhealthy


async def check_ollama(client: httpx.AsyncClient, base_url: str) -> CheckStatus:
    if not base_url:
        log.warning("Readiness dependency missing config, dep=ollama")
        return CheckStatus.missing_config

    try:
        r = await client.get(f"{base_url}/api/version")
    except httpx.TimeoutException:
        log.warning("Readiness dependency timeout, dep=ollama")
        return CheckStatus.error
    except httpx.RequestError as e:
        log.warning(
            "Readiness dependency request error, dep=ollama, type=%s", type(e).__name__
        )
        return CheckStatus.error

    if 200 <= r.status_code < 300:
        return CheckStatus.ok

    if 400 <= r.status_code < 500:
        log.warning(
            "Readiness dependency degraded, dep=ollama, status_code=%s", r.status_code
        )
        return CheckStatus.degraded

    log.warning(
        "Readiness dependency unhealthy, dep=ollama, status_code=%s", r.status_code
    )
    return CheckStatus.unhealthy
