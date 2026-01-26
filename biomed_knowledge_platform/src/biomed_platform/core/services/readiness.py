from __future__ import annotations

import re

import asyncpg
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


def normalize_asyncpg_dsn(url: str) -> str:
    if not url:
        return ""
    if url.startswith("postgresql+asyncpg://"):
        return "postgresql://" + url[len("postgresql+asyncpg://") :]
    return url


def evaluate_readiness_status(checks: ReadinessChecks) -> ReadinessStatus:
    all_ok = (
        (checks.qdrant == CheckStatus.ok)
        and (checks.llm == CheckStatus.ok)
        and (checks.rdbms == CheckStatus.ok)
    )
    return ReadinessStatus.ready if all_ok else ReadinessStatus.not_ready


async def compute_readiness(
    *,
    qdrant_url: str,
    ollama_url: str,
    postgres_url: str,
    timeout: httpx.Timeout,
    client: httpx.AsyncClient | None = None,
) -> ReadinessResult:
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout)

    errors: dict[str, dict[str, object]] = {}

    try:
        qdrant_status, qdrant_err = await check_qdrant(client, qdrant_url)
        if qdrant_err is not None:
            errors["qdrant"] = qdrant_err

        llm_status, llm_err = await check_ollama(client, ollama_url)
        if llm_err is not None:
            errors["ollama"] = llm_err

        rdbms_status, rdbms_err = await check_postgres(postgres_url)
        if rdbms_err is not None:
            errors["rdbms"] = rdbms_err

        checks = ReadinessChecks(qdrant=qdrant_status, llm=llm_status, rdbms=rdbms_status)
        status = evaluate_readiness_status(checks)

        return ReadinessResult(
            status=status,
            checks=checks,
            errors=errors or None,
        )
    finally:
        if owns_client and client is not None:
            await client.aclose()


async def check_qdrant(
    client: httpx.AsyncClient, base_url: str
) -> tuple[CheckStatus, dict[str, object] | None]:
    if not base_url:
        log.warning("Readiness dependency missing config, dep=qdrant")
        return CheckStatus.missing_config, {"reason": "Missing configuration"}

    try:
        r = await client.get(f"{base_url}/collections")
    except httpx.TimeoutException:
        log.warning("Readiness dependency timeout, dep=qdrant")
        return CheckStatus.error, {"reason": "timeout", "base_url": base_url}
    except httpx.RequestError as e:
        log.warning("Readiness dependency request error, dep=qdrant, type=%s", type(e).__name__)
        return CheckStatus.error, {"reason": type(e).__name__, "base_url": base_url}

    if 200 <= r.status_code < 300:
        return CheckStatus.ok, None

    if 400 <= r.status_code < 500:
        log.warning("Readiness dependency degraded, dep=qdrant, status_code=%s", r.status_code)
        return CheckStatus.degraded, {"reason": "http_4xx", "status_code": r.status_code}

    log.warning("Readiness dependency unhealthy, dep=qdrant, status_code=%s", r.status_code)
    return CheckStatus.unhealthy, {"reason": "http_5xx", "status_code": r.status_code}


async def check_ollama(
    client: httpx.AsyncClient, base_url: str
) -> tuple[CheckStatus, dict[str, object] | None]:
    if not base_url:
        log.warning("Readiness dependency missing config, dep=ollama")
        return CheckStatus.missing_config, {"reason": "Missing configuration"}

    try:
        r = await client.get(f"{base_url}/api/version")
    except httpx.TimeoutException:
        log.warning("Readiness dependency timeout, dep=ollama")
        return CheckStatus.error, {"reason": "timeout", "base_url": base_url}
    except httpx.RequestError as e:
        log.warning("Readiness dependency request error, dep=ollama, type=%s", type(e).__name__)
        return CheckStatus.error, {"reason": type(e).__name__, "base_url": base_url}

    if 200 <= r.status_code < 300:
        return CheckStatus.ok, None

    if 400 <= r.status_code < 500:
        log.warning("Readiness dependency degraded, dep=ollama, status_code=%s", r.status_code)
        return CheckStatus.degraded, {"reason": "http_4xx", "status_code": r.status_code}

    log.warning("Readiness dependency unhealthy, dep=ollama, status_code=%s", r.status_code)
    return CheckStatus.unhealthy, {"reason": "http_5xx", "status_code": r.status_code}


async def check_postgres(
    postgres_url: str,
) -> tuple[CheckStatus, dict[str, object] | None]:
    dsn = normalize_asyncpg_dsn(postgres_url)
    if not dsn:
        log.warning("Readiness dependency missing config, dep=rdbms")
        return CheckStatus.missing_config, {"reason": "Missing configuration"}

    try:
        conn = await asyncpg.connect(dsn=dsn, timeout=2.0)
    except (asyncpg.InvalidCatalogNameError, asyncpg.InvalidAuthorizationSpecificationError) as e:
        log.warning("Readiness dependency degraded, dep=rdbms, type=%s", type(e).__name__)
        return CheckStatus.degraded, {"reason": type(e).__name__}
    except (asyncpg.CannotConnectNowError, asyncpg.TooManyConnectionsError) as e:
        log.warning("Readiness dependency unhealthy, dep=rdbms, type=%s", type(e).__name__)
        return CheckStatus.unhealthy, {"reason": type(e).__name__}
    except Exception as e:
        log.warning("Readiness dependency error, dep=rdbms, type=%s", type(e).__name__)
        return CheckStatus.error, {"reason": type(e).__name__}

    try:
        await conn.execute("SELECT 1")
        return CheckStatus.ok, None
    except Exception as e:
        log.warning("Readiness dependency error, dep=rdbms, type=%s", type(e).__name__)
        return CheckStatus.error, {"reason": type(e).__name__}
    finally:
        await conn.close()
