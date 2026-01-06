from __future__ import annotations

from fastapi import APIRouter, Request, Response, status
import httpx

router = APIRouter(tags=["System"])


@router.get("/health", summary="Health check")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness check")
async def readiness_check(request: Request, response: Response) -> dict:
    settings = getattr(request.app.state, "settings", None)

    qdrant_url = ""
    ollama_url = ""

    if settings is not None:
        qdrant_cfg = settings.require_qdrant()
        llm_cfg = settings.require_llm()
        qdrant_url = str(qdrant_cfg.get("url", "")).rstrip("/")
        ollama_url = str(llm_cfg.get("ollama_base_url", "")).rstrip("/version")

    checks: dict[str, str] = {"qdrant": "unknown", "llm": "unknown"}
    timeout = httpx.Timeout(connect=2.0, read=2.0, write=2.0, pool=2.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        checks["qdrant"] = await _check_qdrant(client, qdrant_url)
        checks["llm"] = await _check_ollama(client, ollama_url)

    all_ok = all(v == "ok" for v in checks.values())
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "checks": checks}

    return {"status": "ready", "checks": checks}


async def _check_qdrant(client: httpx.AsyncClient, base_url: str) -> str:
    if not base_url:
        return "missing_config"
    try:
        r = await client.get(f"{base_url}/collections")
        if 200 <= r.status_code < 300:
            return "ok"
        return f"http_{r.status_code}"
    except httpx.TimeoutException:
        return "timeout"
    except httpx.RequestError:
        return "unreachable"


async def _check_ollama(client: httpx.AsyncClient, base_url: str) -> str:
    if not base_url:
        return "missing_config"
    try:
        r = await client.get(f"{base_url}/api/version")
        if 200 <= r.status_code < 300:
            return "ok"
        return f"http_{r.status_code}"
    except httpx.TimeoutException:
        return "timeout"
    except httpx.RequestError:
        return "unreachable"
