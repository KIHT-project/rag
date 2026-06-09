from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import requests

from config import PROJECT_ROOT, logger


DISEASE = "thrombosis"
SOURCE_TYPE = "pubmed_abstract"


def _env_required(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var {name}. Add it to .env or export it in your shell.")
    return value


def _env_optional(name: str, default: str) -> str:
    value = (os.getenv(name) or "").strip()
    return value if value else default


def _resolve_path_from_project_root(raw: str) -> Path:
    if os.path.isabs(raw):
        return Path(raw)
    return (Path(PROJECT_ROOT) / raw).resolve()


def _chunks(seq: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"tasks json not found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("tasks json must contain a list")
    return data


def _task_to_rag_item(task: dict[str, Any]) -> dict[str, Any] | None:
    data = task.get("data")
    if not isinstance(data, dict):
        return None

    required = ["doi", "year", "title", "journal", "authors", "abstract"]
    for k in required:
        v = data.get(k)
        if v in (None, "", []):
            return None

    try:
        year = int(data["year"])
    except Exception:
        return None

    authors = data["authors"]
    if not isinstance(authors, list) or not all(isinstance(a, str) and a.strip() for a in authors):
        return None

    return {
        "doi": str(data["doi"]).strip(),
        "disease": DISEASE,
        "year": year,
        "source_type": SOURCE_TYPE,
        "title": str(data["title"]).strip(),
        "journal": str(data["journal"]).strip(),
        "authors": [a.strip() for a in authors],
        "content_text": str(data["abstract"]).strip(),
    }


def _parse_retry_after_seconds(resp: requests.Response) -> float | None:
    try:
        payload = resp.json()
    except Exception:
        payload = None

    if isinstance(payload, dict):
        details = payload.get("details")
        if isinstance(details, dict):
            ra = details.get("retry_after_seconds")
            try:
                if ra is not None:
                    return float(ra)
            except Exception:
                pass

    hdr = resp.headers.get("Retry-After")
    if hdr:
        try:
            return float(hdr)
        except Exception:
            return None

    return None


def _sleep_with_jitter(seconds: float) -> None:
    if seconds <= 0:
        return
    jitter = seconds * 0.15 * random.random()
    time.sleep(seconds + jitter)


def _post_with_retries(
    *,
    session: requests.Session,
    url: str,
    json_payload: dict[str, Any],
    timeout_s: int,
    max_retries: int,
    backoff_base_s: float,
    backoff_max_s: float,
) -> requests.Response:
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = session.post(url, json=json_payload, timeout=timeout_s)
        except requests.RequestException as e:
            if attempt > max_retries:
                raise RuntimeError(f"Request failed after {attempt} attempts, last error: {e}") from e
            sleep_s = min(backoff_max_s, backoff_base_s * (2 ** (attempt - 1)))
            _sleep_with_jitter(sleep_s)
            continue

        if resp.status_code == 429:
            retry_after = _parse_retry_after_seconds(resp) or min(backoff_max_s, backoff_base_s * (2 ** (attempt - 1)))
            if attempt > max_retries:
                raise RuntimeError(f"Ingest failed 429 after {attempt} attempts: {resp.text}")
            _sleep_with_jitter(retry_after)
            continue

        if 500 <= resp.status_code <= 599:
            if attempt > max_retries:
                raise RuntimeError(f"Ingest failed {resp.status_code} after {attempt} attempts: {resp.text}")
            sleep_s = min(backoff_max_s, backoff_base_s * (2 ** (attempt - 1)))
            _sleep_with_jitter(sleep_s)
            continue

        return resp


def main() -> None:
    base_url = _env_required("RAG_API_BASE_URL").rstrip("/")
    ingest_url = f"{base_url}/v1/ingest/items"

    default_clean_tasks = "ls_data/tasks_clean.json"
    clean_tasks_raw = _env_optional("RAG_TASKS_CLEAN_JSON_PATH", default_clean_tasks)
    clean_tasks_path = _resolve_path_from_project_root(clean_tasks_raw)

    batch_size = int(_env_optional("RAG_BATCH_SIZE", "25"))
    timeout_s = int(_env_optional("RAG_TIMEOUT", "60"))

    post_success_sleep_s = float(_env_optional("RAG_POST_SUCCESS_SLEEP_S", "2.0"))
    max_retries = int(_env_optional("RAG_MAX_RETRIES", "12"))
    backoff_base_s = float(_env_optional("RAG_BACKOFF_BASE_S", "1.0"))
    backoff_max_s = float(_env_optional("RAG_BACKOFF_MAX_S", "60.0"))

    tasks = _load_tasks(clean_tasks_path)

    items: list[dict[str, Any]] = []
    skipped = 0
    for t in tasks:
        item = _task_to_rag_item(t)
        if item is None:
            skipped += 1
            continue
        items.append(item)

    if not items:
        raise RuntimeError("No valid items to ingest")

    logger.info("Prepared %d items for ingestion, skipped %d", len(items), skipped)
    logger.info("Tasks source: %s", str(clean_tasks_path))
    logger.info("Ingest URL: %s", ingest_url)
    logger.info(
        "Batch config, batch_size=%d, timeout_s=%d, post_success_sleep_s=%s, max_retries=%d",
        batch_size,
        timeout_s,
        post_success_sleep_s,
        max_retries,
    )

    session = requests.Session()
    total_ingested = 0

    for batch in _chunks(items, batch_size):
        resp = _post_with_retries(
            session=session,
            url=ingest_url,
            json_payload={"items": batch},
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_max_s=backoff_max_s,
        )

        if resp.status_code >= 400:
            raise RuntimeError(f"Ingest failed {resp.status_code}: {resp.text}")

        total_ingested += len(batch)
        logger.info("Ingested %d of %d", total_ingested, len(items))

        _sleep_with_jitter(post_success_sleep_s)

    logger.info("RAG ingestion completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
