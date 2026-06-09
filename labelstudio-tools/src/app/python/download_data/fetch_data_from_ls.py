from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import requests


class LabelStudioAuthError(RuntimeError):
    pass


class LabelStudioHttpError(RuntimeError):
    pass


def _strip_trailing_slash(url: str) -> str:
    return url[:-1] if url.endswith("/") else url



def get_access_token(*, base_url: str, refresh_token: str, timeout_s: int) -> str:
    url = f"{_strip_trailing_slash(base_url)}/api/token/refresh/"
    r = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={"refresh": refresh_token},
        timeout=timeout_s,
    )

    if r.status_code != 200:
        raise LabelStudioAuthError(f"JWT refresh failed, status={r.status_code}, body={r.text[:800]}")

    payload = r.json()
    access = payload.get("access")
    if not isinstance(access, str) or not access.strip():
        raise LabelStudioAuthError(f"JWT refresh response missing access token: {payload}")

    return access.strip()


def get_project_tasks_page(
    *,
    base_url: str,
    project_id: int,
    access_token: str,
    page: int,
    page_size: int,
    timeout_s: int,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Returns tasks, next_url

    Supports:
      1) DRF pagination dict with results and next
      2) plain list
    """
    url = f"{_strip_trailing_slash(base_url)}/api/projects/{project_id}/tasks"
    params = {"page": page, "page_size": page_size}

    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        params=params,
        timeout=timeout_s,
    )

    if r.status_code == 401:
        raise LabelStudioAuthError(f"Unauthorized, status=401, body={r.text[:800]}")
    if r.status_code >= 400:
        raise LabelStudioHttpError(f"HTTP error, status={r.status_code}, body={r.text[:800]}")

    data = r.json()

    if isinstance(data, list):
        return data, None

    if isinstance(data, dict) and "results" in data:
        results = data.get("results")
        if not isinstance(results, list):
            raise LabelStudioHttpError(f"Unexpected pagination shape: {data}")
        next_url = data.get("next")
        return results, next_url if isinstance(next_url, str) and next_url else None

    raise LabelStudioHttpError(f"Unexpected response shape: {data}")


def export_all_tasks(
    *,
    base_url: str,
    project_id: int,
    refresh_token: str,
    page_size: int,
    timeout_s: int,
) -> List[Dict[str, Any]]:
    access = get_access_token(base_url=base_url, refresh_token=refresh_token, timeout_s=timeout_s)

    all_tasks: List[Dict[str, Any]] = []
    page = 1
    refreshed_once = False

    while True:
        try:
            tasks, next_url = get_project_tasks_page(
                base_url=base_url,
                project_id=project_id,
                access_token=access,
                page=page,
                page_size=page_size,
                timeout_s=timeout_s,
            )
        except LabelStudioAuthError:
            if refreshed_once:
                raise
            access = get_access_token(base_url=base_url, refresh_token=refresh_token, timeout_s=timeout_s)
            refreshed_once = True
            continue

        all_tasks.extend(tasks)

        if next_url is None:
            break

        page += 1
        time.sleep(0.1)

    return all_tasks



