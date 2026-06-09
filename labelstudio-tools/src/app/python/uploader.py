import time
from typing import List, Dict, Any
from label_studio_sdk import LabelStudio, __version__ as lsdk_version
from config import (
    LABEL_STUDIO_URL, API_KEY, PROJECT_ID, IMPORT_RETRIES, IMPORT_BACKOFF, logger
)

def _import_with_retries(ls: LabelStudio, project_id: int, batch, retries: int, backoff: float) -> None:
    attempt = 0
    while True:
        try:
            ls.projects.import_tasks(id=project_id, request=batch)
            return
        except Exception as e:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"Import failed after {retries} retries: {e}") from e
            sleep_s = backoff ** attempt
            logger.warning("Import attempt %d failed: %s. Retrying in %.2fs", attempt, e, sleep_s)
            time.sleep(sleep_s)

def upload_tasks(tasks: List[Dict[str, Any]]) -> None:
    if not API_KEY:
        raise ValueError("LABEL_STUDIO_API_KEY is not set")

    logger.info("label-studio-sdk version: %s", lsdk_version)
    try:
        ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)
        proj_ids = [p.id for p in ls.projects.list()]
    except Exception as e:
        raise RuntimeError(f"Failed to authenticate or list projects: {e}") from e

    if PROJECT_ID not in proj_ids:
        raise PermissionError(f"No access to project {PROJECT_ID}. Visible: {proj_ids}")

    CHUNK = 500
    total = 0
    for i in range(0, len(tasks), CHUNK):
        batch = tasks[i:i + CHUNK]
        _import_with_retries(ls, PROJECT_ID, batch, IMPORT_RETRIES, IMPORT_BACKOFF)
        total += len(batch)
        logger.info("Imported batch. Total imported so far: %d", total)

    logger.info("Import complete. Total tasks imported: %d", total)
