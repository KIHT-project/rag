# src/app/python/download_data/main.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT, LABEL_STUDIO_URL, PROJECT_ID, logger
from download_data.fetch_data_from_ls import export_all_tasks
from download_data.find_doi import enrich_tasks_with_doi
from download_data.save_data import load_export_target


def _get_refresh_token() -> str:
    token = (os.getenv("LABEL_STUDIO_API_KEY") or "").strip()
    if token:
        return token
    raise RuntimeError("Missing refresh token, set LABEL_STUDIO_API_KEY")


def _atomic_write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(obj, ensure_ascii=False, indent=2)
    tmp.write_text(payload, encoding="utf-8")
    json.loads(tmp.read_text(encoding="utf-8"))
    os.replace(tmp, path)


def _assert_tasks_json_safe(tasks: list[dict[str, Any]]) -> None:
    bad = 0
    for t in tasks:
        d = t.get("data")
        if not isinstance(d, dict):
            continue
        for k in ("doi", "source", "source_id", "title", "abstract", "journal"):
            v = d.get(k)
            if isinstance(v, str) and "from __future__ import annotations" in v:
                bad += 1
                break
    if bad:
        raise RuntimeError("Detected python source text inside task fields, refusing to write json")


def _is_nonempty_str(v: object) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _is_valid_year(v: object) -> bool:
    if isinstance(v, int):
        return 1800 <= v <= 2100
    if isinstance(v, str) and v.isdigit():
        y = int(v)
        return 1800 <= y <= 2100
    return False


def _is_nonempty_authors(v: object) -> bool:
    if not isinstance(v, list) or not v:
        return False
    for a in v:
        if not _is_nonempty_str(a):
            return False
    return True


def _validate_task_for_ingest(task: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    d = task.get("data")
    if not isinstance(d, dict):
        return False, ["missing data dict"]

    if not _is_nonempty_str(d.get("source")):
        reasons.append("missing source")

    if not _is_nonempty_str(d.get("source_id")):
        reasons.append("missing source_id")

    if not _is_nonempty_str(d.get("doi")):
        reasons.append("missing doi")

    if not _is_valid_year(d.get("year")):
        reasons.append("missing or invalid year")

    if not _is_nonempty_str(d.get("journal")):
        reasons.append("missing journal")

    if not _is_nonempty_authors(d.get("authors")):
        reasons.append("missing authors")

    if not _is_nonempty_str(d.get("title")):
        reasons.append("missing title")

    if not _is_nonempty_str(d.get("abstract")):
        reasons.append("missing abstract")

    return len(reasons) == 0, reasons


def main() -> None:
    timeout_s = int(os.getenv("LS_TIMEOUT", "1200"))
    page_size = int(os.getenv("LS_PAGE_SIZE", "1300"))

    refresh_token = _get_refresh_token()
    target = load_export_target(project_root=PROJECT_ROOT)

    tasks = export_all_tasks(
        base_url=LABEL_STUDIO_URL,
        project_id=PROJECT_ID,
        refresh_token=refresh_token,
        page_size=page_size,
        timeout_s=timeout_s,
    )
    logger.info("Fetched %d tasks from Label Studio", len(tasks))

    tasks, missing_source_ids = enrich_tasks_with_doi(tasks)

    _assert_tasks_json_safe(tasks)

    out_dir = target.directory
    raw_path = out_dir / "tasks_raw.json"
    clean_path = out_dir / "tasks_clean.json"
    rejected_path = out_dir / "tasks_rejected.json"
    missing_path = out_dir / "missing_source_ids.json"

    clean: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for t in tasks:
        ok, reasons = _validate_task_for_ingest(t)
        if ok:
            clean.append(t)
        else:
            d = t.get("data") if isinstance(t.get("data"), dict) else {}
            rejected.append(
                {
                    "id": t.get("id"),
                    "source": d.get("source"),
                    "source_id": d.get("source_id"),
                    "reasons": reasons,
                }
            )

    _atomic_write_json(raw_path, tasks)
    _atomic_write_json(clean_path, clean)
    _atomic_write_json(rejected_path, rejected)

    if missing_source_ids:
        _atomic_write_json(
            missing_path,
            {"missing_count": len(missing_source_ids), "missing_source_ids": missing_source_ids},
        )

    logger.info("Wrote raw tasks to %s", raw_path)
    logger.info("Wrote clean ingest tasks to %s", clean_path)
    logger.info("Wrote rejected report to %s", rejected_path)

    logger.info(
        "Counts, total=%d, clean=%d, rejected=%d, missing_doi_source_ids=%d",
        len(tasks),
        len(clean),
        len(rejected),
        len(missing_source_ids),
    )


if __name__ == "__main__":
    main()
