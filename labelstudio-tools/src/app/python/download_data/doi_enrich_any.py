# src/app/python/download_data/doi_enrich_any.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from download_data.find_doi import enrich_tasks_with_doi


JsonFormat = Literal["json", "jsonl"]


@dataclass(frozen=True)
class _Schema:
    name: str
    kind: Literal["labelstudio_task", "flat_record"]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _detect_format(path: Path) -> JsonFormat:
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    return "json"


def _load_input(path: Path) -> list[dict[str, Any]]:
    fmt = _detect_format(path)
    raw = _read_text(path)

    if fmt == "jsonl":
        items: list[dict[str, Any]] = []
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            if not isinstance(obj, dict):
                raise ValueError("jsonl line must be an object")
            items.append(obj)
        return items

    obj = json.loads(raw)
    if not isinstance(obj, list):
        raise ValueError("json must be a list of objects")
    for it in obj:
        if not isinstance(it, dict):
            raise ValueError("json list must contain only objects")
    return obj


def _dump_output(path: Path, items: list[dict[str, Any]]) -> None:
    fmt = _detect_format(path)
    if fmt == "jsonl":
        out = "\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n"
        _write_text(path, out)
        return

    out = json.dumps(items, ensure_ascii=False, indent=2)
    _write_text(path, out)


def _detect_schema(item: dict[str, Any]) -> _Schema:
    if isinstance(item.get("data"), dict):
        d = item["data"]
        if "source_id" in d or "pmid" in d:
            return _Schema(name="labelstudio_task_data", kind="labelstudio_task")
    if "source_id" in item or "pmid" in item:
        return _Schema(name="flat_record", kind="flat_record")
    return _Schema(name="unknown", kind="flat_record")


def _to_task_shape(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[_Schema]]:
    schemas: list[_Schema] = []
    tasks: list[dict[str, Any]] = []

    for it in items:
        sch = _detect_schema(it)
        schemas.append(sch)

        if sch.kind == "labelstudio_task":
            tasks.append(it)
            continue

        data: dict[str, Any] = {}
        for k in (
            "source",
            "source_id",
            "source_url",
            "pmid",
            "title",
            "abstract",
            "year",
            "journal",
            "authors",
            "doi",
            "doi_url",
        ):
            if k in it:
                data[k] = it.get(k)
        tasks.append({"data": data})

    return tasks, schemas


def _from_task_shape(tasks: list[dict[str, Any]], schemas: list[_Schema]) -> list[dict[str, Any]]:
    if len(tasks) != len(schemas):
        raise ValueError("tasks and schemas length mismatch")

    out: list[dict[str, Any]] = []

    for task, sch in zip(tasks, schemas, strict=True):
        if sch.kind == "labelstudio_task":
            out.append(task)
            continue

        d = task.get("data")
        if not isinstance(d, dict):
            out.append({})
            continue

        rec: dict[str, Any] = {}
        for k, v in d.items():
            rec[k] = v
        out.append(rec)

    return out


def enrich_file_with_doi(*, input_path: Path, output_path: Path) -> dict[str, Any]:
    items = _load_input(input_path)
    tasks, schemas = _to_task_shape(items)

    tasks_enriched, missing_source_ids = enrich_tasks_with_doi(tasks)

    restored = _from_task_shape(tasks_enriched, schemas)
    _dump_output(output_path, restored)

    return {
        "items_in": len(items),
        "items_out": len(restored),
        "missing_source_ids": missing_source_ids,
        "missing_count": len(missing_source_ids),
        "output_path": str(output_path),
    }
