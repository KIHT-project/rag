from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

POOL_REQUIRED_FIELDS = [
    "query_id",
    "question",
    "resolved_doc_id",
    "doc_id",
    "chunk_id",
    "doi",
    "pmid",
    "title",
    "corpus_version",
    "run_id",
]

TEMPLATE_FIELDS = [
    "query_id",
    "question",
    "resolved_doc_id",
    "doc_id",
    "chunk_id",
    "doi",
    "pmid",
    "title",
    "relevance",
    "rationale",
    "annotator",
    "adjudication_status",
]

REQUIRED_ANNOTATION_COLUMNS = set(TEMPLATE_FIELDS)
ALLOWED_RELEVANCE_VALUES = {"0", "1", "2"}
ALLOWED_ADJUDICATION_STATUSES = {"pending", "accepted", "review_needed"}
IDENTIFIER_PRIORITY = ("doc_id", "pmid", "doi", "chunk_id")


def _normalized_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _resolve_doc_identity(row: dict[str, Any]) -> str:
    for field_name in IDENTIFIER_PRIORITY:
        value = _normalized_text(row.get(field_name))
        if value:
            return value

    query_id = _normalized_text(row.get("query_id")) or "<missing query_id>"
    title = _normalized_text(row.get("title")) or "<missing title>"
    raise ValueError(
        "Candidate is missing a stable identifier for "
        f"query_id={query_id} title={title}. "
        "Expected one of doc_id, pmid, doi, or chunk_id."
    )


def _canonicalize_candidate(row: dict[str, Any]) -> dict[str, str]:
    candidate = {field_name: _normalized_text(row.get(field_name)) for field_name in POOL_REQUIRED_FIELDS}
    candidate["resolved_doc_id"] = _resolve_doc_identity(row)
    return candidate


def _merge_candidate_rows(existing: dict[str, str], incoming: dict[str, str]) -> dict[str, str]:
    merged = dict(existing)
    for field_name in POOL_REQUIRED_FIELDS:
        current_value = merged.get(field_name, "")
        incoming_value = incoming.get(field_name, "")
        if not current_value and incoming_value:
            merged[field_name] = incoming_value
    return merged


def build_pooled_candidates(direct_path: Path, hyde_path: Path) -> list[dict[str, str]]:
    pooled_by_key: dict[tuple[str, str], dict[str, str]] = {}

    for source_path in (direct_path, hyde_path):
        for row in _read_jsonl(source_path):
            candidate = _canonicalize_candidate(row)
            query_id = candidate["query_id"]
            resolved_doc_id = candidate["resolved_doc_id"]
            if not query_id:
                raise ValueError(f"Candidate in {source_path} is missing query_id.")
            key = (query_id, resolved_doc_id)
            if key in pooled_by_key:
                pooled_by_key[key] = _merge_candidate_rows(pooled_by_key[key], candidate)
            else:
                pooled_by_key[key] = candidate

    return sorted(
        pooled_by_key.values(),
        key=lambda row: (row["query_id"], row["resolved_doc_id"]),
    )


def write_pooled_candidates(rows: list[dict[str, str]], output_path: Path) -> None:
    _write_jsonl(output_path, rows)


def write_annotation_template(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEMPLATE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "query_id": row["query_id"],
                    "question": row["question"],
                    "resolved_doc_id": row["resolved_doc_id"],
                    "doc_id": row["doc_id"],
                    "chunk_id": row["chunk_id"],
                    "doi": row["doi"],
                    "pmid": row["pmid"],
                    "title": row["title"],
                    "relevance": "",
                    "rationale": "",
                    "annotator": "",
                    "adjudication_status": "",
                }
            )


def create_template(
    *,
    direct_path: Path,
    hyde_path: Path,
    pooled_output_path: Path,
    template_output_path: Path,
) -> list[dict[str, str]]:
    pooled_rows = build_pooled_candidates(direct_path=direct_path, hyde_path=hyde_path)
    write_pooled_candidates(pooled_rows, pooled_output_path)
    write_annotation_template(pooled_rows, template_output_path)
    return pooled_rows


def _read_annotations_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_columns = [column for column in TEMPLATE_FIELDS if column not in fieldnames]
        if missing_columns:
            joined = ", ".join(missing_columns)
            raise ValueError(f"Annotation CSV is missing required columns: {joined}")

        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({key: _normalized_text(value) for key, value in row.items()})
    return rows


def validate_annotations(
    *,
    pooled_path: Path,
    annotations_path: Path,
) -> list[dict[str, str]]:
    pooled_rows = [_canonicalize_candidate(row) for row in _read_jsonl(pooled_path)]
    pooled_keys = {(row["query_id"], row["resolved_doc_id"]) for row in pooled_rows}
    annotations = _read_annotations_csv(annotations_path)

    errors: list[str] = []
    seen_keys: set[tuple[str, str]] = set()

    for row_number, row in enumerate(annotations, start=2):
        query_id = row.get("query_id", "")
        resolved_doc_id = row.get("resolved_doc_id", "")
        relevance = row.get("relevance", "")
        rationale = row.get("rationale", "")
        adjudication_status = row.get("adjudication_status", "")

        if not query_id:
            errors.append(f"Row {row_number}: query_id is empty.")
        if not resolved_doc_id:
            errors.append(f"Row {row_number}: resolved_doc_id is empty.")

        key = (query_id, resolved_doc_id)
        if query_id and resolved_doc_id:
            if key in seen_keys:
                errors.append(
                    "Duplicate annotation row for "
                    f"query_id={query_id} resolved_doc_id={resolved_doc_id}."
                )
            seen_keys.add(key)

            if key not in pooled_keys:
                errors.append(
                    "Unknown annotation candidate "
                    f"query_id={query_id} resolved_doc_id={resolved_doc_id}."
                )

        if not relevance:
            errors.append(f"Row {row_number}: relevance is empty.")
        elif relevance not in ALLOWED_RELEVANCE_VALUES:
            errors.append(f"Row {row_number}: relevance must be one of 0, 1, 2.")

        if relevance in {"1", "2"} and not rationale:
            errors.append(f"Row {row_number}: rationale is required when relevance is {relevance}.")

        if adjudication_status and adjudication_status not in ALLOWED_ADJUDICATION_STATUSES:
            errors.append(
                f"Row {row_number}: adjudication_status must be one of "
                "pending, accepted, review_needed."
            )

    missing_keys = sorted(pooled_keys - seen_keys)
    for query_id, resolved_doc_id in missing_keys:
        errors.append(
            "Missing annotation row for "
            f"query_id={query_id} resolved_doc_id={resolved_doc_id}."
        )

    if errors:
        raise ValueError("\n".join(errors))

    return sorted(
        annotations,
        key=lambda row: (row["query_id"], row["resolved_doc_id"]),
    )


def generate_qrels(
    *,
    pooled_path: Path,
    annotations_path: Path,
    qrels_output_path: Path,
) -> list[dict[str, str]]:
    annotations = validate_annotations(
        pooled_path=pooled_path,
        annotations_path=annotations_path,
    )

    qrels_output_path.parent.mkdir(parents=True, exist_ok=True)
    with qrels_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query_id", "doc_id", "relevance"])
        for row in annotations:
            writer.writerow([row["query_id"], row["resolved_doc_id"], row["relevance"]])

    return annotations
