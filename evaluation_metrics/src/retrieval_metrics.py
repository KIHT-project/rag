from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation_metrics.src.qrels_annotation import IDENTIFIER_PRIORITY

REQUIRED_QRELS_COLUMNS = ("query_id", "doc_id", "relevance")
ALLOWED_RELEVANCE_VALUES = {0, 1, 2}
DEFAULT_K_VALUES = (5, 10)


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


def _resolve_doc_identity(row: dict[str, Any], *, source: Path, line_number: int) -> str:
    for field_name in IDENTIFIER_PRIORITY:
        value = _normalized_text(row.get(field_name))
        if value:
            return value

    query_id = _normalized_text(row.get("query_id")) or "<missing query_id>"
    raise ValueError(
        f"Retrieval row in {source} line {line_number} has no stable identifier for query_id={query_id}. "
        "Expected one of doc_id, pmid, doi, or chunk_id."
    )


def _dcg_at_k(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for index, rel in enumerate(relevances[:k], start=1):
        dcg += (2**rel - 1) / math.log2(index + 1)
    return dcg


def _first_relevant_rank(relevances: list[int]) -> int | None:
    for index, rel in enumerate(relevances, start=1):
        if rel > 0:
            return index
    return None


def _metric_key(prefix: str, k: int) -> str:
    return f"{prefix}_at_{k}"


def _latex_metric_label(metric_name: str) -> str:
    if metric_name == "mrr":
        return "MRR"
    if metric_name.startswith("precision_at_"):
        return f"Precision@{metric_name.split('_')[-1]}"
    if metric_name.startswith("recall_at_"):
        return f"Recall@{metric_name.split('_')[-1]}"
    if metric_name.startswith("ndcg_at_"):
        return f"nDCG@{metric_name.split('_')[-1]}"
    return metric_name


def _aggregate_metric_name_order(k_values: list[int]) -> list[str]:
    names = ["mrr"]
    names.extend(_metric_key("precision", k) for k in k_values)
    names.extend(_metric_key("recall", k) for k in k_values)
    names.extend(_metric_key("ndcg", k) for k in k_values)
    return names


def _load_qrels(qrels_path: Path) -> tuple[dict[str, dict[str, int]], Counter[int]]:
    if not qrels_path.exists():
        raise ValueError(f"qrels.tsv not found: {qrels_path}")

    with qrels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        missing_columns = [column for column in REQUIRED_QRELS_COLUMNS if column not in fieldnames]
        if missing_columns:
            raise ValueError(
                "qrels.tsv is missing required columns: " + ", ".join(missing_columns)
            )

        qrels_by_query: dict[str, dict[str, int]] = {}
        seen_pairs: set[tuple[str, str]] = set()
        relevance_distribution: Counter[int] = Counter()

        for row_number, row in enumerate(reader, start=2):
            query_id = _normalized_text(row.get("query_id"))
            doc_id = _normalized_text(row.get("doc_id"))
            relevance_raw = _normalized_text(row.get("relevance"))

            if not query_id:
                raise ValueError(f"qrels.tsv row {row_number} has empty query_id.")
            if not doc_id:
                raise ValueError(f"qrels.tsv row {row_number} has empty doc_id.")

            try:
                relevance = int(relevance_raw)
            except ValueError as exc:
                raise ValueError(
                    f"qrels.tsv row {row_number} has invalid relevance {relevance_raw!r}; expected 0, 1, or 2."
                ) from exc
            if relevance not in ALLOWED_RELEVANCE_VALUES:
                raise ValueError(
                    f"qrels.tsv row {row_number} has invalid relevance {relevance}; expected 0, 1, or 2."
                )

            pair = (query_id, doc_id)
            if pair in seen_pairs:
                raise ValueError(
                    f"Duplicate qrels row for query_id={query_id} doc_id={doc_id}."
                )
            seen_pairs.add(pair)

            qrels_by_query.setdefault(query_id, {})[doc_id] = relevance
            relevance_distribution[relevance] += 1

    return qrels_by_query, relevance_distribution


def _load_retrieval_mode(
    retrieval_path: Path,
    *,
    mode: str,
    qrels_query_ids: set[str],
) -> dict[str, list[str]]:
    rows = _read_jsonl(retrieval_path)
    unexpected_queries: set[str] = set()
    ranked_rows: dict[str, list[tuple[int, int, str]]] = {}

    for line_number, row in enumerate(rows, start=1):
        query_id = _normalized_text(row.get("query_id"))
        if not query_id:
            raise ValueError(f"Retrieval row in {retrieval_path} line {line_number} has empty query_id.")
        if query_id not in qrels_query_ids:
            unexpected_queries.add(query_id)

        resolved_doc_id = _resolve_doc_identity(row, source=retrieval_path, line_number=line_number)
        rank_raw = row.get("rank")
        try:
            rank = int(rank_raw) if rank_raw is not None and str(rank_raw).strip() != "" else line_number
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Retrieval row in {retrieval_path} line {line_number} has invalid rank {rank_raw!r}."
            ) from exc

        ranked_rows.setdefault(query_id, []).append((rank, line_number, resolved_doc_id))

    if unexpected_queries:
        joined = ", ".join(sorted(unexpected_queries))
        raise ValueError(
            f"Retrieval mode {mode} contains queries missing from qrels.tsv: {joined}."
        )

    deduped: dict[str, list[str]] = {}
    for query_id, items in ranked_rows.items():
        seen_doc_ids: set[str] = set()
        ordered_doc_ids: list[str] = []
        for _, _, resolved_doc_id in sorted(items, key=lambda item: (item[0], item[1])):
            if resolved_doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(resolved_doc_id)
            ordered_doc_ids.append(resolved_doc_id)
        deduped[query_id] = ordered_doc_ids

    return deduped


def compute_retrieval_metrics(
    *,
    direct_path: Path,
    hyde_path: Path,
    qrels_path: Path,
    k_values: list[int] | None = None,
    strict: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    k_values = sorted({int(k) for k in (k_values or DEFAULT_K_VALUES)})
    if not k_values:
        raise ValueError("At least one k value is required.")
    if any(k <= 0 for k in k_values):
        raise ValueError("All k values must be positive integers.")

    qrels_by_query, relevance_distribution = _load_qrels(qrels_path)
    qrels_query_ids = set(qrels_by_query)

    retrieval_by_mode = {
        "direct": _load_retrieval_mode(direct_path, mode="direct", qrels_query_ids=qrels_query_ids),
        "hyde": _load_retrieval_mode(hyde_path, mode="hyde", qrels_query_ids=qrels_query_ids),
    }

    missing_queries_by_mode = {
        mode: sorted(qrels_query_ids - set(mode_rows))
        for mode, mode_rows in retrieval_by_mode.items()
    }
    if strict and any(missing_queries_by_mode.values()):
        details = "; ".join(
            f"{mode}: {', '.join(query_ids)}"
            for mode, query_ids in missing_queries_by_mode.items()
            if query_ids
        )
        raise ValueError(f"qrels.tsv contains queries missing from retrieval files in strict mode: {details}.")

    invalid_depths: list[str] = []
    for mode, mode_rows in retrieval_by_mode.items():
        for k in k_values:
            too_shallow = [
                query_id
                for query_id, doc_ids in sorted(mode_rows.items())
                if len(doc_ids) < k
            ]
            if too_shallow:
                invalid_depths.append(
                    f"mode={mode} k={k} queries={', '.join(too_shallow)}"
                )
    if invalid_depths:
        raise ValueError(
            "Requested metric depth exceeds available retrieval depth after deduplication: "
            + "; ".join(invalid_depths)
        )

    judged_relevant_count_by_query = {
        query_id: sum(1 for rel in doc_rels.values() if rel > 0)
        for query_id, doc_rels in qrels_by_query.items()
    }

    per_query_rows: list[dict[str, Any]] = []
    aggregate_values_by_mode: dict[str, dict[str, list[float]]] = {}
    excluded_counts_by_mode: dict[str, dict[str, int]] = {}
    retrieval_pool_depth_by_mode: dict[str, dict[str, Any]] = {}

    for mode, mode_rows in retrieval_by_mode.items():
        aggregate_values_by_mode[mode] = {metric_name: [] for metric_name in _aggregate_metric_name_order(k_values)}
        excluded_counts_by_mode[mode] = {metric_name: 0 for metric_name in _aggregate_metric_name_order(k_values)}
        retrieval_pool_depth_by_mode[mode] = {
            "min_depth": min((len(doc_ids) for doc_ids in mode_rows.values()), default=0),
            "max_depth": max((len(doc_ids) for doc_ids in mode_rows.values()), default=0),
            "depth_by_query": {query_id: len(mode_rows[query_id]) for query_id in sorted(mode_rows)},
        }

        for query_id in sorted(mode_rows):
            doc_ids = mode_rows[query_id]
            qrels_for_query = qrels_by_query[query_id]
            graded_relevances = [int(qrels_for_query.get(doc_id, 0)) for doc_id in doc_ids]
            binary_relevances = [1 if rel > 0 else 0 for rel in graded_relevances]
            judged_relevant_count = judged_relevant_count_by_query[query_id]

            first_relevant_rank = _first_relevant_rank(binary_relevances)
            mrr = 0.0 if first_relevant_rank is None else 1.0 / float(first_relevant_rank)

            row: dict[str, Any] = {
                "query_id": query_id,
                "retrieval_mode": mode,
                "retrieved_count": len(doc_ids),
                "judged_relevant_count": judged_relevant_count,
                "mrr": mrr,
            }
            aggregate_values_by_mode[mode]["mrr"].append(mrr)

            ideal_relevances = sorted(qrels_for_query.values(), reverse=True)

            for k in k_values:
                precision_key = _metric_key("precision", k)
                recall_key = _metric_key("recall", k)
                ndcg_key = _metric_key("ndcg", k)

                precision = sum(binary_relevances[:k]) / float(k)
                row[precision_key] = precision
                aggregate_values_by_mode[mode][precision_key].append(precision)

                if judged_relevant_count == 0:
                    row[recall_key] = None
                    excluded_counts_by_mode[mode][recall_key] += 1
                else:
                    recall = sum(binary_relevances[:k]) / float(judged_relevant_count)
                    row[recall_key] = recall
                    aggregate_values_by_mode[mode][recall_key].append(recall)

                idcg = _dcg_at_k(ideal_relevances, k)
                if idcg == 0.0:
                    row[ndcg_key] = None
                    excluded_counts_by_mode[mode][ndcg_key] += 1
                else:
                    ndcg = _dcg_at_k(graded_relevances, k) / idcg
                    row[ndcg_key] = ndcg
                    aggregate_values_by_mode[mode][ndcg_key].append(ndcg)

            per_query_rows.append(row)

    aggregate_metrics_by_mode: dict[str, dict[str, float | None]] = {}
    for mode, metrics in aggregate_values_by_mode.items():
        aggregate_metrics_by_mode[mode] = {}
        for metric_name in _aggregate_metric_name_order(k_values):
            values = metrics[metric_name]
            aggregate_metrics_by_mode[mode][metric_name] = (
                sum(values) / float(len(values)) if values else None
            )

    evaluated_query_ids = sorted(set().union(*(set(mode_rows) for mode_rows in retrieval_by_mode.values())))
    summary = {
        "run_id": _derive_run_id(retrieval_by_mode, direct_path=direct_path, hyde_path=hyde_path),
        "retrieval_modes": ["direct", "hyde"],
        "metric_depths": k_values,
        "retrieval_pool_depth_by_mode": retrieval_pool_depth_by_mode,
        "query_count": len(evaluated_query_ids),
        "evaluated_query_count_by_mode": {
            mode: len(mode_rows) for mode, mode_rows in retrieval_by_mode.items()
        },
        "judged_candidate_count": int(sum(relevance_distribution.values())),
        "relevance_distribution": {
            str(key): int(relevance_distribution.get(key, 0))
            for key in sorted(ALLOWED_RELEVANCE_VALUES)
        },
        "excluded_query_counts_by_metric": excluded_counts_by_mode,
        "aggregate_metrics_by_mode": aggregate_metrics_by_mode,
        "qrels_queries_missing_in_retrieval_by_mode": missing_queries_by_mode,
    }

    ordered_per_query_rows = sorted(
        per_query_rows,
        key=lambda row: (str(row["query_id"]), str(row["retrieval_mode"])),
    )
    return summary, ordered_per_query_rows


def _derive_run_id(
    retrieval_by_mode: dict[str, dict[str, list[str]]],
    *,
    direct_path: Path,
    hyde_path: Path,
) -> str:
    # Prefer a shared parent run directory when the two retrieval files were exported together.
    direct_parent = direct_path.parent.name
    hyde_parent = hyde_path.parent.name
    if direct_parent and direct_parent == hyde_parent:
        return direct_parent

    # Fall back to a deterministic identifier derived from filenames when paths differ.
    mode_sizes = ",".join(
        f"{mode}:{len(rows)}" for mode, rows in sorted(retrieval_by_mode.items())
    )
    return f"{direct_path.stem}+{hyde_path.stem}:{mode_sizes}"


def write_outputs(
    *,
    summary: dict[str, Any],
    per_query_rows: list[dict[str, Any]],
    summary_output: Path,
    per_query_output: Path,
    latex_output: Path,
) -> None:
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = list(_ordered_per_query_fieldnames(per_query_rows))
    per_query_output.parent.mkdir(parents=True, exist_ok=True)
    with per_query_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_query_rows:
            writer.writerow(
                {
                    field: "" if row.get(field) is None else row.get(field)
                    for field in fieldnames
                }
            )

    latex_output.parent.mkdir(parents=True, exist_ok=True)
    latex_output.write_text(_build_latex_table(summary), encoding="utf-8")


def _ordered_per_query_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return [
            "query_id",
            "retrieval_mode",
            "retrieved_count",
            "judged_relevant_count",
            "mrr",
        ]

    metric_keys = [key for key in rows[0] if key not in {
        "query_id", "retrieval_mode", "retrieved_count", "judged_relevant_count", "mrr"
    }]
    return [
        "query_id",
        "retrieval_mode",
        "retrieved_count",
        "judged_relevant_count",
        "mrr",
        *metric_keys,
    ]


def _build_latex_table(summary: dict[str, Any]) -> str:
    k_values = [int(k) for k in summary["metric_depths"]]
    ordered_metrics = _aggregate_metric_name_order(k_values)
    direct = summary["aggregate_metrics_by_mode"]["direct"]
    hyde = summary["aggregate_metrics_by_mode"]["hyde"]

    lines = [
        "\\begin{tabular}{lcc}",
        "\\hline",
        "Metric & Direct & HyDE \\\\",
        "\\hline",
    ]
    for metric_name in ordered_metrics:
        direct_value = direct.get(metric_name)
        hyde_value = hyde.get(metric_name)
        direct_text = "" if direct_value is None else f"{float(direct_value):.4f}"
        hyde_text = "" if hyde_value is None else f"{float(hyde_value):.4f}"
        lines.append(f"{_latex_metric_label(metric_name)} & {direct_text} & {hyde_text} \\\\")
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "",
        ]
    )
    return "\n".join(lines)
