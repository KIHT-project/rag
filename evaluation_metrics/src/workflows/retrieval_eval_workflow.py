from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from evaluation_metrics.src.clients.rag_api import RagApiClient
from evaluation_metrics.src.qrels_annotation import (
    IDENTIFIER_PRIORITY,
    build_pooled_candidates,
    generate_qrels,
    validate_annotations,
    write_annotation_template,
    write_pooled_candidates,
)
from evaluation_metrics.src.retrieval_export import run_retrieval_export
from evaluation_metrics.src.retrieval_metrics import (
    DEFAULT_K_VALUES,
    compute_retrieval_metrics,
    write_outputs,
)
from evaluation_metrics.src.schemas.models import RunContext

RETRIEVAL_DIRECT = "retrieval_direct.jsonl"
RETRIEVAL_HYDE = "retrieval_hyde.jsonl"
RETRIEVAL_FINAL_CONTEXT = "retrieval_final_context.jsonl"
POOLED_CANDIDATES = "pooled_candidates.jsonl"
QRELS_ANNOTATION_TEMPLATE = "qrels_annotation_template.csv"
QRELS_ANNOTATION_COMPLETED = "qrels_annotation_completed.csv"
QRELS = "qrels.tsv"
METRICS_SUMMARY = "retrieval_metrics_summary.json"
METRICS_PER_QUERY = "retrieval_metrics_per_query.csv"
METRICS_COMPARISON_TEX = "retrieval_metrics_comparison.tex"


@dataclass(frozen=True, slots=True)
class RetrievalEvalPrepareSummary:
    run_dir: Path
    retrieval_direct_path: Path
    retrieval_hyde_path: Path
    retrieval_final_context_path: Path
    pooled_candidates_path: Path
    annotation_template_path: Path
    retrieval_direct_row_count: int
    retrieval_hyde_row_count: int
    retrieval_final_context_row_count: int
    pooled_candidate_count: int


@dataclass(frozen=True, slots=True)
class RetrievalEvalFinalizeSummary:
    run_dir: Path
    qrels_path: Path
    metrics_summary_path: Path
    per_query_metrics_path: Path
    latex_table_path: Path
    metric_depths: list[int]
    query_count: int
    judged_candidate_count: int


def _count_jsonl_rows(path: Path, *, label: str) -> int:
    if not path.exists():
        raise RuntimeError(f"{label} is missing: {path}")
    if not path.is_file():
        raise RuntimeError(f"{label} is not a file: {path}")

    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{label} contains invalid JSON on line {line_number}: {path}"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"{label} line {line_number} is not a JSON object: {path}"
                )
            count += 1

    if count == 0:
        raise RuntimeError(f"{label} is empty: {path}")
    return count


def _resolve_depth_doc_identity(
    row: dict[str, Any], *, label: str, line_number: int
) -> str:
    for field_name in IDENTIFIER_PRIORITY:
        value = str(row.get(field_name) or "").strip()
        if value:
            return value
    query_id = str(row.get("query_id") or "").strip() or "<missing query_id>"
    raise RuntimeError(
        f"{label} line {line_number} has no stable identifier for query_id={query_id}. "
        "Expected one of doc_id, pmid, doi, or chunk_id."
    )


def _deduped_depth_by_query(path: Path, *, label: str) -> dict[str, int]:
    ranked_rows: dict[str, list[tuple[int, int, str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"{label} line {line_number} is not a JSON object: {path}"
                )

            query_id = str(row.get("query_id") or "").strip()
            if not query_id:
                raise RuntimeError(
                    f"{label} line {line_number} has empty query_id: {path}"
                )

            rank_raw = row.get("rank")
            try:
                rank = (
                    int(rank_raw)
                    if rank_raw is not None and str(rank_raw).strip() != ""
                    else line_number
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{label} line {line_number} has invalid rank {rank_raw!r}: {path}"
                ) from exc

            doc_id = _resolve_depth_doc_identity(
                row,
                label=label,
                line_number=line_number,
            )
            ranked_rows.setdefault(query_id, []).append((rank, line_number, doc_id))

    depths: dict[str, int] = {}
    for query_id, items in ranked_rows.items():
        seen_doc_ids: set[str] = set()
        for _, _, doc_id in sorted(items, key=lambda item: (item[0], item[1])):
            seen_doc_ids.add(doc_id)
        depths[query_id] = len(seen_doc_ids)
    return depths


def _validate_metric_depths(
    *,
    direct_path: Path,
    hyde_path: Path,
    metric_depths: list[int],
) -> None:
    max_k = max(metric_depths)
    invalid: list[str] = []
    for mode, path in (("direct", direct_path), ("hyde", hyde_path)):
        depths = _deduped_depth_by_query(path, label=f"{mode} retrieval")
        too_shallow = [
            query_id for query_id, depth in sorted(depths.items()) if depth < max_k
        ]
        if too_shallow:
            invalid.append(f"mode={mode} k={max_k} queries={', '.join(too_shallow)}")

    if invalid:
        raise ValueError(
            "Requested metric depth exceeds available retrieval depth after deduplication: "
            + "; ".join(invalid)
        )


def _normalize_k_values(k_values: Iterable[int] | None) -> list[int]:
    try:
        normalized = sorted({int(k) for k in (k_values or DEFAULT_K_VALUES)})
    except (TypeError, ValueError) as exc:
        raise ValueError("All k values must be positive integers.") from exc

    if not normalized:
        raise ValueError("At least one k value is required.")
    if any(k <= 0 for k in normalized):
        raise ValueError("All k values must be positive integers.")
    return normalized


async def prepare_retrieval_evaluation(
    *,
    ctx: RunContext,
    rag: RagApiClient,
    queries_jsonl: Path,
    hyde_header_name: str,
    hyde_header_value: str,
    retrieval_pool_depth: int,
    max_queries: int | None = None,
    force: bool = False,
) -> RetrievalEvalPrepareSummary:
    if int(retrieval_pool_depth) <= 0:
        raise ValueError("retrieval_pool_depth must be a positive integer.")

    run_dir = Path(ctx.run_dir)
    completed_annotations = run_dir / QRELS_ANNOTATION_COMPLETED
    if completed_annotations.exists() and not force:
        raise RuntimeError(
            f"{QRELS_ANNOTATION_COMPLETED} already exists in {run_dir}. "
            "Refusing to prepare over a completed annotation run without --force."
        )

    await run_retrieval_export(
        ctx=ctx,
        rag=rag,
        queries_jsonl=queries_jsonl,
        hyde_header_name=hyde_header_name,
        hyde_header_value=hyde_header_value,
        retrieval_pool_depth=int(retrieval_pool_depth),
        max_queries=max_queries,
    )

    retrieval_direct_path = run_dir / RETRIEVAL_DIRECT
    retrieval_hyde_path = run_dir / RETRIEVAL_HYDE
    retrieval_final_context_path = run_dir / RETRIEVAL_FINAL_CONTEXT

    retrieval_direct_count = _count_jsonl_rows(
        retrieval_direct_path,
        label=RETRIEVAL_DIRECT,
    )
    retrieval_hyde_count = _count_jsonl_rows(
        retrieval_hyde_path,
        label=RETRIEVAL_HYDE,
    )
    retrieval_final_context_count = _count_jsonl_rows(
        retrieval_final_context_path,
        label=RETRIEVAL_FINAL_CONTEXT,
    )

    pooled_candidates_path = run_dir / POOLED_CANDIDATES
    annotation_template_path = run_dir / QRELS_ANNOTATION_TEMPLATE
    pooled_rows = build_pooled_candidates(
        direct_path=retrieval_direct_path,
        hyde_path=retrieval_hyde_path,
    )
    if not pooled_rows:
        raise RuntimeError(
            "No pooled candidates were generated from retrieval artifacts."
        )

    write_pooled_candidates(pooled_rows, pooled_candidates_path)
    write_annotation_template(pooled_rows, annotation_template_path)

    return RetrievalEvalPrepareSummary(
        run_dir=run_dir,
        retrieval_direct_path=retrieval_direct_path,
        retrieval_hyde_path=retrieval_hyde_path,
        retrieval_final_context_path=retrieval_final_context_path,
        pooled_candidates_path=pooled_candidates_path,
        annotation_template_path=annotation_template_path,
        retrieval_direct_row_count=retrieval_direct_count,
        retrieval_hyde_row_count=retrieval_hyde_count,
        retrieval_final_context_row_count=retrieval_final_context_count,
        pooled_candidate_count=len(pooled_rows),
    )


def finalize_retrieval_evaluation(
    *,
    run_dir: Path,
    annotations_path: Path | None = None,
    k_values: Iterable[int] | None = None,
) -> RetrievalEvalFinalizeSummary:
    metric_depths = _normalize_k_values(k_values)
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise RuntimeError(f"run_dir does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise RuntimeError(f"run_dir is not a directory: {run_dir}")

    pooled_candidates_path = run_dir / POOLED_CANDIDATES
    retrieval_direct_path = run_dir / RETRIEVAL_DIRECT
    retrieval_hyde_path = run_dir / RETRIEVAL_HYDE
    annotations = annotations_path or (run_dir / QRELS_ANNOTATION_COMPLETED)

    if not annotations.exists():
        if annotations_path is None:
            raise RuntimeError(
                f"{QRELS_ANNOTATION_COMPLETED} is missing in {run_dir}. "
                "manual annotation is required. "
                f"Copy {QRELS_ANNOTATION_TEMPLATE} to {QRELS_ANNOTATION_COMPLETED}, "
                "fill relevance and rationale, then run retrieval-eval-finalize."
            )
        raise RuntimeError(f"annotations file does not exist: {annotations}")
    if not annotations.is_file():
        raise RuntimeError(f"annotations path is not a file: {annotations}")

    _count_jsonl_rows(pooled_candidates_path, label=POOLED_CANDIDATES)
    _count_jsonl_rows(retrieval_direct_path, label=RETRIEVAL_DIRECT)
    _count_jsonl_rows(retrieval_hyde_path, label=RETRIEVAL_HYDE)
    _validate_metric_depths(
        direct_path=retrieval_direct_path,
        hyde_path=retrieval_hyde_path,
        metric_depths=metric_depths,
    )

    validated_annotations = validate_annotations(
        pooled_path=pooled_candidates_path,
        annotations_path=annotations,
    )

    qrels_path = run_dir / QRELS
    generate_qrels(
        pooled_path=pooled_candidates_path,
        annotations_path=annotations,
        qrels_output_path=qrels_path,
    )

    summary, per_query_rows = compute_retrieval_metrics(
        direct_path=retrieval_direct_path,
        hyde_path=retrieval_hyde_path,
        qrels_path=qrels_path,
        k_values=metric_depths,
    )

    metrics_summary_path = run_dir / METRICS_SUMMARY
    per_query_metrics_path = run_dir / METRICS_PER_QUERY
    latex_table_path = run_dir / METRICS_COMPARISON_TEX
    write_outputs(
        summary=summary,
        per_query_rows=per_query_rows,
        summary_output=metrics_summary_path,
        per_query_output=per_query_metrics_path,
        latex_output=latex_table_path,
    )

    return RetrievalEvalFinalizeSummary(
        run_dir=run_dir,
        qrels_path=qrels_path,
        metrics_summary_path=metrics_summary_path,
        per_query_metrics_path=per_query_metrics_path,
        latex_table_path=latex_table_path,
        metric_depths=metric_depths,
        query_count=int(summary["query_count"]),
        judged_candidate_count=len(validated_annotations),
    )


def format_prepare_summary(summary: RetrievalEvalPrepareSummary) -> str:
    return "\n".join(
        [
            "Retrieval evaluation prepare summary:",
            f"run_dir: {summary.run_dir}",
            f"retrieval_direct row count: {summary.retrieval_direct_row_count}",
            f"retrieval_hyde row count: {summary.retrieval_hyde_row_count}",
            f"retrieval_final_context row count: {summary.retrieval_final_context_row_count}",
            f"pooled candidate count: {summary.pooled_candidate_count}",
            f"annotation template path: {summary.annotation_template_path}",
            "",
            "Copy qrels_annotation_template.csv to qrels_annotation_completed.csv.",
            "Fill relevance and rationale.",
            "Then run retrieval-eval-finalize.",
        ]
    )


def format_finalize_summary(summary: RetrievalEvalFinalizeSummary) -> str:
    return "\n".join(
        [
            "Retrieval evaluation finalize summary:",
            f"run_dir: {summary.run_dir}",
            f"qrels path: {summary.qrels_path}",
            f"metrics summary path: {summary.metrics_summary_path}",
            f"per query metrics path: {summary.per_query_metrics_path}",
            f"latex table path: {summary.latex_table_path}",
            f"metric depths: {', '.join(str(k) for k in summary.metric_depths)}",
            f"query count: {summary.query_count}",
            f"judged candidate count: {summary.judged_candidate_count}",
        ]
    )
