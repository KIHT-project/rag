from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from evaluation_metrics.src.clients.rag_api import RagApiClient
from evaluation_metrics.src.schemas.models import QueryItem, RunContext

log = logging.getLogger(__name__)


def _normalized_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_debug_rows(raw: Any, key: str) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    rows = raw.get("debug")
    if not isinstance(rows, dict):
        return []
    values = rows.get(key)
    if not isinstance(values, list):
        return []
    return [row for row in values if isinstance(row, dict)]


def _require_debug_rows(*, raw: Any, key: str, question: str, retrieval_mode: str) -> list[dict[str, Any]]:
    rows = _extract_debug_rows(raw, key)
    if rows:
        return rows
    raise RuntimeError(
        "Missing retrieval debug trace "
        f"key={key} for retrieval_mode={retrieval_mode} question={question!r}. "
        "Ensure the RAG API supports debug retrieval export."
    )


def _artifact_row(
    *,
    query_id: str,
    question: str,
    retrieval_mode: str,
    rank: int,
    run_id: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "question": question,
        "retrieval_mode": retrieval_mode,
        "rank": int(rank),
        "doc_id": _normalized_text(candidate.get("doc_id")),
        "chunk_id": _normalized_text(candidate.get("chunk_id")),
        "doi": _normalized_text(candidate.get("doi")),
        "pmid": _normalized_text(candidate.get("pmid")),
        "title": _normalized_text(candidate.get("title")),
        "score": float(candidate.get("score", 0.0) or 0.0),
        "candidate_origin": _normalized_text(candidate.get("origin")),
        "selected_for_context": bool(candidate.get("selected_for_context", False)),
        "corpus_version": "",
        "run_id": run_id,
    }


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            _normalized_text(row.get("query_id")),
            _normalized_text(row.get("retrieval_mode")),
            int(row.get("rank", 0) or 0),
            _normalized_text(row.get("chunk_id")),
            _normalized_text(row.get("doc_id")),
        ),
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in _sorted_rows(rows):
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


async def run_retrieval_export(
    *,
    ctx: RunContext,
    rag: RagApiClient,
    queries_jsonl: Path,
    hyde_header_name: str,
    hyde_header_value: str,
    retrieval_pool_depth: int,
    filters: dict[str, Any] | None = None,
    max_queries: int | None = None,
) -> dict[str, Path]:
    direct_rows: list[dict[str, Any]] = []
    hyde_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    out_direct = Path(ctx.run_dir) / "retrieval_direct.jsonl"
    out_hyde = Path(ctx.run_dir) / "retrieval_hyde.jsonl"
    out_final = Path(ctx.run_dir) / "retrieval_final_context.jsonl"

    query_limit = int(max_queries) if max_queries is not None and int(max_queries) > 0 else None
    total = 0

    with queries_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if query_limit is not None and total >= query_limit:
                break

            query = QueryItem.model_validate_json(stripped)
            total += 1

            direct_response = await rag.ask(
                question=query.text,
                filters=filters,
                hyde_enabled=False,
                debug_enabled=True,
                debug_ask_max_chunks_candidate=int(retrieval_pool_depth),
                hyde_header_name=hyde_header_name,
                hyde_header_value=hyde_header_value,
            )
            hyde_response = await rag.ask(
                question=query.text,
                filters=filters,
                hyde_enabled=True,
                debug_enabled=True,
                debug_ask_max_chunks_candidate=int(retrieval_pool_depth),
                hyde_header_name=hyde_header_name,
                hyde_header_value=hyde_header_value,
            )

            for rank, candidate in enumerate(
                _require_debug_rows(
                    raw=direct_response.raw,
                    key="question_candidates",
                    question=query.text,
                    retrieval_mode="direct",
                ),
                start=1,
            ):
                direct_rows.append(
                    _artifact_row(
                        query_id=query.id,
                        question=query.text,
                        retrieval_mode="direct",
                        rank=rank,
                        run_id=ctx.run_id,
                        candidate=candidate,
                    )
                )

            for rank, candidate in enumerate(
                _require_debug_rows(
                    raw=hyde_response.raw,
                    key="hyde_candidates",
                    question=query.text,
                    retrieval_mode="hyde",
                ),
                start=1,
            ):
                hyde_rows.append(
                    _artifact_row(
                        query_id=query.id,
                        question=query.text,
                        retrieval_mode="hyde",
                        rank=rank,
                        run_id=ctx.run_id,
                        candidate=candidate,
                    )
                )

            for retrieval_mode, response in (("direct", direct_response), ("hyde", hyde_response)):
                for rank, candidate in enumerate(
                    _require_debug_rows(
                        raw=response.raw,
                        key="selected_candidates",
                        question=query.text,
                        retrieval_mode=retrieval_mode,
                    ),
                    start=1,
                ):
                    final_rows.append(
                        _artifact_row(
                            query_id=query.id,
                            question=query.text,
                            retrieval_mode=retrieval_mode,
                            rank=rank,
                            run_id=ctx.run_id,
                            candidate=candidate,
                        )
                    )

    _write_jsonl(out_direct, direct_rows)
    _write_jsonl(out_hyde, hyde_rows)
    _write_jsonl(out_final, final_rows)

    log.info(
        "retrieval_export | done | queries=%d | direct=%d | hyde=%d | final=%d | out_dir=%s",
        total,
        len(direct_rows),
        len(hyde_rows),
        len(final_rows),
        ctx.run_dir,
    )

    return {
        "direct": out_direct,
        "hyde": out_hyde,
        "final_context": out_final,
    }
