from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeakLabelRule:
    """
    Defines when a labeled DOI is considered relevant.

    Default rule is conservative:
      related_to_vte must be Yes
      reason_label must contain Inclusion Evidence, if reason_label exists
    """

    require_related_to_vte_yes: bool = True
    require_inclusion_evidence: bool = False  # set True if you want stricter
    allow_missing_reason_label: bool = True


def _precision_at_k(rels: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    rels_k = rels[:k]
    return float(sum(rels_k)) / float(k)


def _recall_in_pool_at_k(rels: list[int], total_relevant_in_pool: int, k: int) -> float:
    """
    This is NOT true recall. It is recall relative to the union of relevant docs
    that appear anywhere in the Phase 3 pool for that query.
    """
    if total_relevant_in_pool <= 0 or k <= 0:
        return 0.0
    rels_k = rels[:k]
    return min(1.0, float(sum(rels_k)) / float(total_relevant_in_pool))


def _mrr(rels: list[int]) -> float:
    for idx, r in enumerate(rels, start=1):
        if r > 0:
            return 1.0 / float(idx)
    return 0.0


def _norm_doi(v: Any) -> str:
    if not isinstance(v, str):
        return ""
    s = v.strip()
    if not s:
        return ""
    s = s.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return s.strip().lower()


def _extract_first_choice(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    if not isinstance(choices[0], str):
        return ""
    return choices[0].strip()


def _extract_rating(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    rating = value.get("rating")
    if isinstance(rating, int):
        return rating
    if isinstance(rating, float):
        return int(rating)
    if isinstance(rating, str) and rating.strip().isdigit():
        return int(rating.strip())
    return None


def _extract_labels(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    labels = value.get("labels")
    if not isinstance(labels, list) or not labels:
        return []
    out: list[str] = []
    for x in labels:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out


def _label_map_from_tasks_clean(tasks_clean_path: Path) -> dict[str, dict[str, Any]]:
    """
    Returns a dict keyed by normalized doi.
    Value is a compact label dict extracted from the first annotation.
    """
    raw = json.loads(tasks_clean_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError("tasks_clean must be a JSON array")

    doi_to_labels: dict[str, dict[str, Any]] = {}

    for task in raw:
        if not isinstance(task, dict):
            continue
        data = task.get("data")
        if not isinstance(data, dict):
            continue

        doi = _norm_doi(data.get("doi"))
        if not doi:
            continue

        annotations = task.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            continue

        ann0 = annotations[0]
        if not isinstance(ann0, dict):
            continue
        results = ann0.get("result")
        if not isinstance(results, list):
            continue

        labels: dict[str, Any] = {}
        for r in results:
            if not isinstance(r, dict):
                continue
            from_name = r.get("from_name")
            if not isinstance(from_name, str) or not from_name.strip():
                continue
            from_name_s = from_name.strip()

            value = r.get("value")
            r_type = r.get("type")

            if r_type == "choices":
                labels[from_name_s] = _extract_first_choice(value)
            elif r_type == "rating":
                labels[from_name_s] = _extract_rating(value)
            elif r_type == "labels":
                labels[from_name_s] = _extract_labels(value)
            elif r_type == "textarea":
                if isinstance(value, dict):
                    text = value.get("text")
                    if isinstance(text, list) and text and isinstance(text[0], str):
                        labels[from_name_s] = text[0].strip()

        labels["source"] = data.get("source")
        labels["source_id"] = data.get("source_id")  # universal id, not pmid wording

        doi_to_labels[doi] = labels

    return doi_to_labels


def _is_relevant(labels: dict[str, Any], rule: WeakLabelRule) -> bool:
    if rule.require_related_to_vte_yes:
        v = labels.get("related_to_vte")
        if not isinstance(v, str) or v.strip().lower() != "yes":
            return False

    if rule.require_inclusion_evidence:
        rl = labels.get("reason_label")
        if rl is None:
            return rule.allow_missing_reason_label
        if isinstance(rl, list):
            has_incl = any(isinstance(x, str) and x.strip() == "Inclusion Evidence" for x in rl)
            return has_incl
        return False

    return True


def compute_phase5_metrics(
    *,
    pool_jsonl: Path,
    tasks_clean_json: Path,
    k_values: Iterable[int],
    rule: WeakLabelRule | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      summary_df, per_query_df, overlap_df

    overlap_df includes pool coverage stats independent of k
    """
    if rule is None:
        rule = WeakLabelRule()

    log.info("Phase5 | loading retrieval pool from %s", pool_jsonl)
    pool = pd.read_json(pool_jsonl, lines=True)

    required_cols = {"query_id", "rank", "doi"}
    missing = required_cols.difference(set(pool.columns))
    if missing:
        raise RuntimeError(f"Phase5 pool missing required columns: {sorted(missing)}")

    pool["doi_norm"] = pool["doi"].map(_norm_doi)

    log.info(
        "Phase5 | pool loaded, rows=%d, unique_queries=%d",
        len(pool),
        int(pool["query_id"].nunique()),
    )

    doi_to_labels = _label_map_from_tasks_clean(tasks_clean_json)
    log.info("Phase5 | loaded labeled dois=%d from %s", len(doi_to_labels), tasks_clean_json)

    def label_hit(doi_norm: str) -> int:
        if not doi_norm:
            return 0
        labels = doi_to_labels.get(doi_norm)
        if not isinstance(labels, dict):
            return 0
        return 1 if _is_relevant(labels, rule) else 0

    pool["is_relevant"] = pool["doi_norm"].map(label_hit).astype(int)

    overlap_rows: list[dict[str, Any]] = []
    for qid, g in pool.groupby("query_id"):
        g_sorted = g.sort_values("rank")
        rel_total_in_pool = int(g_sorted["is_relevant"].sum())
        overlap_rows.append(
            {
                "query_id": qid,
                "retrieved_docs": int(len(g_sorted)),
                "relevant_docs_in_pool": rel_total_in_pool,
                "has_any_relevant_in_pool": 1 if rel_total_in_pool > 0 else 0,
            }
        )
    overlap_df = pd.DataFrame(overlap_rows).sort_values("query_id")

    per_query_rows: list[dict[str, Any]] = []
    zero_relevance_queries = 0

    log.info("Phase5 | computing metrics for %d queries", int(pool["query_id"].nunique()))

    for qid, g in pool.groupby("query_id"):
        g_sorted = g.sort_values("rank")
        rels = g_sorted["is_relevant"].tolist()

        total_relevant_in_pool = int(sum(rels))
        if total_relevant_in_pool == 0:
            zero_relevance_queries += 1

        per_query: dict[str, Any] = {
            "query_id": qid,
            "mrr": _mrr(rels),
            "total_relevant_in_pool": total_relevant_in_pool,
        }

        for k in k_values:
            per_query[f"precision@{k}"] = _precision_at_k(rels, int(k))
            per_query[f"recall_in_pool@{k}"] = _recall_in_pool_at_k(rels, total_relevant_in_pool, int(k))

        per_query_rows.append(per_query)

    log.info(
        "Phase5 | queries with zero relevant docs in pool: %d / %d",
        zero_relevance_queries,
        int(pool["query_id"].nunique()),
    )

    per_query_df = pd.DataFrame(per_query_rows).sort_values("query_id")

    summary: dict[str, list[Any]] = {"metric": [], "value": []}

    summary["metric"].append("mrr")
    summary["value"].append(float(per_query_df["mrr"].mean()))

    for k in k_values:
        summary["metric"].append(f"precision@{k}")
        summary["value"].append(float(per_query_df[f"precision@{k}"].mean()))
        summary["metric"].append(f"recall_in_pool@{k}")
        summary["value"].append(float(per_query_df[f"recall_in_pool@{k}"].mean()))

    summary["metric"].append("queries_with_any_relevant_in_pool_rate")
    summary["value"].append(float(overlap_df["has_any_relevant_in_pool"].mean()))

    summary_df = pd.DataFrame(summary)

    return summary_df, per_query_df, overlap_df


def write_phase5_metrics(
    *,
    out_dir: Path,
    summary_df: pd.DataFrame,
    per_query_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "phase5_retrieval_metrics_summary.csv").write_text(
        summary_df.to_csv(index=False),
        encoding="utf-8",
    )
    (out_dir / "phase5_retrieval_metrics_per_query.csv").write_text(
        per_query_df.to_csv(index=False),
        encoding="utf-8",
    )
    (out_dir / "phase5_retrieval_overlap_per_query.csv").write_text(
        overlap_df.to_csv(index=False),
        encoding="utf-8",
    )
    (out_dir / "phase5_retrieval_metrics_summary.json").write_text(
        json.dumps(summary_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
