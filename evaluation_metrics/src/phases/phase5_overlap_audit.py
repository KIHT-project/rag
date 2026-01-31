from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

log = logging.getLogger(__name__)


def _norm_doi(v: Any) -> str:
    if not isinstance(v, str):
        return ""
    s = v.strip().lower()
    if not s:
        return ""
    for prefix in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "doi:"):
        if s.startswith(prefix):
            s = s[len(prefix) :].strip()
    return s


def _norm_text(v: Any) -> str:
    if not isinstance(v, str):
        return ""
    s = v.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def _title_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, _norm_text(a), _norm_text(b)).ratio()


@dataclass(frozen=True)
class LabelFieldInfo:
    name: str
    types: set[str]


def discover_label_fields(tasks_clean_json: Path) -> list[LabelFieldInfo]:
    raw = json.loads(tasks_clean_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError("tasks_clean must be a JSON array")

    seen: dict[str, set[str]] = {}

    for task in raw:
        if not isinstance(task, dict):
            continue
        annotations = task.get("annotations")
        if not isinstance(annotations, list):
            continue
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            results = ann.get("result")
            if not isinstance(results, list):
                continue
            for r in results:
                if not isinstance(r, dict):
                    continue
                from_name = r.get("from_name")
                r_type = r.get("type")
                if isinstance(from_name, str) and from_name.strip() and isinstance(r_type, str) and r_type.strip():
                    seen.setdefault(from_name.strip(), set()).add(r_type.strip())

    out = [LabelFieldInfo(name=k, types=v) for k, v in sorted(seen.items(), key=lambda x: x[0])]
    return out


def _extract_label_value(result: dict[str, Any]) -> Any:
    r_type = result.get("type")
    value = result.get("value")
    if r_type == "choices":
        if isinstance(value, dict):
            choices = value.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], str):
                return choices[0].strip()
        return None
    if r_type == "rating":
        if isinstance(value, dict):
            rating = value.get("rating")
            if isinstance(rating, (int, float)):
                return int(rating)
            if isinstance(rating, str) and rating.strip().isdigit():
                return int(rating.strip())
        return None
    if r_type == "labels":
        if isinstance(value, dict):
            labels = value.get("labels")
            if isinstance(labels, list):
                out: list[str] = []
                for x in labels:
                    if isinstance(x, str) and x.strip():
                        out.append(x.strip())
                return out
        return None
    if r_type == "textarea":
        if isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, list) and text and isinstance(text[0], str):
                return text[0].strip()
        return None
    return None


def _index_tasks_by_doi(tasks_clean_json: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(tasks_clean_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError("tasks_clean must be a JSON array")

    doi_map: dict[str, dict[str, Any]] = {}

    for task in raw:
        if not isinstance(task, dict):
            continue
        data = task.get("data")
        if not isinstance(data, dict):
            continue

        doi = _norm_doi(data.get("doi"))
        if not doi:
            continue

        rec: dict[str, Any] = {
            "doi": doi,
            "title": data.get("title") if isinstance(data.get("title"), str) else "",
            "year": data.get("year"),
            "journal": data.get("journal") if isinstance(data.get("journal"), str) else "",
            "source": data.get("source") if isinstance(data.get("source"), str) else "",
            "source_id": data.get("source_id") if isinstance(data.get("source_id"), str) else "",
        }

        labels: dict[str, Any] = {}
        annotations = task.get("annotations")
        if isinstance(annotations, list) and annotations:
            ann0 = annotations[0]
            if isinstance(ann0, dict):
                results = ann0.get("result")
                if isinstance(results, list):
                    for r in results:
                        if not isinstance(r, dict):
                            continue
                        from_name = r.get("from_name")
                        if not isinstance(from_name, str) or not from_name.strip():
                            continue
                        labels[from_name.strip()] = _extract_label_value(r)

        rec["labels"] = labels
        doi_map[doi] = rec

    return doi_map


def _load_phase3_pool(phase3_pool_jsonl: Path) -> pd.DataFrame:
    pool = pd.read_json(phase3_pool_jsonl, lines=True)
    required = {"query_id", "rank", "doi"}
    missing = required.difference(set(pool.columns))
    if missing:
        raise RuntimeError(f"phase3_pool missing columns: {sorted(missing)}")
    pool["doi_norm"] = pool["doi"].map(_norm_doi)
    pool["rank"] = pool["rank"].astype(int)
    if "score" in pool.columns:
        pool["score"] = pool["score"].astype(float)
    else:
        pool["score"] = 0.0
    return pool


def build_overlap_audit(
    *,
    phase3_pool_jsonl: Path,
    tasks_clean_json: Path,
    out_dir: Path,
    k_values: Iterable[int],
    positive_label_field: str | None = None,
    positive_yes_value: str = "Yes",
    min_title_similarity: float = 0.92,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    doi_to_task = _index_tasks_by_doi(tasks_clean_json)
    pool = _load_phase3_pool(phase3_pool_jsonl)

    tasks_dois = set(doi_to_task.keys())
    pool_dois = set([x for x in pool["doi_norm"].tolist() if isinstance(x, str) and x])

    intersection = tasks_dois.intersection(pool_dois)

    summary: dict[str, Any] = {
        "tasks_unique_dois": len(tasks_dois),
        "phase3_pool_unique_dois": len(pool_dois),
        "intersection_unique_dois": len(intersection),
        "tasks_covered_by_pool_rate": (len(intersection) / len(tasks_dois)) if tasks_dois else 0.0,
        "pool_covered_by_tasks_rate": (len(intersection) / len(pool_dois)) if pool_dois else 0.0,
        "positive_label_field": positive_label_field or "",
    }

    (out_dir / "phase5_overlap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta_rows: list[dict[str, Any]] = []

    pool_by_doi = pool.sort_values("rank").groupby("doi_norm", as_index=False).first()

    for _, row in pool_by_doi.iterrows():
        doi = row["doi_norm"]
        if not isinstance(doi, str) or not doi:
            continue
        t = doi_to_task.get(doi)
        if t is None:
            continue

        title_pool = row.get("title") if isinstance(row.get("title"), str) else ""
        title_task = t.get("title") if isinstance(t.get("title"), str) else ""

        year_pool = row.get("year")
        year_task = t.get("year")

        journal_pool = row.get("journal") if isinstance(row.get("journal"), str) else ""
        journal_task = t.get("journal") if isinstance(t.get("journal"), str) else ""

        title_sim = _title_similarity(title_pool, title_task)

        year_match = 0
        if isinstance(year_pool, (int, float)) and isinstance(year_task, (int, float)):
            year_match = 1 if int(year_pool) == int(year_task) else 0

        journal_match = 0
        if journal_pool and journal_task:
            journal_match = 1 if _norm_text(journal_pool) == _norm_text(journal_task) else 0

        meta_rows.append(
            {
                "doi": doi,
                "title_similarity": float(title_sim),
                "title_sim_ok": 1 if title_sim >= min_title_similarity else 0,
                "year_pool": year_pool,
                "year_task": year_task,
                "year_match": year_match,
                "journal_pool": journal_pool,
                "journal_task": journal_task,
                "journal_match": journal_match,
                "source": t.get("source", ""),
                "source_id": t.get("source_id", ""),
            }
        )

    meta_columns = [
        "doi",
        "title_similarity",
        "title_sim_ok",
        "year_pool",
        "year_task",
        "year_match",
        "journal_pool",
        "journal_task",
        "journal_match",
        "source",
        "source_id",
    ]
    log.info(
        "Phase5 overlap | unique_dois tasks=%d pool=%d intersection=%d",
        len(tasks_dois),
        len(pool_dois),
        len(intersection),
    )

    if meta_rows:
        meta_df = pd.DataFrame(meta_rows, columns=meta_columns)
        meta_df = meta_df.sort_values(
            ["title_sim_ok", "title_similarity"],
            ascending=[True, True],
        )
    else:
        meta_df = pd.DataFrame(columns=meta_columns)

    (out_dir / "phase5_overlap_metadata.csv").write_text(
        meta_df.to_csv(index=False),
        encoding="utf-8",
    )

    if positive_label_field:
        per_query_rows: list[dict[str, Any]] = []
        for qid, g in pool.groupby("query_id"):
            g_sorted = g.sort_values("rank")
            seen: set[str] = set()
            rels: list[int] = []

            for doi in g_sorted["doi_norm"].tolist():
                if not isinstance(doi, str) or not doi:
                    continue
                if doi in seen:
                    continue
                seen.add(doi)

                task = doi_to_task.get(doi)
                if task is None:
                    rels.append(0)
                    continue
                labels = task.get("labels")
                if not isinstance(labels, dict):
                    rels.append(0)
                    continue
                v = labels.get(positive_label_field)
                if isinstance(v, str) and v.strip().lower() == positive_yes_value.strip().lower():
                    rels.append(1)
                else:
                    rels.append(0)

            def precision_at_k(k: int) -> float:
                if k <= 0:
                    return 0.0
                rels_k = rels[:k]
                return float(sum(rels_k)) / float(k)

            def mrr() -> float:
                for i, r in enumerate(rels, start=1):
                    if r == 1:
                        return 1.0 / float(i)
                return 0.0

            row_out: dict[str, Any] = {
                "query_id": qid,
                "mrr": float(mrr()),
                "retrieved_unique_docs": int(len(rels)),
                "relevant_docs_in_retrieved": int(sum(rels)),
            }
            for k in k_values:
                row_out[f"precision@{int(k)}"] = float(precision_at_k(int(k)))
            per_query_rows.append(row_out)

        per_query_df = pd.DataFrame(per_query_rows).sort_values("query_id")
        (out_dir / "phase5_weaklabel_metrics_per_query.csv").write_text(
            per_query_df.to_csv(index=False), encoding="utf-8"
        )

        agg: dict[str, list[Any]] = {"metric": [], "value": []}
        agg["metric"].append("mrr")
        agg["value"].append(float(per_query_df["mrr"].mean()))
        for k in k_values:
            agg["metric"].append(f"precision@{int(k)}")
            agg["value"].append(float(per_query_df[f"precision@{int(k)}"].mean()))
        agg["metric"].append("queries_with_any_relevant_rate")
        agg["value"].append(float((per_query_df["relevant_docs_in_retrieved"] > 0).mean()))

        summary_df = pd.DataFrame(agg)
        (out_dir / "phase5_weaklabel_metrics_summary.csv").write_text(summary_df.to_csv(index=False), encoding="utf-8")
        (out_dir / "phase5_weaklabel_metrics_summary.json").write_text(
            json.dumps(summary_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
