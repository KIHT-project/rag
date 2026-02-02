from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from evaluation_metrics.src.utils.output_writer import write_outputs

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def _precision_at_k(rels: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    rels_k = rels[:k]
    return sum(rels_k) / float(k)


def _recall_at_k(rels: list[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0 or k <= 0:
        return 0.0
    rels_k = rels[:k]
    return min(1.0, sum(rels_k) / float(total_relevant))


def _mrr(rels: list[int]) -> float:
    for idx, r in enumerate(rels, start=1):
        if r > 0:
            return 1.0 / float(idx)
    return 0.0


def compute_retrieval_metrics(
    *,
    pool_jsonl: Path,
    qrels_tsv: Path,
    k_values: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("phase2 | loading retrieval pool from %s", pool_jsonl)
    pool = pd.read_json(pool_jsonl, lines=True)
    log.info("phase2 | pool loaded, rows=%d, unique_queries=%d",
             len(pool), pool["query_id"].nunique())

    log.info("phase2 | loading qrels from %s", qrels_tsv)
    qrels = pd.read_csv(
        qrels_tsv,
        sep="\t",
        header=None,
        names=["query_id", "doc_id", "relevance"],
    )
    qrels["relevance"] = qrels["relevance"].astype(int)
    log.info("phase2 | qrels loaded, rows=%d, unique_queries=%d, unique_docs=%d",
             len(qrels), qrels["query_id"].nunique(), qrels["doc_id"].nunique())

    rel_map = {(r.query_id, r.doc_id): r.relevance for r in qrels.itertuples(index=False)}
    total_rel = qrels.groupby("query_id")["relevance"].sum().to_dict()

    per_query_rows: list[dict] = []
    zero_relevance_queries = 0

    log.info("phase2 | computing metrics for %d queries", pool["query_id"].nunique())

    for qid, g in pool.groupby("query_id"):
        g_sorted = g.sort_values("rank")
        rels = [
            1 if rel_map.get((qid, doc_id), 0) > 0 else 0
            for doc_id in g_sorted["doc_id"].tolist()
        ]

        total_relevant = int(total_rel.get(qid, 0))
        if total_relevant == 0:
            zero_relevance_queries += 1

        mrr = _mrr(rels)

        per_query = {
            "query_id": qid,
            "mrr": mrr,
            "total_relevant": total_relevant,
        }

        for k in k_values:
            per_query[f"precision@{k}"] = _precision_at_k(rels, k)
            per_query[f"recall@{k}"] = _recall_at_k(rels, total_relevant, k)

        log.debug(
            "phase2 | query=%s total_relevant=%d mrr=%.3f",
            qid, total_relevant, mrr
        )

        per_query_rows.append(per_query)

    log.info(
        "phase2 | queries with zero relevant docs: %d / %d",
        zero_relevance_queries,
        pool["query_id"].nunique(),
    )

    per_query_df = pd.DataFrame(per_query_rows).sort_values("query_id")

    summary = {"metric": [], "value": []}

    summary["metric"].append("mrr")
    summary["value"].append(float(per_query_df["mrr"].mean()))

    for k in k_values:
        summary["metric"].append(f"precision@{k}")
        summary["value"].append(float(per_query_df[f"precision@{k}"].mean()))
        summary["metric"].append(f"recall@{k}")
        summary["value"].append(float(per_query_df[f"recall@{k}"].mean()))

    summary_df = pd.DataFrame(summary)

    log.info("phase2 | aggregate metrics computed")
    for _, row in summary_df.iterrows():
        log.info("phase2 | %s = %.4f", row["metric"], row["value"])

    return summary_df, per_query_df


def write_retrieval_metrics(
    *,
    out_dir: Path,
    summary_df: pd.DataFrame,
    per_query_df: pd.DataFrame,
) -> None:
    log.info("phase2 | writing metrics to %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_outputs(summary_df, out_dir / "phase2_retrieval_metrics_summary.csv")
    write_outputs(per_query_df, out_dir / "phase2_retrieval_metrics_per_query.csv")

    log.info("phase2 | metrics written successfully")
