from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from evaluation_metrics.src.clients.rag_api import RagApiClient
from evaluation_metrics.src.schemas.models import RunContext, QueryItem

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)



async def run_phase3_pool(
    *,
    ctx: RunContext,
    rag: RagApiClient,
    queries_jsonl: Path,
    top_k_pool: int,
    filters: Optional[dict[str, Any]] = None,
) -> Path:
    out_path = Path(ctx.run_dir) / "phase3_pool.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "Phase3 | start | run_id=%s | queries_file=%s | top_k=%d",
        ctx.run_id,
        str(queries_jsonl),
        top_k_pool,
    )

    total_queries = 0
    total_hits = 0
    failures = 0
    t_start = time.perf_counter()

    with queries_jsonl.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            q = QueryItem.model_validate_json(line)
            total_queries += 1

            log.info(
                "Phase3 | query=%d | query_id=%s",
                total_queries,
                q.id,
            )

            try:
                t0 = time.perf_counter()
                resp = await rag.search(
                    query=q.text,
                    top_k=top_k_pool,
                    filters=filters,
                )
                t1 = time.perf_counter()

                hits = resp.hits or []
                total_hits += len(hits)

                if not hits:
                    log.warning(
                        "Phase3 | query_id=%s | no hits returned",
                        q.id,
                    )

                log.info(
                    "Phase3 | query_id=%s | hits=%d | search_ms=%.2f",
                    q.id,
                    len(hits),
                    (t1 - t0) * 1000.0,
                )

                for rank, hit in enumerate(hits, start=1):
                    rec = {
                        "query_id": q.id,
                        "question": q.text,
                        "rank": rank,
                        "score": hit.score,
                        "doc_id": hit.doc_id,
                        "doi": hit.doi,
                        "title": hit.title,
                        "authors": ", ".join(hit.authors)
                        if isinstance(hit.authors, list)
                        else hit.authors,
                        "year": hit.year,
                        "journal": hit.journal,
                        "content_text": hit.content_text,
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if total_queries % 10 == 0:
                    log.info(
                        "Phase3 | progress | queries_done=%d",
                        total_queries,
                    )

            except Exception as e:
                failures += 1
                log.exception(
                    "Phase3 | query_id=%s | failed | error=%s",
                    q.id,
                    str(e),
                )
                continue

    t_end = time.perf_counter()

    log.info(
        "Phase3 | done | total_queries=%d | total_hits=%d | failures=%d | duration_sec=%.2f | out=%s",
        total_queries,
        total_hits,
        failures,
        t_end - t_start,
        str(out_path),
    )

    return out_path
