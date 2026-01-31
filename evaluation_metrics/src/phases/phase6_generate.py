from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from evaluation_metrics.src.clients.ollama_api import OllamaClient
from evaluation_metrics.src.clients.rag_api import RagApiClient
from evaluation_metrics.src.schemas.models import QueryItem, RunContext

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


LLM_ONLY_SYSTEM_PROMPT = (
    "You are a biomedical assistant. Answer the question as accurately as possible. "
    "If you are uncertain or evidence is insufficient, state limitations explicitly."
)


def _cap(s: str, n: int = 160) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


async def run_phase6_generate(
    *,
    ctx: RunContext,
    rag: RagApiClient,
    ollama: OllamaClient,
    queries_jsonl: Path,
    search_top_k_context: int,
    hyde_header_name: str,
    hyde_header_value: str,
    ollama_model: str,
    ollama_temperature: float,
    ollama_num_predict: int,
    filters: Optional[dict[str, Any]] = None,
) -> dict[str, Path]:
    out_rag = Path(ctx.run_dir) / "phase6_answers_rag_no_hyde.jsonl"
    out_hyde = Path(ctx.run_dir) / "phase6_answers_rag_hyde.jsonl"
    out_llm = Path(ctx.run_dir) / "phase6_answers_llm_only.jsonl"

    out_rag.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "Phase6 | start | run_id=%s | queries=%s | ctx_top_k=%d | model=%s",
        ctx.run_id,
        str(queries_jsonl),
        search_top_k_context,
        ollama_model,
    )

    total = 0
    failures = 0

    with queries_jsonl.open("r", encoding="utf-8") as f_in, \
        out_rag.open("w", encoding="utf-8") as f_rag, \
        out_hyde.open("w", encoding="utf-8") as f_hyde, \
        out_llm.open("w", encoding="utf-8") as f_llm:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            q = QueryItem.model_validate_json(line)
            total += 1
            qid = q.id
            question = q.text

            log.info("Phase6 | q=%d | query_id=%s | %s", total, qid, _cap(question, 120))

            try:
                t0 = time.perf_counter()
                search_resp = await rag.search(query=question, top_k=search_top_k_context, filters=filters)
                contexts = [h.content_text for h in search_resp.hits if h.content_text]
                t1 = time.perf_counter()
                log.info(
                    "Phase6 | query_id=%s | contexts=%d | search_ms=%.2f",
                    qid,
                    len(contexts),
                    (t1 - t0) * 1000.0,
                )

                t2 = time.perf_counter()
                rag_resp = await rag.ask(
                    question=question,
                    filters=filters,
                    hyde_enabled=False,
                    hyde_header_name=hyde_header_name,
                    hyde_header_value=hyde_header_value,
                )
                t3 = time.perf_counter()

                rec_rag = {
                    "query_id": qid,
                    "question": question,
                    "mode": "rag_no_hyde",
                    "answer_raw": rag_resp.raw,
                    "contexts": contexts,
                }
                f_rag.write(json.dumps(rec_rag, ensure_ascii=False) + "\n")

                preview = ""
                if isinstance(rag_resp.raw, dict):
                    ans = rag_resp.raw.get("answer")
                    if isinstance(ans, dict):
                        preview = ans.get("summary") or ""
                    elif isinstance(ans, str):
                        preview = ans
                log.info(
                    "Phase6 | query_id=%s | rag_no_hyde_ms=%.2f | answer_preview=%s",
                    qid,
                    (t3 - t2) * 1000.0,
                    _cap(preview),
                )

                t4 = time.perf_counter()
                hyde_resp = await rag.ask(
                    question=question,
                    filters=filters,
                    hyde_enabled=True,
                    hyde_header_name=hyde_header_name,
                    hyde_header_value=hyde_header_value,
                )
                t5 = time.perf_counter()

                rec_hyde = {
                    "query_id": qid,
                    "question": question,
                    "mode": "rag_hyde",
                    "answer_raw": hyde_resp.raw,
                    "contexts": contexts,
                }
                f_hyde.write(json.dumps(rec_hyde, ensure_ascii=False) + "\n")

                preview = ""
                if isinstance(hyde_resp.raw, dict):
                    ans = hyde_resp.raw.get("answer")
                    if isinstance(ans, dict):
                        preview = ans.get("summary") or ""
                    elif isinstance(ans, str):
                        preview = ans
                log.info(
                    "Phase6 | query_id=%s | rag_hyde_ms=%.2f | answer_preview=%s",
                    qid,
                    (t5 - t4) * 1000.0,
                    _cap(preview),
                )

                t6 = time.perf_counter()
                llm_text = await ollama.chat(
                    model=ollama_model,
                    system=LLM_ONLY_SYSTEM_PROMPT,
                    user=question,
                    temperature=ollama_temperature,
                    num_predict=ollama_num_predict,
                )
                t7 = time.perf_counter()

                rec_llm = {
                    "query_id": qid,
                    "question": question,
                    "mode": "llm_only",
                    "answer_text": llm_text,
                    "contexts": [],
                }
                f_llm.write(json.dumps(rec_llm, ensure_ascii=False) + "\n")

                log.info(
                    "Phase6 | query_id=%s | llm_only_ms=%.2f | answer_preview=%s",
                    qid,
                    (t7 - t6) * 1000.0,
                    _cap(llm_text),
                )

                if total % 5 == 0:
                    log.info("Phase6 | progress | done=%d", total)

            except Exception as e:
                failures += 1
                log.exception("Phase6 | query_id=%s | failed | error=%s", qid, str(e))
                continue

    log.info(
        "Phase6 | done | total=%d | failures=%d | out_dir=%s",
        total,
        failures,
        ctx.run_dir,
    )

    return {"rag_no_hyde": out_rag, "rag_hyde": out_hyde, "llm_only": out_llm}
