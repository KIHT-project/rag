# evaluation_metrics/src/phases/phase4_ragas.py

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable, List, Tuple

from datasets import Dataset
from openai import AsyncOpenAI, OpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextPrecisionWithoutReference

from evaluation_metrics.src.utils.output_writer import write_outputs

log = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"phase4 missing env var {name}")
    return v


def _probe_openai_json_mode(*, base_url: str, api_key: str, model: str) -> None:
    """
    Probe evaluator JSON compliance using a schema closer to what graders need.

    A trivial {"ok": true} probe is too weak. Some backends pass that test but fail
    once prompts get longer and schema constraints tighten.
    """
    client = OpenAI(base_url=base_url, api_key=api_key)

    prompt = (
        "Return ONLY valid JSON with EXACTLY these keys: "
        "scores (array of objects with keys metric:string and score:number), "
        "explanation (string). "
        "No extra keys. Example shape: "
        "{\"scores\":[{\"metric\":\"m\",\"score\":0.5}],\"explanation\":\"x\"}"
    )
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = (r.choices[0].message.content or "").strip()
    obj = json.loads(content)
    if not isinstance(obj, dict):
        raise ValueError("JSON probe did not return an object")
    if set(obj.keys()) != {"scores", "explanation"}:
        raise ValueError("JSON probe returned unexpected keys")
    scores = obj.get("scores")
    if not isinstance(scores, list) or not scores:
        raise ValueError("JSON probe scores is empty")
    first = scores[0]
    if not isinstance(first, dict) or "metric" not in first or "score" not in first:
        raise ValueError("JSON probe scores item invalid")


def _cap_contexts(
    contexts: list[str],
    *,
    max_contexts: int,
    max_chars_per_context: int,
    max_total_chars: int,
) -> Tuple[list[str], dict[str, Any]]:
    """Apply hard caps to reduce evaluator prompt size and failure rate."""
    raw_contexts = [c.strip() for c in contexts if isinstance(c, str) and c.strip()]
    raw_contexts = raw_contexts[:max_contexts]

    capped: list[str] = []
    total = 0
    for c in raw_contexts:
        if max_chars_per_context > 0 and len(c) > max_chars_per_context:
            c2 = c[:max_chars_per_context].rstrip()
        else:
            c2 = c
        if max_total_chars > 0 and total + len(c2) > max_total_chars:
            remaining = max_total_chars - total
            if remaining <= 0:
                break
            c2 = c2[:remaining].rstrip()
        capped.append(c2)
        total += len(c2)

    stats = {
        "contexts_in": len(contexts or []),
        "contexts_out": len(capped),
        "total_chars_out": total,
        "max_contexts": max_contexts,
        "max_chars_per_context": max_chars_per_context,
        "max_total_chars": max_total_chars,
    }
    return capped, stats


class _SentenceTransformersEmbeddings:
    """
    Minimal LangChain compatible embeddings interface.
    RAGAS 0.4.3 expects embed_query and embed_documents.
    """

    def __init__(self, *, model_name_or_path: str, device: str, local_only: bool) -> None:
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {"device": device}
        if local_only:
            kwargs["local_files_only"] = True

        self._model = SentenceTransformer(model_name_or_path, **kwargs)

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
        return vec.tolist()

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        texts_list = [t for t in texts]
        if not texts_list:
            return []
        vecs = self._model.encode(texts_list, convert_to_numpy=True, normalize_embeddings=False)
        return vecs.tolist()


def _build_ragas_embeddings(*, model: str, device: str, local_only: bool) -> Any:
    return _SentenceTransformersEmbeddings(
        model_name_or_path=model,
        device=device,
        local_only=local_only,
    )


def _extract_answer_text(rec: dict[str, Any]) -> str:
    if isinstance(rec.get("answer_text"), str):
        return rec["answer_text"]

    raw = rec.get("answer_raw")
    if isinstance(raw, dict):
        answer_obj = raw.get("answer")
        if isinstance(answer_obj, dict):
            summary = answer_obj.get("summary")
            if isinstance(summary, str):
                return summary
        if isinstance(answer_obj, str):
            return answer_obj
        output = raw.get("output")
        if isinstance(output, str):
            return output

    return ""


def run_ragas(
    *,
    input_jsonl: Path,
    out_csv: Path,
    embeddings_model: str,
    embeddings_device: str,
    embeddings_local_only: bool = False,
) -> Path:
    log.info("phase4 start input=%s", input_jsonl)

    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    empty_answer = 0
    empty_contexts = 0

    max_contexts = int(os.environ.get("RAGAS_MAX_CONTEXTS", "5"))
    max_chars_per_context = int(os.environ.get("RAGAS_MAX_CHARS_PER_CONTEXT", "2000"))
    max_total_context_chars = int(os.environ.get("RAGAS_MAX_TOTAL_CONTEXT_CHARS", "9000"))

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            answer = _extract_answer_text(rec)
            contexts_raw = rec.get("contexts") or []
            contexts, ctx_stats = _cap_contexts(
                list(contexts_raw) if isinstance(contexts_raw, list) else [],
                max_contexts=max_contexts,
                max_chars_per_context=max_chars_per_context,
                max_total_chars=max_total_context_chars,
            )

            if not answer:
                empty_answer += 1
            if not contexts:
                empty_contexts += 1

            qid = rec.get("query_id")
            records.append({"query_id": qid, "question": rec.get("question"), "answer": answer, "contexts": contexts})
            diagnostics.append({"query_id": qid, "ctx": ctx_stats, "contexts_source": rec.get("contexts_source")})

    log.info(
        "phase4 loaded_records=%d empty_answer=%d empty_contexts=%d",
        len(records),
        empty_answer,
        empty_contexts,
    )
    if not records:
        raise RuntimeError("phase4 no records loaded")

    ds = Dataset.from_list(records)

    base_url = _require_env("OPENAI_BASE_URL").rstrip("/")
    api_key = _require_env("OPENAI_API_KEY")
    model = _require_env("EVAL_OLLAMA_MODEL")

    log.info("phase4 evaluator base_url=%s model=%s", base_url, model)

    try:
        _probe_openai_json_mode(base_url=base_url, api_key=api_key, model=model)
        log.info("phase4 json_mode_probe ok")
    except Exception as e:
        raise RuntimeError(
            "phase4 evaluator failed JSON mode probe. "
            "Your backend is not OpenAI JSON mode compatible, so RAGAS graders commonly return NaNs. "
            "Fix the server or switch evaluator to a model that supports response_format json_object."
        ) from e

    timeout_seconds = float(os.environ.get("RAGAS_EVAL_TIMEOUT_SEC", "240"))

    evaluator_client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout_seconds,
        max_retries=0,
    )

    evaluator_llm = llm_factory(model=model, client=evaluator_client)

    log.info(
        "phase4 embeddings model=%s device=%s local_only=%s",
        embeddings_model,
        embeddings_device,
        embeddings_local_only,
    )
    embeddings = _build_ragas_embeddings(
        model=embeddings_model,
        device=embeddings_device,
        local_only=embeddings_local_only,
    )

    # Choose metrics based on data availability.
    # If contexts are missing, context based metrics are undefined and will just produce NaNs.
    all_empty_contexts = all((r.get("contexts") or []) == [] for r in records)
    if all_empty_contexts:
        metrics = [AnswerRelevancy(llm=evaluator_llm)]
        log.info("phase4 metrics=answer_relevancy only (no contexts in input)")
    else:
        metrics = [Faithfulness(llm=evaluator_llm), AnswerRelevancy(llm=evaluator_llm), LLMContextPrecisionWithoutReference(llm=evaluator_llm)]
        log.info("phase4 metrics=faithfulness, answer_relevancy, llm_context_precision_wo_ref")

    run_config = None
    try:
        from ragas.run_config import RunConfig

        run_config = RunConfig(
            timeout=timeout_seconds,
            max_workers=1,
            max_wait=timeout_seconds,
            max_retries=0,
        )
    except Exception:
        run_config = None

    log.info("phase4 running RAGAS")
    t0 = time.perf_counter()
    if run_config is not None:
        result = evaluate(ds, metrics=metrics, llm=evaluator_llm, embeddings=embeddings, run_config=run_config)
    else:
        result = evaluate(ds, metrics=metrics, llm=evaluator_llm, embeddings=embeddings)
    t1 = time.perf_counter()

    df = result.to_pandas()

    # Write diagnostics alongside results to make NaNs debuggable.
    diag_path = out_csv.with_suffix(".diagnostics.jsonl")
    try:
        with diag_path.open("w", encoding="utf-8") as f:
            for d in diagnostics:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        log.info("phase4 diagnostics written %s", diag_path)
    except Exception:
        log.exception("phase4 failed to write diagnostics")

    metric_cols = [
        c
        for c in df.columns
        if c
        not in {
            "query_id",
            "question",
            "answer",
            "contexts",
            "user_input",
            "response",
            "retrieved_contexts",
        }
    ]
    if metric_cols and df[metric_cols].isna().all().all():
        raise RuntimeError(
            "phase4 produced only NaNs. The evaluator calls likely succeeded but the grader parsing failed. "
            "Common causes are evaluator incompatibility or repeated timeouts."
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_outputs(df, out_csv)
    log.info("phase4 done duration_sec=%.2f rows=%d out=%s", t1 - t0, len(df), out_csv)
    return out_csv
