# evaluation_metrics/src/phases/phase7_ragas.py

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable, List

from datasets import Dataset
from openai import AsyncOpenAI, OpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextPrecisionWithoutReference

log = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"Phase7 missing env var {name}")
    return v


def _probe_openai_json_mode(*, base_url: str, api_key: str, model: str) -> None:
    """
    If this fails, you should assume RAGAS grading will likely produce NaNs.
    """
    client = OpenAI(base_url=base_url, api_key=api_key)
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": 'Return ONLY valid JSON: {"ok": true}'}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = (r.choices[0].message.content or "").strip()
    json.loads(content)


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
    log.info("Phase7 start input=%s", input_jsonl)

    records: list[dict[str, Any]] = []
    empty_answer = 0
    empty_contexts = 0

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            answer = _extract_answer_text(rec)
            contexts = rec.get("contexts") or []

            if not answer:
                empty_answer += 1
            if not contexts:
                empty_contexts += 1

            records.append(
                {
                    "question": rec.get("question"),
                    "answer": answer,
                    "contexts": contexts,
                }
            )

    log.info(
        "Phase7 loaded_records=%d empty_answer=%d empty_contexts=%d",
        len(records),
        empty_answer,
        empty_contexts,
    )
    if not records:
        raise RuntimeError("Phase7 no records loaded")

    ds = Dataset.from_list(records)

    base_url = _require_env("OPENAI_BASE_URL").rstrip("/")
    api_key = _require_env("OPENAI_API_KEY")
    model = _require_env("EVAL_OLLAMA_MODEL")

    log.info("Phase7 evaluator base_url=%s model=%s", base_url, model)

    try:
        _probe_openai_json_mode(base_url=base_url, api_key=api_key, model=model)
        log.info("Phase7 json_mode_probe ok")
    except Exception as e:
        raise RuntimeError(
            "Phase7 evaluator failed JSON mode probe. "
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
        "Phase7 embeddings model=%s device=%s local_only=%s",
        embeddings_model,
        embeddings_device,
        embeddings_local_only,
    )
    embeddings = _build_ragas_embeddings(
        model=embeddings_model,
        device=embeddings_device,
        local_only=embeddings_local_only,
    )

    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm),
        LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    ]

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

    log.info("Phase7 running RAGAS")
    t0 = time.perf_counter()
    if run_config is not None:
        result = evaluate(ds, metrics=metrics, llm=evaluator_llm, embeddings=embeddings, run_config=run_config)
    else:
        result = evaluate(ds, metrics=metrics, llm=evaluator_llm, embeddings=embeddings)
    t1 = time.perf_counter()

    df = result.to_pandas()

    metric_cols = [
        c
        for c in df.columns
        if c not in {"question", "answer", "contexts", "user_input", "response", "retrieved_contexts"}
    ]
    if metric_cols and df[metric_cols].isna().all().all():
        raise RuntimeError(
            "Phase7 produced only NaNs. The evaluator calls likely succeeded but the grader parsing failed. "
            "Common causes are evaluator incompatibility or repeated timeouts."
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("Phase7 done duration_sec=%.2f rows=%d out=%s", t1 - t0, len(df), out_csv)
    return out_csv
