from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from evaluation_metrics.src.clients.ollama_api import OllamaClient
from evaluation_metrics.src.clients.rag_api import RagApiClient
from evaluation_metrics.src.phases.phase1_pool import run_phase1_pool
from evaluation_metrics.src.phases.phase2_beir import compute_retrieval_metrics, write_retrieval_metrics
from evaluation_metrics.src.phases.phase2_overlap_audit import build_overlap_audit, discover_label_fields
from evaluation_metrics.src.phases.phase3_generate import run_phase3_generate
from evaluation_metrics.src.phases.phase4_ragas import run_ragas
from evaluation_metrics.src.phases.phase5_audit import build_audit_sample
from evaluation_metrics.src.schemas.models import RunContext


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    gitsha = os.environ.get("GIT_SHA", "nogit")
    return f"{ts}_{gitsha}"


def _init_run(config: dict[str, Any]) -> RunContext:
    run_id = _new_run_id()
    runs_dir = Path(config["paths"]["runs_dir"])
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return RunContext(run_id=run_id, run_dir=str(run_dir))


async def _cmd_phase1(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config))
    ctx = _init_run(config)
    rag = RagApiClient(
        base_url=config["rag_api"]["base_url"],
        timeout_seconds=float(config["rag_api"]["timeout_seconds"]),
    )
    await run_phase1_pool(
        ctx=ctx,
        rag=rag,
        queries_jsonl=Path(config["paths"]["queries_jsonl"]),
        top_k_pool=int(config["rag_api"]["search_top_k_pool"]),
    )


async def _cmd_phase3(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config))
    ctx = _init_run(config)
    rag = RagApiClient(
        base_url=config["rag_api"]["base_url"],
        timeout_seconds=float(config["rag_api"]["timeout_seconds"]),
    )
    ollama = OllamaClient(base_url=config["ollama"]["base_url"])
    await run_phase3_generate(
        ctx=ctx,
        rag=rag,
        ollama=ollama,
        queries_jsonl=Path(config["paths"]["queries_jsonl"]),
        search_top_k_context=int(config["rag_api"]["search_top_k_context"]),
        hyde_header_name=str(config["ask"]["hyde_header_name"]),
        hyde_header_value=str(config["ask"]["hyde_header_value"]),
        ollama_model=str(config["ollama"]["model"]),
        ollama_temperature=float(config["ollama"]["temperature"]),
        ollama_num_predict=int(config["ollama"]["num_predict"]),
    )


def _cmd_phase2(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config))
    ctx = _init_run(config)

    pool = Path(args.pool_jsonl)
    qrels = Path(args.qrels_tsv)
    k_values = [int(x) for x in config["metrics"]["k_values"]]

    summary_df, per_query_df = compute_retrieval_metrics(
        pool_jsonl=pool,
        qrels_tsv=qrels,
        k_values=k_values,
    )
    write_retrieval_metrics(
        out_dir=Path(ctx.run_dir),
        summary_df=summary_df,
        per_query_df=per_query_df,
    )


def _cmd_phase2_overlap(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config))
    ctx = _init_run(config)

    tasks_clean = Path(args.tasks_clean_json)
    k_values = [int(x) for x in config["metrics"]["k_values"]]

    if args.list_labels:
        fields = discover_label_fields(tasks_clean)
        for lf in fields:
            types = ", ".join(sorted(lf.types))
            logging.info("phase2 overlap label_field=%s types=%s", lf.name, types)
        return

    positive_label_field = str(args.positive_label_field or "").strip()
    if not positive_label_field:
        positive_label_field = None

    build_overlap_audit(
        phase1_pool_jsonl=Path(args.phase1_pool_jsonl),
        tasks_clean_json=tasks_clean,
        out_dir=Path(ctx.run_dir),
        k_values=k_values,
        positive_label_field=positive_label_field,
        positive_yes_value=str(args.positive_yes_value),
        min_title_similarity=float(args.min_title_similarity),
    )


def _cmd_phase4(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config))
    ctx = _init_run(config)

    in_path = Path(args.input_jsonl)
    out_csv = Path(ctx.run_dir) / f"phase4_ragas_{in_path.stem}.csv"

    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    os.environ["OPENAI_BASE_URL"] = str(config["ollama"]["base_url"]).rstrip("/") + "/v1"
    os.environ["EVAL_OLLAMA_MODEL"] = str(config["ollama"]["model"])

    embeddings_model = str(config["ragas"]["embeddings_model"])
    embeddings_device = str(config["ragas"]["embeddings_device"])
    embeddings_local_only = bool(config["ragas"].get("embeddings_local_only", False))

    run_ragas(
        input_jsonl=in_path,
        out_csv=out_csv,
        embeddings_model=embeddings_model,
        embeddings_device=embeddings_device,
        embeddings_local_only=embeddings_local_only,
    )


def _cmd_phase5(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config))
    ctx = _init_run(config)

    out = Path(ctx.run_dir) / "phase5_audit_sample.jsonl"
    build_audit_sample(
        rag_no_hyde_jsonl=Path(args.rag_no_hyde),
        rag_hyde_jsonl=Path(args.rag_hyde),
        llm_only_jsonl=Path(args.llm_only),
        out_jsonl=out,
        sample_size=int(config["audit"]["sample_size"]),
        seed=int(config["audit"]["seed"]),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="evaluation_metrics/config/eval.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s3 = sub.add_parser("phase1")
    s3.set_defaults(func=lambda a: asyncio.run(_cmd_phase1(a)))

    s6 = sub.add_parser("phase3")
    s6.set_defaults(func=lambda a: asyncio.run(_cmd_phase3(a)))

    s5 = sub.add_parser("phase2")
    s5.add_argument("--pool-jsonl", required=True)
    s5.add_argument("--qrels-tsv", required=True)
    s5.set_defaults(func=_cmd_phase2)

    s5o = sub.add_parser("phase2_overlap")
    s5o.add_argument("--phase1-pool-jsonl", required=True)
    s5o.add_argument("--tasks-clean-json", required=True)
    s5o.add_argument("--list-labels", action="store_true")
    s5o.add_argument("--positive-label-field", default="")
    s5o.add_argument("--positive-yes-value", default="Yes")
    s5o.add_argument("--min-title-similarity", default="0.92")
    s5o.set_defaults(func=_cmd_phase2_overlap)

    s7 = sub.add_parser("phase4")
    s7.add_argument("--input-jsonl", required=True)
    s7.set_defaults(func=_cmd_phase4)

    s8 = sub.add_parser("phase5")
    s8.add_argument("--rag-no-hyde", required=True)
    s8.add_argument("--rag-hyde", required=True)
    s8.add_argument("--llm-only", required=True)
    s8.set_defaults(func=_cmd_phase5)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
