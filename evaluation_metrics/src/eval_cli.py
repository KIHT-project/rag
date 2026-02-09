from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from evaluation_metrics.src.audit import EvaluationPostgresAudit
from evaluation_metrics.src.clients.ollama_api import OllamaClient
from evaluation_metrics.src.clients.rag_api import RagApiClient
from evaluation_metrics.src.phases.phase1_pool import run_phase1_pool
from evaluation_metrics.src.phases.phase2_beir import compute_retrieval_metrics, write_retrieval_metrics
from evaluation_metrics.src.phases.phase2_overlap_audit import build_overlap_audit, discover_label_fields
from evaluation_metrics.src.phases.phase3_generate import run_phase3_generate
from evaluation_metrics.src.phases.phase4_ragas import run_ragas
from evaluation_metrics.src.phases.phase5_audit import build_audit_sample
from evaluation_metrics.src.phases.phase6_extraction import run_phase6_extraction
from evaluation_metrics.src.schemas.models import RunContext


def _run_async(coro: Any) -> Any:
    """
    Run a coroutine in an isolated event loop.

    We intentionally avoid asyncio.run() because Python 3.14 + nest_asyncio can
    raise during shutdown_default_executor() in CLI teardown.
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        asyncio.set_event_loop(None)
        loop.close()


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


def _init_sub_run(parent: RunContext, *, name: str) -> RunContext:
    run_dir = Path(parent.run_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(run_id=f"{parent.run_id}_{name}", run_dir=str(run_dir))


def _summarize_phase4_json(path: Path) -> dict[str, float]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError(f"phase4 output is not a list: {path}")

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if not isinstance(row, dict):
            continue
        for k, v in row.items():
            if isinstance(v, (int, float)):
                sums[k] += float(v)
                counts[k] += 1

    out: dict[str, float] = {}
    for k, total in sums.items():
        n = counts.get(k, 0)
        if n > 0:
            out[k] = total / n
    return out


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var**0.5


def _resolve_enabled_modes(paper_cfg: dict[str, Any]) -> set[str]:
    raw = paper_cfg.get("modes")
    default = {"rag_no_hyde", "rag_hyde", "llm_only"}
    if raw is None:
        return default
    if not isinstance(raw, list):
        raise RuntimeError("paper.modes must be a list of mode names")
    out = {str(v).strip() for v in raw if str(v).strip()}
    if not out:
        raise RuntimeError("paper.modes cannot be empty")
    return out


def _resolve_max_queries(paper_cfg: dict[str, Any]) -> int | None:
    raw = paper_cfg.get("max_queries")
    if raw is None:
        return None
    try:
        val = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("paper.max_queries must be an integer") from exc
    if val <= 0:
        return None
    return val


def _audit_dsn(config: dict[str, Any]) -> str:
    env = os.environ.get("EVAL_AUDIT_POSTGRES_DSN", "").strip()
    if env:
        return env
    audit_cfg = config.get("audit", {})
    dsn = str(audit_cfg.get("postgres_dsn", "")).strip() if isinstance(audit_cfg, dict) else ""
    if not dsn:
        raise RuntimeError(
            "Missing audit Postgres DSN. Set EVAL_AUDIT_POSTGRES_DSN or audit.postgres_dsn in eval.yaml."
        )
    return dsn


def _resolve_extraction_cfg(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("extraction", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RuntimeError("extraction must be a mapping in eval config")
    return raw


def _resolve_tasks_clean_path(extraction_cfg: dict[str, Any]) -> Path:
    raw = extraction_cfg.get("tasks_clean_json", "evaluation_metrics/tasks_clean.json")
    out = Path(str(raw))
    if not out.exists():
        raise RuntimeError(f"tasks_clean_json not found: {out}")
    return out


def _run_phase6_for_outputs(
    *,
    config: dict[str, Any],
    outputs: dict[str, Path],
    out_dir: Path,
) -> dict[str, dict[str, float]]:
    extraction_cfg = _resolve_extraction_cfg(config)
    if not bool(extraction_cfg.get("enabled", False)):
        return {}

    include_modes_raw = extraction_cfg.get("include_modes", [])
    include_modes: set[str] | None = None
    if isinstance(include_modes_raw, list) and include_modes_raw:
        include_modes = {str(x).strip() for x in include_modes_raw if str(x).strip()}

    tasks_clean_json = _resolve_tasks_clean_path(extraction_cfg)

    per_mode_summary: dict[str, dict[str, float]] = {}
    for mode, in_path in outputs.items():
        if include_modes is not None and mode not in include_modes:
            continue

        summary = run_phase6_extraction(
            input_jsonl=in_path,
            tasks_clean_json=tasks_clean_json,
            out_dir=out_dir,
            reports_risk_field=str(extraction_cfg.get("reports_risk_field", "reports_risk_factors")),
            reports_positive_value=str(extraction_cfg.get("reports_positive_value", "Yes")),
            reports_confidence_field=str(
                extraction_cfg.get("reports_confidence_field", "confidence_reports_risk_factors")
            ),
            min_confidence=int(extraction_cfg.get("min_confidence", 3)),
            allow_missing_confidence=bool(extraction_cfg.get("allow_missing_confidence", True)),
            factor_source_field=str(extraction_cfg.get("factor_source_field", "reason_label")),
            factor_source_fallback_to_abstract=bool(
                extraction_cfg.get("factor_source_fallback_to_abstract", True)
            ),
            no_gold_policy=str(extraction_cfg.get("no_gold_policy", "skip")),
            pred_closed_set_only=bool(extraction_cfg.get("pred_closed_set_only", False)),
            pred_include_aliases=bool(extraction_cfg.get("pred_include_aliases", True)),
            pred_include_summary_factors=bool(
                extraction_cfg.get("pred_include_summary_factors", False)
            ),
            pred_include_citation_snippets=bool(
                extraction_cfg.get("pred_include_citation_snippets", False)
            ),
            canonical_factors=(
                extraction_cfg.get("canonical_factors")
                if isinstance(extraction_cfg.get("canonical_factors"), dict)
                else None
            ),
        )
        per_mode_summary[mode] = summary

    return per_mode_summary


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
    try:
        await rag.probe()
    except Exception as e:
        raise RuntimeError(
            f"RAG API unreachable at {config['rag_api']['base_url']}. "
            "Start the API service and retry."
        ) from e
    try:
        await ollama.probe()
    except Exception as e:
        raise RuntimeError(
            f"Ollama API unreachable at {config['ollama']['base_url']}. "
            "Start/reach Ollama and retry."
        ) from e
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
    ragas_metrics = config["ragas"].get("metrics")

    run_ragas(
        input_jsonl=in_path,
        out_csv=out_csv,
        embeddings_model=embeddings_model,
        embeddings_device=embeddings_device,
        embeddings_local_only=embeddings_local_only,
        metric_names=[str(m) for m in ragas_metrics] if isinstance(ragas_metrics, list) else None,
    )


def _run_phase4_for_outputs(
    *,
    config: dict[str, Any],
    outputs: dict[str, Path],
    out_dir: Path,
) -> dict[str, dict[str, float]]:
    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    os.environ["OPENAI_BASE_URL"] = str(config["ollama"]["base_url"]).rstrip("/") + "/v1"
    os.environ["EVAL_OLLAMA_MODEL"] = str(config["ollama"]["model"])

    embeddings_model = str(config["ragas"]["embeddings_model"])
    embeddings_device = str(config["ragas"]["embeddings_device"])
    embeddings_local_only = bool(config["ragas"].get("embeddings_local_only", False))
    ragas_metrics = config["ragas"].get("metrics")

    per_mode_summary: dict[str, dict[str, float]] = {}
    for mode, in_path in outputs.items():
        out_csv = out_dir / f"phase4_ragas_{in_path.stem}.csv"
        run_ragas(
            input_jsonl=in_path,
            out_csv=out_csv,
            embeddings_model=embeddings_model,
            embeddings_device=embeddings_device,
            embeddings_local_only=embeddings_local_only,
            metric_names=[str(m) for m in ragas_metrics] if isinstance(ragas_metrics, list) else None,
        )
        per_mode_summary[mode] = _summarize_phase4_json(out_csv.with_suffix(".json"))
    return per_mode_summary


async def _cmd_paper(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config))
    ctx = _init_run(config)
    request_id = os.environ.get("EVAL_REQUEST_ID")
    audit = EvaluationPostgresAudit(dsn=_audit_dsn(config))
    await audit.start()

    paper_cfg = config.get("paper", {})
    primary_seed = paper_cfg.get("primary_seed")
    primary_temperature = float(paper_cfg.get("primary_temperature", 0.0))
    robustness_temperature = float(paper_cfg.get("robustness_temperature", 0.2))
    robustness_seeds = list(paper_cfg.get("robustness_seeds", [11, 22, 33, 44, 55]))
    enabled_modes = _resolve_enabled_modes(paper_cfg)
    max_queries = _resolve_max_queries(paper_cfg)
    model_name = str(config["ollama"]["model"])
    queries_path = str(config["paths"]["queries_jsonl"])

    await audit.create_run(
        run_id=ctx.run_id,
        request_id=request_id,
        run_type="PAPER",
        trigger_source="CLI",
        dataset_name=queries_path,
        dataset_version=None,
        config_snapshot=config,
        model_provider="ollama",
        model_name=model_name,
        model_params={
            "primary_seed": primary_seed,
            "primary_temperature": primary_temperature,
            "robustness_temperature": robustness_temperature,
            "robustness_seeds": robustness_seeds,
            "enabled_modes": sorted(enabled_modes),
            "max_queries": max_queries,
            "num_predict": int(config["ollama"]["num_predict"]),
        },
        seed=int(primary_seed) if primary_seed is not None else None,
    )
    await audit.create_event(
        run_id=ctx.run_id,
        request_id=request_id,
        event_type="EVAL_RUN_STARTED",
        status="STARTED",
        phase="PAPER",
        message="paper evaluation run started",
        payload={"run_dir": ctx.run_dir},
    )

    rag = RagApiClient(
        base_url=config["rag_api"]["base_url"],
        timeout_seconds=float(config["rag_api"]["timeout_seconds"]),
    )
    ollama = OllamaClient(base_url=config["ollama"]["base_url"])
    try:
        try:
            await rag.probe()
        except Exception as e:
            raise RuntimeError(
                f"RAG API unreachable at {config['rag_api']['base_url']}. "
                "Start the API service and retry."
            ) from e
        try:
            await ollama.probe()
        except Exception as e:
            raise RuntimeError(
                f"Ollama API unreachable at {config['ollama']['base_url']}. "
                "Start/reach Ollama and retry."
            ) from e

        await audit.create_event(
            run_id=ctx.run_id,
            request_id=request_id,
            event_type="EVAL_PHASE_STARTED",
            status="STARTED",
            phase="PRIMARY",
            message="primary deterministic run started",
            payload={"modes": sorted(enabled_modes), "max_queries": max_queries},
        )

        primary_ctx = _init_sub_run(ctx, name="primary_deterministic")
        primary_outputs = await run_phase3_generate(
            ctx=primary_ctx,
            rag=rag,
            ollama=ollama,
            queries_jsonl=Path(config["paths"]["queries_jsonl"]),
            search_top_k_context=int(config["rag_api"]["search_top_k_context"]),
            hyde_header_name=str(config["ask"]["hyde_header_name"]),
            hyde_header_value=str(config["ask"]["hyde_header_value"]),
            ollama_model=str(config["ollama"]["model"]),
            ollama_temperature=primary_temperature,
            ollama_num_predict=int(config["ollama"]["num_predict"]),
            ollama_seed=int(primary_seed) if primary_seed is not None else None,
            enabled_modes=enabled_modes,
            max_queries=max_queries,
        )
        primary_summary = _run_phase4_for_outputs(
            config=config, outputs=primary_outputs, out_dir=Path(primary_ctx.run_dir)
        )
        primary_extraction_summary = _run_phase6_for_outputs(
            config=config, outputs=primary_outputs, out_dir=Path(primary_ctx.run_dir)
        )
        for mode, metrics in primary_extraction_summary.items():
            mode_metrics = primary_summary.setdefault(mode, {})
            for metric, value in metrics.items():
                mode_metrics[f"extraction_{metric}"] = float(value)

        for mode, metrics in primary_summary.items():
            for metric, value in metrics.items():
                await audit.create_metric(
                    run_id=ctx.run_id,
                    request_id=request_id,
                    phase="PRIMARY",
                    metric_name=f"{mode}.{metric}",
                    metric_value=float(value),
                    seed=int(primary_seed) if primary_seed is not None else None,
                    aggregation="mean",
                    metadata={"run_type": "primary_deterministic"},
                )
        await audit.create_event(
            run_id=ctx.run_id,
            request_id=request_id,
            event_type="EVAL_PHASE_COMPLETED",
            status="COMPLETED",
            phase="PRIMARY",
            message="primary deterministic run completed",
            payload={"summary": primary_summary},
        )

        robust_root = _init_sub_run(ctx, name="robustness")
        all_runs: list[dict[str, Any]] = []
        metrics_acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        await audit.create_event(
            run_id=ctx.run_id,
            request_id=request_id,
            event_type="EVAL_PHASE_STARTED",
            status="STARTED",
            phase="ROBUSTNESS",
            message="robustness runs started",
            payload={"seeds": [int(s) for s in robustness_seeds]},
        )
        for seed_raw in robustness_seeds:
            seed = int(seed_raw)
            seed_ctx = _init_sub_run(robust_root, name=f"seed_{seed}")
            outputs = await run_phase3_generate(
                ctx=seed_ctx,
                rag=rag,
                ollama=ollama,
                queries_jsonl=Path(config["paths"]["queries_jsonl"]),
                search_top_k_context=int(config["rag_api"]["search_top_k_context"]),
                hyde_header_name=str(config["ask"]["hyde_header_name"]),
                hyde_header_value=str(config["ask"]["hyde_header_value"]),
                ollama_model=str(config["ollama"]["model"]),
                ollama_temperature=robustness_temperature,
                ollama_num_predict=int(config["ollama"]["num_predict"]),
                ollama_seed=seed,
                enabled_modes=enabled_modes,
                max_queries=max_queries,
            )
            run_summary = _run_phase4_for_outputs(
                config=config, outputs=outputs, out_dir=Path(seed_ctx.run_dir)
            )
            run_extraction_summary = _run_phase6_for_outputs(
                config=config, outputs=outputs, out_dir=Path(seed_ctx.run_dir)
            )
            for mode, metrics in run_extraction_summary.items():
                mode_metrics = run_summary.setdefault(mode, {})
                for metric, value in metrics.items():
                    mode_metrics[f"extraction_{metric}"] = float(value)
            all_runs.append({"seed": seed, "summary": run_summary, "run_dir": seed_ctx.run_dir})
            for mode, metrics in run_summary.items():
                for metric, value in metrics.items():
                    metrics_acc[mode][metric].append(float(value))

        aggregate: dict[str, dict[str, dict[str, Any]]] = {}
        for mode, metric_vals in metrics_acc.items():
            aggregate[mode] = {}
            for metric, values in metric_vals.items():
                mean, std = _mean_std(values)
                aggregate[mode][metric] = {"mean": mean, "std": std, "n": len(values)}
                await audit.create_metric(
                    run_id=ctx.run_id,
                    request_id=request_id,
                    phase="ROBUSTNESS",
                    metric_name=f"{mode}.{metric}",
                    metric_value=mean,
                    seed=None,
                    aggregation="mean",
                    metadata={"std": std, "n": len(values)},
                )

        await audit.create_event(
            run_id=ctx.run_id,
            request_id=request_id,
            event_type="EVAL_PHASE_COMPLETED",
            status="COMPLETED",
            phase="ROBUSTNESS",
            message="robustness runs completed",
            payload={"aggregate": aggregate},
        )

        out_summary = {
            "primary_deterministic": {
                "seed": primary_seed,
                "temperature": primary_temperature,
                "summary": primary_summary,
                "run_dir": primary_ctx.run_dir,
            },
            "robustness": {
                "temperature": robustness_temperature,
                "seeds": [int(s) for s in robustness_seeds],
                "runs": all_runs,
                "aggregate": aggregate,
                "run_dir": robust_root.run_dir,
            },
        }

        out_path = Path(ctx.run_dir) / "paper_benchmark_summary.json"
        content_raw = json.dumps(out_summary, ensure_ascii=False, indent=2)
        out_path.write_text(content_raw, encoding="utf-8")
        await audit.create_artifact(
            run_id=ctx.run_id,
            request_id=request_id,
            artifact_type="REPORT",
            artifact_name=out_path.name,
            mime_type="application/json",
            content_raw=content_raw,
            metadata={"path": str(out_path)},
        )
        await audit.create_event(
            run_id=ctx.run_id,
            request_id=request_id,
            event_type="EVAL_ARTIFACT_SAVED",
            status="COMPLETED",
            phase="SUMMARY",
            message="paper summary saved",
            payload={"path": str(out_path)},
        )
        await audit.complete_run(run_id=ctx.run_id, status="SUCCESS")
        await audit.create_event(
            run_id=ctx.run_id,
            request_id=request_id,
            event_type="EVAL_RUN_COMPLETED",
            status="COMPLETED",
            phase="PAPER",
            message="paper evaluation run completed",
            payload={"run_dir": ctx.run_dir},
        )
        logging.info("paper benchmark summary written %s", out_path)
    except Exception as exc:
        event_id = await audit.create_event(
            run_id=ctx.run_id,
            request_id=request_id,
            event_type="EXCEPTION_RAISED",
            status="ERROR",
            phase="PAPER",
            message=str(exc),
            payload={"exception": type(exc).__name__},
            stacktrace_raw="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        error_id = await audit.create_error(
            run_id=ctx.run_id,
            request_id=request_id,
            event_id=event_id,
            exc=exc,
            phase="PAPER",
            error_code="paper_run_failed",
        )
        await audit.complete_run(run_id=ctx.run_id, status="ERROR", error_id=error_id)
        raise
    finally:
        await audit.close()


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


def _cmd_phase6(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config))
    ctx = _init_run(config)
    extraction_cfg = _resolve_extraction_cfg(config)

    tasks_clean = (
        Path(args.tasks_clean_json)
        if str(args.tasks_clean_json or "").strip()
        else _resolve_tasks_clean_path(extraction_cfg)
    )

    run_phase6_extraction(
        input_jsonl=Path(args.input_jsonl),
        tasks_clean_json=tasks_clean,
        out_dir=Path(ctx.run_dir),
        reports_risk_field=str(extraction_cfg.get("reports_risk_field", "reports_risk_factors")),
        reports_positive_value=str(extraction_cfg.get("reports_positive_value", "Yes")),
        reports_confidence_field=str(
            extraction_cfg.get("reports_confidence_field", "confidence_reports_risk_factors")
        ),
        min_confidence=int(extraction_cfg.get("min_confidence", 3)),
        allow_missing_confidence=bool(extraction_cfg.get("allow_missing_confidence", True)),
        factor_source_field=str(extraction_cfg.get("factor_source_field", "reason_label")),
        factor_source_fallback_to_abstract=bool(
            extraction_cfg.get("factor_source_fallback_to_abstract", True)
        ),
        no_gold_policy=str(extraction_cfg.get("no_gold_policy", "skip")),
        pred_closed_set_only=bool(extraction_cfg.get("pred_closed_set_only", False)),
        pred_include_aliases=bool(extraction_cfg.get("pred_include_aliases", True)),
        pred_include_summary_factors=bool(
            extraction_cfg.get("pred_include_summary_factors", False)
        ),
        pred_include_citation_snippets=bool(
            extraction_cfg.get("pred_include_citation_snippets", False)
        ),
        canonical_factors=(
            extraction_cfg.get("canonical_factors")
            if isinstance(extraction_cfg.get("canonical_factors"), dict)
            else None
        ),
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
    s3.set_defaults(func=lambda a: _run_async(_cmd_phase1(a)))

    s6 = sub.add_parser("phase3")
    s6.set_defaults(func=lambda a: _run_async(_cmd_phase3(a)))

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

    s10 = sub.add_parser("phase6")
    s10.add_argument("--input-jsonl", required=True)
    s10.add_argument("--tasks-clean-json", default="")
    s10.set_defaults(func=_cmd_phase6)

    s9 = sub.add_parser("paper")
    s9.set_defaults(func=lambda a: _run_async(_cmd_paper(a)))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
