#!/usr/bin/env python3
"""
Generate thesis figures and LaTeX tables from evaluation run artifacts.

Inputs
  Run directory path, example:
    evaluation_metrics/runs/old/feb-10-valid-1-full-run/20260209_114745_nogit

Outputs
  docs/thesis/latex/src/resources/images/*.pdf
  docs/thesis/latex/src/resources/tables/*.tex

Notes
  This script is read only with respect to evaluation artifacts.
  It overwrites generated outputs deterministically.
"""

# python scripts/thesis_generate_results.py --run-dir evaluation_metrics/runs/old/feb-10-valid-1-full-run/20260209_114745_nogit
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class OutputPaths:
    images_dir: Path
    tables_dir: Path


CONFIG_LABELS = {
    "llm_only": "LLM only",
    "rag_no_hyde": "RAG no HyDE",
    "rag_hyde": "RAG HyDE",
}

CONFIG_ORDER = ["llm_only", "rag_no_hyde", "rag_hyde"]

METRIC_LABELS = {
    "answer_relevancy": "Answer relevancy",
    "faithfulness": "Faithfulness",
    "llm_context_precision_without_reference": "Context precision",
    "llm_context_precision_wo_ref": "Context precision",
}

RAGAS_TABLE_NOTE = (
    "Note: Retrieval-dependent metrics are not applicable to the LLM-only baseline "
    "because no contexts are retrieved."
)


def _ensure_dirs(out: OutputPaths) -> None:
    out.images_dir.mkdir(parents=True, exist_ok=True)
    out.tables_dir.mkdir(parents=True, exist_ok=True)


def _find_run_csvs(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    csvs: list[Path] = []
    for p in run_dir.rglob("*.csv"):
        csvs.append(p)

    return sorted(csvs)


def _select_primary_deterministic_csvs(csvs: Iterable[Path]) -> list[Path]:
    primary = [path for path in csvs if "primary_deterministic" in path.parts]
    if primary:
        return sorted(primary)
    return sorted(csvs)


def _infer_config_from_filename(path: Path) -> str | None:
    name = path.name.lower()
    if "llm_only" in name:
        return "llm_only"
    if "rag_no_hyde" in name:
        return "rag_no_hyde"
    if "rag_hyde" in name:
        return "rag_hyde"
    return None


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {path}. Error: {e}") from e


def _numeric_mean_row(df: pd.DataFrame) -> pd.Series:
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return pd.Series(dtype="float64")
    return numeric.mean(numeric_only=True)


def _write_latex_table(
    df: pd.DataFrame,
    out_path: Path,
    caption: str | None = None,
    label: str | None = None,
    note: str | None = None,
) -> None:
    """
    Writes a plain tabular snippet (not a full table float) so chapters can \\input{} it.
    """
    tex = df.to_latex(index=False, escape=True, na_rep="N/A")

    lines: list[str] = []
    if caption or label:
        lines.append("% This is a snippet intended to be included inside a table environment if needed.")
        if caption:
            lines.append(f"% caption: {caption}")
        if label:
            lines.append(f"% label: {label}")
    lines.append(tex)
    if note:
        lines.append(r"\vspace{2pt}")
        lines.append(rf"\parbox{{0.95\linewidth}}{{\footnotesize {note}}}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _order_configs(summary: pd.DataFrame) -> pd.DataFrame:
    if "config" not in summary.columns:
        return summary

    ordered = summary.set_index("config").reindex(CONFIG_ORDER)
    ordered = ordered.dropna(how="all").reset_index()
    return ordered


def _format_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.rename(columns=METRIC_LABELS).copy()
    return formatted


def _save_bar_plot(df: pd.DataFrame, metric_columns: list[str], title: str, out_pdf: Path) -> None:
    """
    df rows must be configs, index or column named config.
    If duplicates exist per config, they are aggregated by mean.
    """
    plot_df = df.copy()

    if "config" in plot_df.columns:
        plot_df = plot_df.set_index("config")

    # If multiple rows exist for the same config, aggregate before reindexing
    if plot_df.index.has_duplicates:
        plot_df = plot_df.groupby(level=0, sort=False).mean(numeric_only=True)

    plot_df = plot_df.reindex(CONFIG_ORDER)

    cols = [c for c in metric_columns if c in plot_df.columns]
    if not cols:
        return

    ax = plot_df[cols].plot(kind="bar", rot=0)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON: {path}. Error: {e}") from e


def _flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    """
    Flattens nested JSON into key paths.
    Example: {"a": {"b": 1}} becomes {"a.b": 1}
    """
    out: dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_json(v, key))
        return out

    if isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            out.update(_flatten_json(v, key))
        return out

    out[prefix] = obj
    return out


def _extract_candidate_metric_paths(flat: dict[str, Any], patterns: Iterable[str]) -> dict[str, float]:
    """
    Returns numeric values whose key matches any regex in patterns.
    If multiple keys map to same metric name, it keeps the first encountered in sorted key order.
    """
    metrics: dict[str, float] = {}
    keys = sorted(flat.keys())

    compiled: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in patterns]

    for key in keys:
        val = flat.get(key)
        if not isinstance(val, (int, float)):
            continue
        for rx in compiled:
            m = rx.search(key)
            if not m:
                continue
            metric_name = m.group(1) if m.groups() else key
            metric_name = metric_name.strip()
            if metric_name not in metrics:
                metrics[metric_name] = float(val)
            break

    return metrics


def build_outputs_base() -> OutputPaths:
    base = Path("docs") / "thesis" / "latex" / "src" / "resources"
    images_dir = base / "images"
    tables_dir = base / "tables"
    return OutputPaths(images_dir=images_dir, tables_dir=tables_dir)


def generate_ragas_tables_and_plots(run_dir: Path, out: OutputPaths) -> None:
    """
    Aggregates RAGAS CSVs by configuration, outputs:
      resources/tables/ragas_metrics_summary.tex
      resources/images/ragas_metrics_summary.pdf
    """
    csvs = _find_run_csvs(run_dir)

    ragas_csvs = [p for p in csvs if "phase4_ragas" in p.name.lower()]
    ragas_csvs = _select_primary_deterministic_csvs(ragas_csvs)
    if not ragas_csvs:
        return

    rows: list[dict[str, Any]] = []
    for p in ragas_csvs:
        cfg = _infer_config_from_filename(p)
        if not cfg:
            continue
        df = _safe_read_csv(p)
        mean_row = _numeric_mean_row(df)

        row: dict[str, Any] = {"config": cfg}
        for k, v in mean_row.to_dict().items():
            row[str(k)] = float(v)
        rows.append(row)

    if not rows:
        return

    raw = pd.DataFrame(rows)

    # Ensure one row per config
    numeric_cols = [c for c in raw.columns if c != "config"]
    summary = raw.groupby("config", as_index=False, sort=False)[numeric_cols].mean(numeric_only=True)

    summary = _order_configs(summary)
    summary["config_label"] = summary["config"].map(CONFIG_LABELS).fillna(summary["config"])

    table_cols = ["config_label"] + [c for c in summary.columns if c not in {"config", "config_label"}]
    summary_view = summary[table_cols].copy()

    numeric_cols_view = [c for c in summary_view.columns if c != "config_label"]
    summary_view[numeric_cols_view] = summary_view[numeric_cols_view].round(6)
    summary_view = _format_metric_columns(summary_view)

    _write_latex_table(
        summary_view.rename(columns={"config_label": "Configuration"}),
        out.tables_dir / "ragas_metrics_summary.tex",
        caption="Mean RAGAS metrics by configuration for the selected run.",
        label="tab:ragas-metrics-summary",
        note=RAGAS_TABLE_NOTE,
    )

    plot_df = summary[["config"] + [c for c in summary.columns if c not in {"config", "config_label"}]].copy()
    metric_cols = [c for c in plot_df.columns if c != "config"]

    _save_bar_plot(
        plot_df,
        metric_columns=metric_cols[:10],
        title="RAGAS metrics, mean by configuration",
        out_pdf=out.images_dir / "ragas_metrics_summary.pdf",
    )


def generate_extraction_tables_and_plots(run_dir: Path, out: OutputPaths) -> None:
    """
    Aggregates extraction CSVs by configuration, outputs:
      resources/tables/extraction_metrics_summary.tex
      resources/images/extraction_metrics_summary.pdf
    """
    csvs = _find_run_csvs(run_dir)

    extraction_csvs = [p for p in csvs if "phase6_extraction" in p.name.lower() and p.suffix.lower() == ".csv"]
    extraction_csvs = [p for p in extraction_csvs if "per_query" in p.name.lower()]
    if not extraction_csvs:
        return

    rows: list[dict[str, Any]] = []
    for p in extraction_csvs:
        cfg = _infer_config_from_filename(p)
        if not cfg:
            continue
        df = _safe_read_csv(p)
        mean_row = _numeric_mean_row(df)

        row: dict[str, Any] = {"config": cfg}
        for k, v in mean_row.to_dict().items():
            row[str(k)] = float(v)
        rows.append(row)

    if not rows:
        return

    raw = pd.DataFrame(rows)

    # Ensure one row per config
    numeric_cols = [c for c in raw.columns if c != "config"]
    summary = raw.groupby("config", as_index=False, sort=False)[numeric_cols].mean(numeric_only=True)

    summary["config_label"] = summary["config"].map(CONFIG_LABELS).fillna(summary["config"])

    table_cols = ["config_label"] + [c for c in summary.columns if c not in {"config", "config_label"}]
    summary_view = summary[table_cols].copy()

    numeric_cols_view = [c for c in summary_view.columns if c != "config_label"]
    summary_view[numeric_cols_view] = summary_view[numeric_cols_view].round(6)

    _write_latex_table(
        summary_view.rename(columns={"config_label": "Configuration"}),
        out.tables_dir / "extraction_metrics_summary.tex",
        caption="Mean extraction metrics by configuration for the selected run.",
        label="tab:extraction-metrics-summary",
    )

    plot_df = summary[["config"] + [c for c in summary.columns if c not in {"config", "config_label"}]].copy()
    metric_cols = [c for c in plot_df.columns if c != "config"]

    _save_bar_plot(
        plot_df,
        metric_columns=metric_cols[:10],
        title="Extraction metrics, mean by configuration",
        out_pdf=out.images_dir / "extraction_metrics_summary.pdf",
    )

def generate_benchmark_summary_table(run_dir: Path, out: OutputPaths) -> None:
    """
    Tries to extract a compact table from paper_benchmark_summary.json if present.
    The JSON schema is unknown, so this is conservative:
      1. Flatten JSON
      2. Extract numeric key paths matching common retrieval terms
      3. Emit a table of discovered metrics
    """
    json_path = run_dir / "paper_benchmark_summary.json"
    obj = _load_json(json_path)
    if obj is None:
        return

    flat = _flatten_json(obj)

    patterns = [
        r"(ndcg(?:@?\d+)?)",
        r"(precision(?:@?\d+)?)",
        r"(recall(?:@?\d+)?)",
        r"(map(?:@?\d+)?)",
        r"(mrr(?:@?\d+)?)",
        r"(f1)",
        r"(accuracy)",
    ]

    metrics = _extract_candidate_metric_paths(flat, patterns)
    if not metrics:
        return

    rows = [{"Metric": k, "Value": v} for k, v in metrics.items()]
    df = pd.DataFrame(rows).sort_values(by="Metric")
    df["Value"] = df["Value"].round(6)

    _write_latex_table(
        df,
        out.tables_dir / "paper_benchmark_summary_metrics.tex",
        caption="Numeric metrics extracted from paper_benchmark_summary.json for the selected run.",
        label="tab:paper-benchmark-summary",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate thesis figures and LaTeX tables from evaluation artifacts.")
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to a specific evaluation run directory.",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir

    out = build_outputs_base()
    _ensure_dirs(out)

    generate_benchmark_summary_table(run_dir, out)
    generate_ragas_tables_and_plots(run_dir, out)
    generate_extraction_tables_and_plots(run_dir, out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
