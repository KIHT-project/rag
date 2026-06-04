from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "thesis_generate_results.py"
    spec = importlib.util.spec_from_file_location("thesis_generate_results", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_ragas_table_renders_na_and_is_deterministic(tmp_path):
    module = _load_module()
    run_dir = (
        Path(__file__).resolve().parents[2]
        / "evaluation_metrics"
        / "runs"
        / "old"
        / "feb-10-valid-1-full-run"
        / "20260209_114745_nogit"
    )
    out = module.OutputPaths(images_dir=tmp_path / "images", tables_dir=tmp_path / "tables")
    module._ensure_dirs(out)

    module.generate_ragas_tables_and_plots(run_dir, out)
    first_output = (out.tables_dir / "ragas_metrics_summary.tex").read_text(encoding="utf-8")

    module.generate_ragas_tables_and_plots(run_dir, out)
    second_output = (out.tables_dir / "ragas_metrics_summary.tex").read_text(encoding="utf-8")

    assert first_output == second_output
    assert "NaN" not in first_output
    assert "LLM only & 0.610415 & N/A & N/A" in first_output
    assert "RAG no HyDE & 0.867172 & 0.816667 & 1.000000" in first_output
    assert "RAG HyDE & 0.820208 & 0.645833 & 0.900000" in first_output
    assert "Answer relevancy" in first_output
    assert "Faithfulness" in first_output
    assert "Context precision" in first_output
    assert module.RAGAS_TABLE_NOTE in first_output
    assert "\\begin{tabular}" in first_output
    assert "\\end{tabular}" in first_output
