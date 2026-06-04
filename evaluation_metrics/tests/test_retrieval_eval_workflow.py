from __future__ import annotations

import asyncio
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation_metrics.src.eval_cli import build_parser
from evaluation_metrics.src.qrels_annotation import TEMPLATE_FIELDS
from evaluation_metrics.src.schemas.models import AskResponse, RunContext
from evaluation_metrics.src.workflows import retrieval_eval_workflow as workflow


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_queries(path: Path) -> None:
    _write_jsonl(path, [{"_id": "q1", "text": "Question 1"}])


def _write_annotations(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEMPLATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class _FakeRagClient:
    async def ask(
        self,
        *,
        question: str,
        filters: dict[str, object] | None = None,
        hyde_enabled: bool = False,
        debug_enabled: bool = False,
        debug_ask_max_chunks_candidate: int | None = None,
        hyde_header_name: str = "X-HyDE-Enabled",
        hyde_header_value: str = "true",
    ) -> AskResponse:
        if hyde_enabled:
            raw = {
                "debug": {
                    "hyde_candidates": [
                        {
                            "doc_id": "doc-hyde",
                            "chunk_id": "chunk-hyde",
                            "title": "HyDE candidate",
                            "score": 0.9,
                            "origin": "hyde",
                        }
                    ],
                    "selected_candidates": [
                        {
                            "doc_id": "doc-hyde",
                            "chunk_id": "chunk-hyde",
                            "title": "HyDE candidate",
                            "score": 0.9,
                            "origin": "hyde",
                            "selected_for_context": True,
                        }
                    ],
                }
            }
        else:
            raw = {
                "debug": {
                    "question_candidates": [
                        {
                            "doc_id": "doc-direct",
                            "chunk_id": "chunk-direct",
                            "title": "Direct candidate",
                            "score": 0.8,
                            "origin": "question",
                        }
                    ],
                    "selected_candidates": [
                        {
                            "doc_id": "doc-direct",
                            "chunk_id": "chunk-direct",
                            "title": "Direct candidate",
                            "score": 0.8,
                            "origin": "question",
                            "selected_for_context": True,
                        }
                    ],
                }
            }
        return AskResponse(answer=None, citations=[], raw=raw)


def _build_finalize_fixture(run_dir: Path) -> None:
    _write_jsonl(
        run_dir / workflow.RETRIEVAL_DIRECT,
        [
            {"query_id": "q1", "question": "Question 1", "doc_id": "doc-1", "rank": 1},
            {"query_id": "q1", "question": "Question 1", "doc_id": "doc-2", "rank": 2},
        ],
    )
    _write_jsonl(
        run_dir / workflow.RETRIEVAL_HYDE,
        [
            {"query_id": "q1", "question": "Question 1", "doc_id": "doc-2", "rank": 1},
            {"query_id": "q1", "question": "Question 1", "doc_id": "doc-1", "rank": 2},
        ],
    )
    _write_jsonl(
        run_dir / workflow.POOLED_CANDIDATES,
        [
            {
                "query_id": "q1",
                "question": "Question 1",
                "doc_id": "doc-1",
                "title": "A",
            },
            {
                "query_id": "q1",
                "question": "Question 1",
                "doc_id": "doc-2",
                "title": "B",
            },
        ],
    )
    _write_annotations(
        run_dir / workflow.QRELS_ANNOTATION_COMPLETED,
        [
            {
                "query_id": "q1",
                "question": "Question 1",
                "resolved_doc_id": "doc-1",
                "doc_id": "doc-1",
                "chunk_id": "",
                "doi": "",
                "pmid": "",
                "title": "A",
                "relevance": "2",
                "rationale": "Directly relevant.",
                "annotator": "ann",
                "adjudication_status": "",
            },
            {
                "query_id": "q1",
                "question": "Question 1",
                "resolved_doc_id": "doc-2",
                "doc_id": "doc-2",
                "chunk_id": "",
                "doi": "",
                "pmid": "",
                "title": "B",
                "relevance": "0",
                "rationale": "",
                "annotator": "ann",
                "adjudication_status": "",
            },
        ],
    )


def test_prepare_creates_pooled_candidates_and_annotation_template(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    queries = tmp_path / "queries.jsonl"
    _write_queries(queries)

    summary = asyncio.run(
        workflow.prepare_retrieval_evaluation(
            ctx=RunContext(run_id="run-1", run_dir=str(run_dir)),
            rag=_FakeRagClient(),
            queries_jsonl=queries,
            hyde_header_name="X-HyDE-Enabled",
            hyde_header_value="true",
            retrieval_pool_depth=10,
            max_queries=1,
        )
    )

    assert summary.run_dir == run_dir
    assert summary.retrieval_direct_row_count == 1
    assert summary.retrieval_hyde_row_count == 1
    assert summary.retrieval_final_context_row_count == 2
    assert summary.pooled_candidate_count == 2
    assert (
        summary.annotation_template_path == run_dir / workflow.QRELS_ANNOTATION_TEMPLATE
    )
    assert (run_dir / workflow.POOLED_CANDIDATES).exists()
    assert (run_dir / workflow.QRELS_ANNOTATION_TEMPLATE).exists()
    assert not (run_dir / workflow.QRELS).exists()

    text = workflow.format_prepare_summary(summary)
    assert f"run_dir: {run_dir}" in text
    assert (
        "Copy qrels_annotation_template.csv to qrels_annotation_completed.csv." in text
    )


def test_prepare_fails_when_retrieval_artifacts_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _missing_export(**kwargs: object) -> None:
        return None

    monkeypatch.setattr(workflow, "run_retrieval_export", _missing_export)

    with pytest.raises(RuntimeError, match="retrieval_direct.jsonl is missing"):
        asyncio.run(
            workflow.prepare_retrieval_evaluation(
                ctx=RunContext(run_id="run-1", run_dir=str(tmp_path / "run")),
                rag=_FakeRagClient(),
                queries_jsonl=tmp_path / "queries.jsonl",
                hyde_header_name="X-HyDE-Enabled",
                hyde_header_value="true",
                retrieval_pool_depth=10,
            )
        )


def test_prepare_fails_when_retrieval_artifacts_are_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _empty_export(**kwargs: object) -> None:
        run_dir = Path(kwargs["ctx"].run_dir)  # type: ignore[index, union-attr]
        run_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            workflow.RETRIEVAL_DIRECT,
            workflow.RETRIEVAL_HYDE,
            workflow.RETRIEVAL_FINAL_CONTEXT,
        ):
            (run_dir / filename).write_text("", encoding="utf-8")

    monkeypatch.setattr(workflow, "run_retrieval_export", _empty_export)

    with pytest.raises(RuntimeError, match="retrieval_direct.jsonl is empty"):
        asyncio.run(
            workflow.prepare_retrieval_evaluation(
                ctx=RunContext(run_id="run-1", run_dir=str(tmp_path / "run")),
                rag=_FakeRagClient(),
                queries_jsonl=tmp_path / "queries.jsonl",
                hyde_header_name="X-HyDE-Enabled",
                hyde_header_value="true",
                retrieval_pool_depth=10,
            )
        )


def test_finalize_fails_when_completed_annotations_are_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(RuntimeError, match="manual annotation"):
        workflow.finalize_retrieval_evaluation(run_dir=run_dir, k_values=[1])


def test_finalize_validates_before_qrels_and_generates_qrels_before_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _build_finalize_fixture(run_dir)
    events: list[str] = []

    def _validate(**kwargs: object) -> list[dict[str, str]]:
        events.append("validate")
        return [{"query_id": "q1", "resolved_doc_id": "doc-1"}]

    def _generate(**kwargs: object) -> list[dict[str, str]]:
        assert events == ["validate"]
        events.append("generate_qrels")
        Path(kwargs["qrels_output_path"]).write_text(
            "query_id\tdoc_id\trelevance\nq1\tdoc-1\t2\n",
            encoding="utf-8",
        )
        return [{"query_id": "q1", "resolved_doc_id": "doc-1"}]

    def _compute(**kwargs: object) -> tuple[dict[str, object], list[dict[str, object]]]:
        assert events == ["validate", "generate_qrels"]
        events.append("metrics")
        return {"query_count": 1}, []

    def _write(**kwargs: object) -> None:
        events.append("write_outputs")

    monkeypatch.setattr(workflow, "validate_annotations", _validate)
    monkeypatch.setattr(workflow, "generate_qrels", _generate)
    monkeypatch.setattr(workflow, "compute_retrieval_metrics", _compute)
    monkeypatch.setattr(workflow, "write_outputs", _write)

    summary = workflow.finalize_retrieval_evaluation(run_dir=run_dir, k_values=[1])

    assert events == ["validate", "generate_qrels", "metrics", "write_outputs"]
    assert summary.query_count == 1
    assert summary.judged_candidate_count == 1


def test_finalize_fails_fast_on_invalid_k(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _build_finalize_fixture(run_dir)

    with pytest.raises(ValueError, match="positive integers"):
        workflow.finalize_retrieval_evaluation(run_dir=run_dir, k_values=[0])

    assert not (run_dir / workflow.QRELS).exists()


def test_finalize_fails_before_qrels_when_k_exceeds_retrieval_depth(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _build_finalize_fixture(run_dir)

    with pytest.raises(ValueError, match="exceeds available retrieval depth"):
        workflow.finalize_retrieval_evaluation(run_dir=run_dir, k_values=[3])

    assert not (run_dir / workflow.QRELS).exists()


def test_finalize_writes_outputs_under_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _build_finalize_fixture(run_dir)

    summary = workflow.finalize_retrieval_evaluation(run_dir=run_dir, k_values=[1, 2])

    assert summary.run_dir == run_dir
    assert summary.qrels_path == run_dir / workflow.QRELS
    assert summary.metrics_summary_path == run_dir / workflow.METRICS_SUMMARY
    assert summary.per_query_metrics_path == run_dir / workflow.METRICS_PER_QUERY
    assert summary.latex_table_path == run_dir / workflow.METRICS_COMPARISON_TEX
    assert summary.metric_depths == [1, 2]
    assert summary.query_count == 1
    assert summary.judged_candidate_count == 2
    assert _read_jsonl(run_dir / workflow.POOLED_CANDIDATES)
    assert summary.qrels_path.exists()
    assert summary.metrics_summary_path.exists()
    assert summary.per_query_metrics_path.exists()
    assert summary.latex_table_path.exists()


def test_existing_individual_command_parsers_still_work() -> None:
    parser = build_parser()

    cases = [
        ["retrieval-export", "--retrieval-pool-depth", "10", "--max-queries", "5"],
        [
            "qrels-create-template",
            "--direct",
            "direct.jsonl",
            "--hyde",
            "hyde.jsonl",
            "--pooled-output",
            "pooled.jsonl",
            "--template-output",
            "template.csv",
        ],
        [
            "qrels-validate",
            "--pooled",
            "pooled.jsonl",
            "--annotations",
            "annotations.csv",
        ],
        [
            "qrels-generate",
            "--pooled",
            "pooled.jsonl",
            "--annotations",
            "annotations.csv",
            "--qrels-output",
            "qrels.tsv",
        ],
        [
            "retrieval-metrics",
            "--direct",
            "direct.jsonl",
            "--hyde",
            "hyde.jsonl",
            "--qrels",
            "qrels.tsv",
            "--summary-output",
            "summary.json",
            "--per-query-output",
            "per_query.csv",
            "--latex-output",
            "table.tex",
            "--k",
            "1",
        ],
        [
            "retrieval-eval-prepare",
            "--retrieval-pool-depth",
            "10",
            "--max-queries",
            "5",
        ],
        ["retrieval-eval-finalize", "--run-dir", "run", "--k", "1", "--k", "5"],
    ]

    for argv in cases:
        args = parser.parse_args(argv)
        assert callable(args.func)


def test_paper_command_parser_still_works() -> None:
    args = build_parser().parse_args(["paper"])

    assert args.cmd == "paper"
    assert callable(args.func)
