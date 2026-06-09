from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation_metrics.src.retrieval_metrics import compute_retrieval_metrics, write_outputs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _write_qrels(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query_id", "doc_id", "relevance"])
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_valid_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    direct = tmp_path / "retrieval_direct.jsonl"
    hyde = tmp_path / "retrieval_hyde.jsonl"
    qrels = tmp_path / "qrels.tsv"

    _write_jsonl(
        direct,
        [
            {"query_id": "q1", "doc_id": "d1", "rank": 1},
            {"query_id": "q1", "doc_id": "d2", "rank": 2},
            {"query_id": "q1", "doc_id": "d3", "rank": 3},
            {"query_id": "q1", "doc_id": "d1", "rank": 4},
            {"query_id": "q1", "doc_id": "d4", "rank": 5},
            {"query_id": "q1", "doc_id": "d5", "rank": 6},
            {"query_id": "q1", "doc_id": "d6", "rank": 7},
            {"query_id": "q1", "doc_id": "d7", "rank": 8},
            {"query_id": "q1", "doc_id": "d8", "rank": 9},
            {"query_id": "q1", "doc_id": "d9", "rank": 10},
            {"query_id": "q1", "doc_id": "d10", "rank": 11},
            {"query_id": "q2", "doc_id": "z1", "rank": 1},
            {"query_id": "q2", "doc_id": "z2", "rank": 2},
            {"query_id": "q2", "doc_id": "z3", "rank": 3},
            {"query_id": "q2", "doc_id": "z4", "rank": 4},
            {"query_id": "q2", "doc_id": "z5", "rank": 5},
            {"query_id": "q2", "doc_id": "z6", "rank": 6},
            {"query_id": "q2", "doc_id": "z7", "rank": 7},
            {"query_id": "q2", "doc_id": "z8", "rank": 8},
            {"query_id": "q2", "doc_id": "z9", "rank": 9},
            {"query_id": "q2", "doc_id": "z10", "rank": 10},
        ],
    )
    _write_jsonl(
        hyde,
        [
            {"query_id": "q1", "doc_id": "d3", "rank": 1},
            {"query_id": "q1", "doc_id": "d1", "rank": 2},
            {"query_id": "q1", "doc_id": "d2", "rank": 3},
            {"query_id": "q1", "doc_id": "d4", "rank": 4},
            {"query_id": "q1", "doc_id": "d5", "rank": 5},
            {"query_id": "q1", "doc_id": "d6", "rank": 6},
            {"query_id": "q1", "doc_id": "d7", "rank": 7},
            {"query_id": "q1", "doc_id": "d8", "rank": 8},
            {"query_id": "q1", "doc_id": "d9", "rank": 9},
            {"query_id": "q1", "doc_id": "d10", "rank": 10},
            {"query_id": "q2", "doc_id": "z1", "rank": 1},
            {"query_id": "q2", "doc_id": "z2", "rank": 2},
            {"query_id": "q2", "doc_id": "z3", "rank": 3},
            {"query_id": "q2", "doc_id": "z4", "rank": 4},
            {"query_id": "q2", "doc_id": "z5", "rank": 5},
            {"query_id": "q2", "doc_id": "z6", "rank": 6},
            {"query_id": "q2", "doc_id": "z7", "rank": 7},
            {"query_id": "q2", "doc_id": "z8", "rank": 8},
            {"query_id": "q2", "doc_id": "z9", "rank": 9},
            {"query_id": "q2", "doc_id": "z10", "rank": 10},
        ],
    )
    _write_qrels(
        qrels,
        [
            ("q1", "d1", 2),
            ("q1", "d2", 1),
            ("q1", "d3", 0),
            ("q1", "d4", 0),
            ("q1", "d5", 0),
            ("q1", "d6", 0),
            ("q1", "d7", 0),
            ("q1", "d8", 0),
            ("q1", "d9", 0),
            ("q1", "d10", 0),
            ("q2", "z1", 0),
            ("q2", "z2", 0),
            ("q2", "z3", 0),
            ("q2", "z4", 0),
            ("q2", "z5", 0),
            ("q2", "z6", 0),
            ("q2", "z7", 0),
            ("q2", "z8", 0),
            ("q2", "z9", 0),
            ("q2", "z10", 0),
        ],
    )
    return direct, hyde, qrels


def test_metric_math_and_exclusions(tmp_path: Path) -> None:
    direct, hyde, qrels = _build_valid_fixture(tmp_path)

    summary, per_query_rows = compute_retrieval_metrics(
        direct_path=direct,
        hyde_path=hyde,
        qrels_path=qrels,
        k_values=[5, 10],
    )

    assert summary["metric_depths"] == [5, 10]
    assert summary["query_count"] == 2
    assert summary["judged_candidate_count"] == 20
    assert summary["relevance_distribution"] == {"0": 18, "1": 1, "2": 1}

    q1_direct = next(row for row in per_query_rows if row["query_id"] == "q1" and row["retrieval_mode"] == "direct")
    q1_hyde = next(row for row in per_query_rows if row["query_id"] == "q1" and row["retrieval_mode"] == "hyde")
    q2_direct = next(row for row in per_query_rows if row["query_id"] == "q2" and row["retrieval_mode"] == "direct")

    assert q1_direct["retrieved_count"] == 10
    assert q1_direct["judged_relevant_count"] == 2
    assert q1_direct["mrr"] == pytest.approx(1.0)
    assert q1_direct["precision_at_5"] == pytest.approx(0.4)
    assert q1_direct["precision_at_10"] == pytest.approx(0.2)
    assert q1_direct["recall_at_5"] == pytest.approx(1.0)
    assert q1_direct["recall_at_10"] == pytest.approx(1.0)
    assert q1_direct["ndcg_at_5"] == pytest.approx(1.0)
    assert q1_direct["ndcg_at_10"] == pytest.approx(1.0)

    assert q1_hyde["mrr"] == pytest.approx(0.5)
    assert q1_hyde["precision_at_5"] == pytest.approx(0.4)
    assert q1_hyde["precision_at_10"] == pytest.approx(0.2)
    assert q1_hyde["recall_at_5"] == pytest.approx(1.0)
    assert q1_hyde["ndcg_at_5"] == pytest.approx(0.6590018048)

    assert q2_direct["mrr"] == pytest.approx(0.0)
    assert q2_direct["recall_at_5"] is None
    assert q2_direct["ndcg_at_5"] is None

    direct_metrics = summary["aggregate_metrics_by_mode"]["direct"]
    hyde_metrics = summary["aggregate_metrics_by_mode"]["hyde"]

    assert direct_metrics["mrr"] == pytest.approx(0.5)
    assert direct_metrics["precision_at_5"] == pytest.approx(0.2)
    assert direct_metrics["precision_at_10"] == pytest.approx(0.1)
    assert direct_metrics["recall_at_5"] == pytest.approx(1.0)
    assert direct_metrics["recall_at_10"] == pytest.approx(1.0)
    assert direct_metrics["ndcg_at_5"] == pytest.approx(1.0)
    assert direct_metrics["ndcg_at_10"] == pytest.approx(1.0)

    assert hyde_metrics["mrr"] == pytest.approx(0.25)
    assert hyde_metrics["precision_at_5"] == pytest.approx(0.2)
    assert hyde_metrics["precision_at_10"] == pytest.approx(0.1)
    assert hyde_metrics["recall_at_5"] == pytest.approx(1.0)
    assert hyde_metrics["ndcg_at_5"] == pytest.approx(0.6590018048)

    assert summary["excluded_query_counts_by_metric"]["direct"]["recall_at_5"] == 1
    assert summary["excluded_query_counts_by_metric"]["direct"]["ndcg_at_10"] == 1


def test_duplicate_qrels_rows_fail(tmp_path: Path) -> None:
    direct, hyde, qrels = _build_valid_fixture(tmp_path)
    _write_qrels(qrels, [("q1", "d1", 2), ("q1", "d1", 1)])

    with pytest.raises(ValueError, match="Duplicate qrels row"):
        compute_retrieval_metrics(direct_path=direct, hyde_path=hyde, qrels_path=qrels)


def test_invalid_relevance_values_fail(tmp_path: Path) -> None:
    direct, hyde, qrels = _build_valid_fixture(tmp_path)
    _write_qrels(qrels, [("q1", "d1", 3)])

    with pytest.raises(ValueError, match="invalid relevance"):
        compute_retrieval_metrics(direct_path=direct, hyde_path=hyde, qrels_path=qrels)


def test_retrieval_row_without_stable_identifier_fails(tmp_path: Path) -> None:
    direct = tmp_path / "retrieval_direct.jsonl"
    hyde = tmp_path / "retrieval_hyde.jsonl"
    qrels = tmp_path / "qrels.tsv"

    _write_jsonl(direct, [{"query_id": "q1", "rank": 1}])
    _write_jsonl(hyde, [{"query_id": "q1", "doc_id": "d1", "rank": 1}])
    _write_qrels(qrels, [("q1", "d1", 2)])

    with pytest.raises(ValueError, match="no stable identifier"):
        compute_retrieval_metrics(direct_path=direct, hyde_path=hyde, qrels_path=qrels)


def test_metrics_fail_when_k_exceeds_available_depth(tmp_path: Path) -> None:
    direct = tmp_path / "retrieval_direct.jsonl"
    hyde = tmp_path / "retrieval_hyde.jsonl"
    qrels = tmp_path / "qrels.tsv"

    _write_jsonl(
        direct,
        [
            {"query_id": "q1", "doc_id": "d1", "rank": 1},
            {"query_id": "q1", "doc_id": "d2", "rank": 2},
        ],
    )
    _write_jsonl(
        hyde,
        [
            {"query_id": "q1", "doc_id": "d1", "rank": 1},
            {"query_id": "q1", "doc_id": "d2", "rank": 2},
        ],
    )
    _write_qrels(qrels, [("q1", "d1", 2), ("q1", "d2", 0)])

    with pytest.raises(ValueError, match="Requested metric depth exceeds available retrieval depth"):
        compute_retrieval_metrics(
            direct_path=direct,
            hyde_path=hyde,
            qrels_path=qrels,
            k_values=[5],
        )


def test_queries_in_retrieval_missing_from_qrels_fail(tmp_path: Path) -> None:
    direct = tmp_path / "retrieval_direct.jsonl"
    hyde = tmp_path / "retrieval_hyde.jsonl"
    qrels = tmp_path / "qrels.tsv"

    _write_jsonl(direct, [{"query_id": "q1", "doc_id": "d1", "rank": 1}])
    _write_jsonl(hyde, [{"query_id": "q2", "doc_id": "d1", "rank": 1}])
    _write_qrels(qrels, [("q1", "d1", 2)])

    with pytest.raises(ValueError, match="missing from qrels.tsv"):
        compute_retrieval_metrics(direct_path=direct, hyde_path=hyde, qrels_path=qrels)


def test_missing_qrels_queries_are_reported_when_not_strict(tmp_path: Path) -> None:
    direct = tmp_path / "retrieval_direct.jsonl"
    hyde = tmp_path / "retrieval_hyde.jsonl"
    qrels = tmp_path / "qrels.tsv"

    _write_jsonl(direct, [{"query_id": "q1", "doc_id": "d1", "rank": 1}])
    _write_jsonl(hyde, [{"query_id": "q1", "doc_id": "d1", "rank": 1}])
    _write_qrels(qrels, [("q1", "d1", 2), ("q2", "d2", 1)])

    summary, _ = compute_retrieval_metrics(
        direct_path=direct,
        hyde_path=hyde,
        qrels_path=qrels,
        k_values=[1],
    )

    assert summary["qrels_queries_missing_in_retrieval_by_mode"] == {
        "direct": ["q2"],
        "hyde": ["q2"],
    }


def test_missing_qrels_queries_fail_in_strict_mode(tmp_path: Path) -> None:
    direct = tmp_path / "retrieval_direct.jsonl"
    hyde = tmp_path / "retrieval_hyde.jsonl"
    qrels = tmp_path / "qrels.tsv"

    _write_jsonl(direct, [{"query_id": "q1", "doc_id": "d1", "rank": 1}])
    _write_jsonl(hyde, [{"query_id": "q1", "doc_id": "d1", "rank": 1}])
    _write_qrels(qrels, [("q1", "d1", 2), ("q2", "d2", 1)])

    with pytest.raises(ValueError, match="strict mode"):
        compute_retrieval_metrics(
            direct_path=direct,
            hyde_path=hyde,
            qrels_path=qrels,
            k_values=[1],
            strict=True,
        )


def test_output_files_are_deterministic_and_traceable(tmp_path: Path) -> None:
    direct, hyde, qrels = _build_valid_fixture(tmp_path)
    summary, per_query_rows = compute_retrieval_metrics(
        direct_path=direct,
        hyde_path=hyde,
        qrels_path=qrels,
        k_values=[5, 10],
    )

    summary_output = tmp_path / "retrieval_metrics_summary.json"
    per_query_output = tmp_path / "retrieval_metrics_per_query.csv"
    latex_output = tmp_path / "retrieval_metrics_comparison.tex"
    write_outputs(
        summary=summary,
        per_query_rows=per_query_rows,
        summary_output=summary_output,
        per_query_output=per_query_output,
        latex_output=latex_output,
    )

    written_summary = json.loads(summary_output.read_text(encoding="utf-8"))
    written_rows = _read_csv(per_query_output)
    latex = latex_output.read_text(encoding="utf-8")

    assert written_summary["aggregate_metrics_by_mode"]["direct"]["mrr"] == pytest.approx(0.5)
    assert [row["query_id"] for row in written_rows] == ["q1", "q1", "q2", "q2"]
    assert [row["retrieval_mode"] for row in written_rows] == ["direct", "hyde", "direct", "hyde"]
    assert "Metric & Direct & HyDE" in latex
    assert "MRR & 0.5000 & 0.2500" in latex
    assert "Precision@5 & 0.2000 & 0.2000" in latex


def test_at_20_not_calculated_by_default(tmp_path: Path) -> None:
    direct, hyde, qrels = _build_valid_fixture(tmp_path)
    summary, per_query_rows = compute_retrieval_metrics(
        direct_path=direct,
        hyde_path=hyde,
        qrels_path=qrels,
    )

    assert summary["metric_depths"] == [5, 10]
    assert "precision_at_20" not in summary["aggregate_metrics_by_mode"]["direct"]
    assert "precision_at_20" not in per_query_rows[0]
