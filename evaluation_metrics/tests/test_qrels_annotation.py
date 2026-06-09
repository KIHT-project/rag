from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation_metrics.src.qrels_annotation import create_template, generate_qrels, validate_annotations


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_annotations(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "query_id",
        "question",
        "resolved_doc_id",
        "doc_id",
        "chunk_id",
        "doi",
        "pmid",
        "title",
        "relevance",
        "rationale",
        "annotator",
        "adjudication_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_create_template_pools_direct_and_hyde_candidates(tmp_path: Path) -> None:
    direct = tmp_path / "retrieval_direct.jsonl"
    hyde = tmp_path / "retrieval_hyde.jsonl"
    pooled = tmp_path / "pooled_candidates.jsonl"
    template = tmp_path / "qrels_annotation_template.csv"

    _write_jsonl(
        direct,
        [
            {"query_id": "q2", "question": "Question 2", "doc_id": "doc-b", "title": "B"},
            {"query_id": "q1", "question": "Question 1", "pmid": "11", "title": "A"},
        ],
    )
    _write_jsonl(
        hyde,
        [
            {"query_id": "q1", "question": "Question 1", "doi": "10.1/x", "title": "C"},
            {"query_id": "q2", "question": "Question 2", "chunk_id": "chunk-z", "title": "D"},
        ],
    )

    create_template(
        direct_path=direct,
        hyde_path=hyde,
        pooled_output_path=pooled,
        template_output_path=template,
    )

    assert _read_jsonl(pooled) == [
        {
            "query_id": "q1",
            "question": "Question 1",
            "resolved_doc_id": "10.1/x",
            "doc_id": "",
            "chunk_id": "",
            "doi": "10.1/x",
            "pmid": "",
            "title": "C",
            "corpus_version": "",
            "run_id": "",
        },
        {
            "query_id": "q1",
            "question": "Question 1",
            "resolved_doc_id": "11",
            "doc_id": "",
            "chunk_id": "",
            "doi": "",
            "pmid": "11",
            "title": "A",
            "corpus_version": "",
            "run_id": "",
        },
        {
            "query_id": "q2",
            "question": "Question 2",
            "resolved_doc_id": "chunk-z",
            "doc_id": "",
            "chunk_id": "chunk-z",
            "doi": "",
            "pmid": "",
            "title": "D",
            "corpus_version": "",
            "run_id": "",
        },
        {
            "query_id": "q2",
            "question": "Question 2",
            "resolved_doc_id": "doc-b",
            "doc_id": "doc-b",
            "chunk_id": "",
            "doi": "",
            "pmid": "",
            "title": "B",
            "corpus_version": "",
            "run_id": "",
        },
    ]


def test_create_template_deduplicates_same_candidate_across_modes(tmp_path: Path) -> None:
    direct = tmp_path / "retrieval_direct.jsonl"
    hyde = tmp_path / "retrieval_hyde.jsonl"
    pooled = tmp_path / "pooled_candidates.jsonl"
    template = tmp_path / "qrels_annotation_template.csv"

    _write_jsonl(
        direct,
        [
            {
                "query_id": "q1",
                "question": "Question 1",
                "doc_id": "doc-1",
                "title": "Title 1",
                "retrieval_mode": "direct",
                "rank": 1,
                "score": 0.9,
                "candidate_origin": "direct",
                "selected_for_context": True,
            }
        ],
    )
    _write_jsonl(
        hyde,
        [
            {
                "query_id": "q1",
                "question": "Question 1",
                "doc_id": "doc-1",
                "title": "Title 1",
                "retrieval_mode": "hyde",
                "rank": 5,
                "score": 0.4,
                "candidate_origin": "hyde",
                "selected_for_context": False,
            }
        ],
    )

    create_template(
        direct_path=direct,
        hyde_path=hyde,
        pooled_output_path=pooled,
        template_output_path=template,
    )

    pooled_rows = _read_jsonl(pooled)
    template_rows = _read_csv(template)

    assert len(pooled_rows) == 1
    assert len(template_rows) == 1
    assert "retrieval_mode" not in template_rows[0]
    assert "rank" not in template_rows[0]
    assert "score" not in template_rows[0]
    assert "candidate_origin" not in template_rows[0]
    assert "selected_for_context" not in template_rows[0]


@pytest.mark.parametrize(
    ("row", "expected_id"),
    [
        ({"query_id": "q1", "question": "Q", "doc_id": "doc-1", "pmid": "p1", "doi": "d1", "chunk_id": "c1"}, "doc-1"),
        ({"query_id": "q1", "question": "Q", "pmid": "p1", "doi": "d1", "chunk_id": "c1"}, "p1"),
        ({"query_id": "q1", "question": "Q", "doi": "d1", "chunk_id": "c1"}, "d1"),
        ({"query_id": "q1", "question": "Q", "chunk_id": "c1"}, "c1"),
    ],
)
def test_identifier_fallback_priority(tmp_path: Path, row: dict, expected_id: str) -> None:
    direct = tmp_path / "retrieval_direct.jsonl"
    hyde = tmp_path / "retrieval_hyde.jsonl"
    pooled = tmp_path / "pooled_candidates.jsonl"
    template = tmp_path / "qrels_annotation_template.csv"

    _write_jsonl(direct, [row])
    _write_jsonl(hyde, [])

    create_template(
        direct_path=direct,
        hyde_path=hyde,
        pooled_output_path=pooled,
        template_output_path=template,
    )

    pooled_rows = _read_jsonl(pooled)
    assert pooled_rows[0]["resolved_doc_id"] == expected_id


def test_create_template_fails_without_stable_identifier(tmp_path: Path) -> None:
    direct = tmp_path / "retrieval_direct.jsonl"
    hyde = tmp_path / "retrieval_hyde.jsonl"

    _write_jsonl(direct, [{"query_id": "q1", "question": "Question 1", "title": "No id"}])
    _write_jsonl(hyde, [])

    with pytest.raises(ValueError, match="missing a stable identifier"):
        create_template(
            direct_path=direct,
            hyde_path=hyde,
            pooled_output_path=tmp_path / "pooled_candidates.jsonl",
            template_output_path=tmp_path / "qrels_annotation_template.csv",
        )


def test_validator_fails_on_duplicate_rows(tmp_path: Path) -> None:
    pooled = tmp_path / "pooled_candidates.jsonl"
    annotations = tmp_path / "annotations.csv"
    _write_jsonl(
        pooled,
        [
            {"query_id": "q1", "question": "Q1", "doc_id": "doc-1", "title": "A"},
        ],
    )
    _write_annotations(
        annotations,
        [
            {
                "query_id": "q1",
                "question": "Q1",
                "resolved_doc_id": "doc-1",
                "doc_id": "doc-1",
                "chunk_id": "",
                "doi": "",
                "pmid": "",
                "title": "A",
                "relevance": "0",
                "rationale": "",
                "annotator": "ann",
                "adjudication_status": "",
            },
            {
                "query_id": "q1",
                "question": "Q1",
                "resolved_doc_id": "doc-1",
                "doc_id": "doc-1",
                "chunk_id": "",
                "doi": "",
                "pmid": "",
                "title": "A",
                "relevance": "0",
                "rationale": "",
                "annotator": "ann",
                "adjudication_status": "",
            },
        ],
    )

    with pytest.raises(ValueError, match="Duplicate annotation row"):
        validate_annotations(pooled_path=pooled, annotations_path=annotations)


def test_validator_fails_on_missing_rows(tmp_path: Path) -> None:
    pooled = tmp_path / "pooled_candidates.jsonl"
    annotations = tmp_path / "annotations.csv"
    _write_jsonl(
        pooled,
        [
            {"query_id": "q1", "question": "Q1", "doc_id": "doc-1", "title": "A"},
            {"query_id": "q1", "question": "Q1", "doc_id": "doc-2", "title": "B"},
        ],
    )
    _write_annotations(
        annotations,
        [
            {
                "query_id": "q1",
                "question": "Q1",
                "resolved_doc_id": "doc-1",
                "doc_id": "doc-1",
                "chunk_id": "",
                "doi": "",
                "pmid": "",
                "title": "A",
                "relevance": "0",
                "rationale": "",
                "annotator": "",
                "adjudication_status": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="Missing annotation row"):
        validate_annotations(pooled_path=pooled, annotations_path=annotations)


def test_validator_fails_on_unknown_rows(tmp_path: Path) -> None:
    pooled = tmp_path / "pooled_candidates.jsonl"
    annotations = tmp_path / "annotations.csv"
    _write_jsonl(
        pooled,
        [
            {"query_id": "q1", "question": "Q1", "doc_id": "doc-1", "title": "A"},
        ],
    )
    _write_annotations(
        annotations,
        [
            {
                "query_id": "q1",
                "question": "Q1",
                "resolved_doc_id": "doc-x",
                "doc_id": "doc-x",
                "chunk_id": "",
                "doi": "",
                "pmid": "",
                "title": "A",
                "relevance": "0",
                "rationale": "",
                "annotator": "",
                "adjudication_status": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="Unknown annotation candidate"):
        validate_annotations(pooled_path=pooled, annotations_path=annotations)


def test_validator_fails_on_invalid_relevance(tmp_path: Path) -> None:
    pooled = tmp_path / "pooled_candidates.jsonl"
    annotations = tmp_path / "annotations.csv"
    _write_jsonl(
        pooled,
        [
            {"query_id": "q1", "question": "Q1", "doc_id": "doc-1", "title": "A"},
        ],
    )
    _write_annotations(
        annotations,
        [
            {
                "query_id": "q1",
                "question": "Q1",
                "resolved_doc_id": "doc-1",
                "doc_id": "doc-1",
                "chunk_id": "",
                "doi": "",
                "pmid": "",
                "title": "A",
                "relevance": "3",
                "rationale": "",
                "annotator": "",
                "adjudication_status": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="relevance must be one of 0, 1, 2"):
        validate_annotations(pooled_path=pooled, annotations_path=annotations)


def test_validator_fails_on_empty_relevance(tmp_path: Path) -> None:
    pooled = tmp_path / "pooled_candidates.jsonl"
    annotations = tmp_path / "annotations.csv"
    _write_jsonl(
        pooled,
        [
            {"query_id": "q1", "question": "Q1", "doc_id": "doc-1", "title": "A"},
        ],
    )
    _write_annotations(
        annotations,
        [
            {
                "query_id": "q1",
                "question": "Q1",
                "resolved_doc_id": "doc-1",
                "doc_id": "doc-1",
                "chunk_id": "",
                "doi": "",
                "pmid": "",
                "title": "A",
                "relevance": "",
                "rationale": "",
                "annotator": "",
                "adjudication_status": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="relevance is empty"):
        validate_annotations(pooled_path=pooled, annotations_path=annotations)


def test_validator_fails_on_empty_rationale_for_positive_relevance(tmp_path: Path) -> None:
    pooled = tmp_path / "pooled_candidates.jsonl"
    annotations = tmp_path / "annotations.csv"
    _write_jsonl(
        pooled,
        [
            {"query_id": "q1", "question": "Q1", "doc_id": "doc-1", "title": "A"},
        ],
    )
    _write_annotations(
        annotations,
        [
            {
                "query_id": "q1",
                "question": "Q1",
                "resolved_doc_id": "doc-1",
                "doc_id": "doc-1",
                "chunk_id": "",
                "doi": "",
                "pmid": "",
                "title": "A",
                "relevance": "2",
                "rationale": "",
                "annotator": "",
                "adjudication_status": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="rationale is required when relevance is 2"):
        validate_annotations(pooled_path=pooled, annotations_path=annotations)


def test_valid_annotations_generate_deterministic_qrels(tmp_path: Path) -> None:
    pooled = tmp_path / "pooled_candidates.jsonl"
    annotations = tmp_path / "annotations.csv"
    qrels = tmp_path / "qrels.tsv"
    _write_jsonl(
        pooled,
        [
            {"query_id": "q2", "question": "Q2", "doc_id": "doc-b", "title": "B"},
            {"query_id": "q1", "question": "Q1", "pmid": "111", "title": "A"},
        ],
    )
    _write_annotations(
        annotations,
        [
            {
                "query_id": "q2",
                "question": "Q2",
                "resolved_doc_id": "doc-b",
                "doc_id": "doc-b",
                "chunk_id": "",
                "doi": "",
                "pmid": "",
                "title": "B",
                "relevance": "1",
                "rationale": "Contextually relevant.",
                "annotator": "ann",
                "adjudication_status": "",
            },
            {
                "query_id": "q1",
                "question": "Q1",
                "resolved_doc_id": "111",
                "doc_id": "",
                "chunk_id": "",
                "doi": "",
                "pmid": "111",
                "title": "A",
                "relevance": "2",
                "rationale": "Directly answers the question.",
                "annotator": "ann",
                "adjudication_status": "accepted",
            },
        ],
    )

    generate_qrels(
        pooled_path=pooled,
        annotations_path=annotations,
        qrels_output_path=qrels,
    )

    assert qrels.read_text(encoding="utf-8").splitlines() == [
        "query_id\tdoc_id\trelevance",
        "q1\t111\t2",
        "q2\tdoc-b\t1",
    ]
