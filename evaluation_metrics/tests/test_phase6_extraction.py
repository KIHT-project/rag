from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation_metrics.src.phases.phase6_extraction import run_phase6_extraction


def _task(
    *,
    doi: str,
    reports_risk: str,
    confidence: int | None,
    reason_text: str,
    abstract: str = "",
) -> dict:
    results: list[dict] = [
        {
            "from_name": "reports_risk_factors",
            "type": "choices",
            "value": {"choices": [reports_risk]},
        },
        {
            "from_name": "reason_label",
            "type": "labels",
            "value": {"text": reason_text, "labels": ["Inclusion Evidence"]},
        },
    ]
    if confidence is not None:
        results.append(
            {
                "from_name": "confidence_reports_risk_factors",
                "type": "rating",
                "value": {"rating": confidence},
            }
        )

    return {
        "data": {"doi": doi, "abstract": abstract},
        "annotations": [{"result": results}],
    }


def _phase3_record(*, query_id: str, risk_factors: list[str], cited_dois: list[str]) -> dict:
    return {
        "query_id": query_id,
        "mode": "rag_no_hyde",
        "answer_raw": {
            "answer": {
                "summary": "summary",
                "risk_factors": [
                    {"normalized_name": rf, "aliases": []} for rf in risk_factors
                ],
            },
            "citations": [{"doi": doi} for doi in cited_dois],
        },
    }


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def test_phase6_extraction_computes_expected_micro_and_macro_metrics(tmp_path: Path) -> None:
    tasks_clean = tmp_path / "tasks_clean.json"
    _write_json(
        tasks_clean,
        [
            _task(
                doi="10.1000/A",
                reports_risk="Yes",
                confidence=5,
                reason_text="Obesity and smoking were associated with venous thrombosis.",
            ),
            _task(
                doi="10.1000/B",
                reports_risk="Yes",
                confidence=5,
                reason_text="History of DVT increased thrombosis risk.",
            ),
        ],
    )

    phase3 = tmp_path / "phase3_answers_rag_no_hyde.jsonl"
    _write_jsonl(
        phase3,
        [
            _phase3_record(
                query_id="q1",
                risk_factors=["obesity", "hypertension"],
                cited_dois=["10.1000/A"],
            ),
            _phase3_record(
                query_id="q2",
                risk_factors=["prior dvt", "smoking"],
                cited_dois=["10.1000/B"],
            ),
        ],
    )

    out_dir = tmp_path / "out"
    metrics = run_phase6_extraction(
        input_jsonl=phase3,
        tasks_clean_json=tasks_clean,
        out_dir=out_dir,
        no_gold_policy="skip",
        canonical_factors={
            "obesity": ["obesity"],
            "smoking": ["smoking"],
            "hypertension": ["hypertension"],
            "prior_vte": ["prior dvt", "history of dvt"],
        },
    )

    assert metrics["micro_precision"] == pytest.approx(0.5, rel=1e-6)
    assert metrics["micro_recall"] == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert metrics["micro_f1"] == pytest.approx(4.0 / 7.0, rel=1e-6)
    assert metrics["macro_precision"] == pytest.approx(0.5, rel=1e-6)
    assert metrics["macro_recall"] == pytest.approx(0.75, rel=1e-6)
    assert metrics["macro_f1"] == pytest.approx((0.5 + (2.0 / 3.0)) / 2.0, rel=1e-6)
    assert metrics["query_coverage_rate"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["evaluated_queries"] == pytest.approx(2.0, rel=1e-6)

    assert (out_dir / "phase6_extraction_phase3_answers_rag_no_hyde_per_query.csv").exists()
    assert (out_dir / "phase6_extraction_phase3_answers_rag_no_hyde_summary.csv").exists()


def test_phase6_extraction_skip_policy_excludes_queries_without_gold(tmp_path: Path) -> None:
    tasks_clean = tmp_path / "tasks_clean.json"
    _write_json(
        tasks_clean,
        [
            _task(
                doi="10.1000/C",
                reports_risk="Yes",
                confidence=5,
                reason_text="This text has no factor keyword from our small map.",
            ),
        ],
    )

    phase3 = tmp_path / "phase3_answers_rag_no_hyde.jsonl"
    _write_jsonl(
        phase3,
        [
            _phase3_record(
                query_id="q1",
                risk_factors=["obesity"],
                cited_dois=["10.1000/C"],
            ),
        ],
    )

    out_dir = tmp_path / "out"
    metrics = run_phase6_extraction(
        input_jsonl=phase3,
        tasks_clean_json=tasks_clean,
        out_dir=out_dir,
        no_gold_policy="skip",
        canonical_factors={"obesity": ["obesity"]},
    )

    assert metrics["micro_f1"] == pytest.approx(0.0, rel=1e-6)
    assert metrics["query_coverage_rate"] == pytest.approx(0.0, rel=1e-6)
    assert metrics["evaluated_queries"] == pytest.approx(0.0, rel=1e-6)
    assert metrics["skipped_no_gold"] == pytest.approx(1.0, rel=1e-6)
    assert (out_dir / "phase6_extraction_phase3_answers_rag_no_hyde_excluded.csv").exists()


def test_phase6_extraction_empty_policy_keeps_queries_without_gold(tmp_path: Path) -> None:
    tasks_clean = tmp_path / "tasks_clean.json"
    _write_json(
        tasks_clean,
        [
            _task(
                doi="10.1000/D",
                reports_risk="Yes",
                confidence=5,
                reason_text="No canonical factor appears here.",
            ),
        ],
    )

    phase3 = tmp_path / "phase3_answers_rag_no_hyde.jsonl"
    _write_jsonl(
        phase3,
        [
            _phase3_record(
                query_id="q1",
                risk_factors=["obesity"],
                cited_dois=["10.1000/D"],
            ),
        ],
    )

    out_dir = tmp_path / "out"
    metrics = run_phase6_extraction(
        input_jsonl=phase3,
        tasks_clean_json=tasks_clean,
        out_dir=out_dir,
        no_gold_policy="empty",
        canonical_factors={"obesity": ["obesity"]},
    )

    assert metrics["micro_precision"] == pytest.approx(0.0, rel=1e-6)
    assert metrics["micro_recall"] == pytest.approx(0.0, rel=1e-6)
    assert metrics["micro_f1"] == pytest.approx(0.0, rel=1e-6)
    assert metrics["query_coverage_rate"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["evaluated_queries"] == pytest.approx(1.0, rel=1e-6)


def test_phase6_extraction_uses_abstract_when_reason_has_no_hits(tmp_path: Path) -> None:
    tasks_clean = tmp_path / "tasks_clean.json"
    _write_json(
        tasks_clean,
        [
            _task(
                doi="10.1000/E",
                reports_risk="Yes",
                confidence=5,
                reason_text="Reason text without canonical clues.",
                abstract="Evidence mentions marker x for thrombosis risk.",
            ),
        ],
    )

    phase3 = tmp_path / "phase3_answers_rag_no_hyde.jsonl"
    _write_jsonl(
        phase3,
        [
            _phase3_record(
                query_id="q1",
                risk_factors=["marker x"],
                cited_dois=["10.1000/E"],
            ),
        ],
    )

    out_dir = tmp_path / "out"
    metrics = run_phase6_extraction(
        input_jsonl=phase3,
        tasks_clean_json=tasks_clean,
        out_dir=out_dir,
        no_gold_policy="skip",
        canonical_factors={"marker_x": ["marker x"]},
    )

    assert metrics["micro_precision"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["micro_recall"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["micro_f1"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["query_coverage_rate"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["evaluated_queries"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["skipped_no_gold"] == pytest.approx(0.0, rel=1e-6)


def test_phase6_default_canonical_factors_cover_catheter_occlusion(tmp_path: Path) -> None:
    tasks_clean = tmp_path / "tasks_clean.json"
    _write_json(
        tasks_clean,
        [
            _task(
                doi="10.1000/F",
                reports_risk="Yes",
                confidence=5,
                reason_text="Integrated catheters significantly reduced occlusion risk.",
            ),
        ],
    )

    phase3 = tmp_path / "phase3_answers_rag_no_hyde.jsonl"
    _write_jsonl(
        phase3,
        [
            _phase3_record(
                query_id="q1",
                risk_factors=["Occlusion"],
                cited_dois=["10.1000/F"],
            ),
        ],
    )

    out_dir = tmp_path / "out"
    metrics = run_phase6_extraction(
        input_jsonl=phase3,
        tasks_clean_json=tasks_clean,
        out_dir=out_dir,
        no_gold_policy="skip",
    )

    assert metrics["micro_precision"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["micro_recall"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["micro_f1"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["query_coverage_rate"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["evaluated_queries"] == pytest.approx(1.0, rel=1e-6)


def test_phase6_extraction_closed_set_drops_unknown_predicted_factors(tmp_path: Path) -> None:
    tasks_clean = tmp_path / "tasks_clean.json"
    _write_json(
        tasks_clean,
        [
            _task(
                doi="10.1000/G",
                reports_risk="Yes",
                confidence=5,
                reason_text="Obesity was a consistent risk factor.",
            ),
        ],
    )

    phase3 = tmp_path / "phase3_answers_rag_no_hyde.jsonl"
    _write_jsonl(
        phase3,
        [
            _phase3_record(
                query_id="q1",
                risk_factors=["obesity", "unmapped token"],
                cited_dois=["10.1000/G"],
            ),
        ],
    )

    out_open = tmp_path / "out_open"
    metrics_open = run_phase6_extraction(
        input_jsonl=phase3,
        tasks_clean_json=tasks_clean,
        out_dir=out_open,
        no_gold_policy="skip",
        canonical_factors={"obesity": ["obesity"]},
        pred_closed_set_only=False,
    )

    out_closed = tmp_path / "out_closed"
    metrics_closed = run_phase6_extraction(
        input_jsonl=phase3,
        tasks_clean_json=tasks_clean,
        out_dir=out_closed,
        no_gold_policy="skip",
        canonical_factors={"obesity": ["obesity"]},
        pred_closed_set_only=True,
    )

    assert metrics_open["micro_precision"] == pytest.approx(0.5, rel=1e-6)
    assert metrics_open["micro_f1"] == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert metrics_closed["micro_precision"] == pytest.approx(1.0, rel=1e-6)
    assert metrics_closed["micro_f1"] == pytest.approx(1.0, rel=1e-6)


def test_phase6_extraction_can_ignore_aliases_for_predicted_factors(tmp_path: Path) -> None:
    tasks_clean = tmp_path / "tasks_clean.json"
    _write_json(
        tasks_clean,
        [
            _task(
                doi="10.1000/H",
                reports_risk="Yes",
                confidence=5,
                reason_text="Venous thromboembolism was associated with risk.",
            ),
        ],
    )

    phase3 = tmp_path / "phase3_answers_rag_no_hyde.jsonl"
    _write_jsonl(
        phase3,
        [
            {
                "query_id": "q1",
                "mode": "rag_no_hyde",
                "answer_raw": {
                    "answer": {
                        "summary": "summary",
                        "risk_factors": [
                            {
                                "normalized_name": "previous venous thromboembolism",
                                "aliases": ["prior vte"],
                            }
                        ],
                    },
                    "citations": [{"doi": "10.1000/H"}],
                },
            }
        ],
    )

    canonical = {
        "thromboembolism": ["venous thromboembolism", "thromboembolism"],
        "prior_vte": ["prior vte", "history of vte"],
    }

    out_with_aliases = tmp_path / "out_aliases"
    metrics_with_aliases = run_phase6_extraction(
        input_jsonl=phase3,
        tasks_clean_json=tasks_clean,
        out_dir=out_with_aliases,
        no_gold_policy="skip",
        canonical_factors=canonical,
        pred_include_aliases=True,
    )

    out_without_aliases = tmp_path / "out_no_aliases"
    metrics_without_aliases = run_phase6_extraction(
        input_jsonl=phase3,
        tasks_clean_json=tasks_clean,
        out_dir=out_without_aliases,
        no_gold_policy="skip",
        canonical_factors=canonical,
        pred_include_aliases=False,
    )

    assert metrics_with_aliases["micro_precision"] == pytest.approx(0.5, rel=1e-6)
    assert metrics_without_aliases["micro_precision"] == pytest.approx(1.0, rel=1e-6)


def test_phase6_extraction_can_include_summary_factors(tmp_path: Path) -> None:
    tasks_clean = tmp_path / "tasks_clean.json"
    _write_json(
        tasks_clean,
        [
            _task(
                doi="10.1000/I",
                reports_risk="Yes",
                confidence=5,
                reason_text="Chronic heart failure increased thrombosis risk.",
            ),
        ],
    )

    phase3 = tmp_path / "phase3_answers_rag_no_hyde.jsonl"
    _write_jsonl(
        phase3,
        [
            {
                "query_id": "q1",
                "mode": "rag_no_hyde",
                "answer_raw": {
                    "answer": {
                        "summary": "Chronic heart failure was a strong risk factor.",
                        "risk_factors": [],
                    },
                    "citations": [{"doi": "10.1000/I"}],
                },
            }
        ],
    )

    out_disabled = tmp_path / "out_summary_disabled"
    metrics_disabled = run_phase6_extraction(
        input_jsonl=phase3,
        tasks_clean_json=tasks_clean,
        out_dir=out_disabled,
        no_gold_policy="skip",
        canonical_factors={"heart_failure": ["heart failure", "chronic heart failure"]},
        pred_include_summary_factors=False,
    )

    out_enabled = tmp_path / "out_summary_enabled"
    metrics_enabled = run_phase6_extraction(
        input_jsonl=phase3,
        tasks_clean_json=tasks_clean,
        out_dir=out_enabled,
        no_gold_policy="skip",
        canonical_factors={"heart_failure": ["heart failure", "chronic heart failure"]},
        pred_include_summary_factors=True,
    )

    assert metrics_disabled["micro_f1"] == pytest.approx(0.0, rel=1e-6)
    assert metrics_enabled["micro_f1"] == pytest.approx(1.0, rel=1e-6)


def test_phase6_extraction_can_include_citation_snippet_factors(tmp_path: Path) -> None:
    tasks_clean = tmp_path / "tasks_clean.json"
    _write_json(
        tasks_clean,
        [
            _task(
                doi="10.1000/J",
                reports_risk="Yes",
                confidence=5,
                reason_text="Catheter occlusion and thrombophlebitis were frequent.",
            ),
        ],
    )

    phase3 = tmp_path / "phase3_answers_rag_no_hyde.jsonl"
    _write_jsonl(
        phase3,
        [
            {
                "query_id": "q1",
                "mode": "rag_no_hyde",
                "answer_raw": {
                    "answer": {
                        "summary": "summary without explicit factors",
                        "risk_factors": [],
                    },
                    "citations": [
                        {
                            "doi": "10.1000/J",
                            "snippet": "This cohort had catheter occlusion and thrombophlebitis events.",
                        }
                    ],
                },
            }
        ],
    )

    canonical = {
        "catheter_occlusion": ["catheter occlusion", "occlusion"],
        "thrombophlebitis": ["thrombophlebitis"],
    }

    out_disabled = tmp_path / "out_snippets_disabled"
    metrics_disabled = run_phase6_extraction(
        input_jsonl=phase3,
        tasks_clean_json=tasks_clean,
        out_dir=out_disabled,
        no_gold_policy="skip",
        canonical_factors=canonical,
        pred_include_citation_snippets=False,
    )

    out_enabled = tmp_path / "out_snippets_enabled"
    metrics_enabled = run_phase6_extraction(
        input_jsonl=phase3,
        tasks_clean_json=tasks_clean,
        out_dir=out_enabled,
        no_gold_policy="skip",
        canonical_factors=canonical,
        pred_include_citation_snippets=True,
    )

    assert metrics_disabled["micro_f1"] == pytest.approx(0.0, rel=1e-6)
    assert metrics_enabled["micro_f1"] == pytest.approx(1.0, rel=1e-6)
