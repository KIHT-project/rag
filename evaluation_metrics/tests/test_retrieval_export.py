from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation_metrics.src.retrieval_export import run_retrieval_export
from evaluation_metrics.src.schemas.models import AskResponse, RunContext


def _write_queries(path: Path) -> None:
    rows = [
        {"_id": "q2", "text": "Question 2"},
        {"_id": "q1", "text": "Question 1"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


class _FakeRagClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

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
        self.calls.append(
            {
                "question": question,
                "filters": filters,
                "hyde_enabled": hyde_enabled,
                "debug_enabled": debug_enabled,
                "debug_ask_max_chunks_candidate": debug_ask_max_chunks_candidate,
                "hyde_header_name": hyde_header_name,
                "hyde_header_value": hyde_header_value,
            }
        )
        q_suffix = "1" if question.endswith("1") else "2"
        if hyde_enabled:
            raw = {
                "debug": {
                    "hyde_candidates": [
                        {
                            "chunk_id": f"hyde-{q_suffix}-b",
                            "doc_id": f"doc-{q_suffix}-b",
                            "doi": f"10.1/{q_suffix}b",
                            "pmid": "",
                            "title": f"HyDE B {q_suffix}",
                            "score": 0.5,
                            "origin": "hyde",
                            "selected_for_context": False,
                        },
                        {
                            "chunk_id": f"hyde-{q_suffix}-a",
                            "doc_id": f"doc-{q_suffix}-a",
                            "doi": f"10.1/{q_suffix}a",
                            "pmid": "",
                            "title": f"HyDE A {q_suffix}",
                            "score": 0.9,
                            "origin": "hyde",
                            "selected_for_context": True,
                        },
                    ],
                    "selected_candidates": [
                        {
                            "chunk_id": f"final-hyde-{q_suffix}",
                            "doc_id": f"doc-final-hyde-{q_suffix}",
                            "doi": f"10.1/fh{q_suffix}",
                            "pmid": "",
                            "title": f"Final HyDE {q_suffix}",
                            "score": 0.8,
                            "origin": "both",
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
                            "chunk_id": f"direct-{q_suffix}-b",
                            "doc_id": f"doc-{q_suffix}-b",
                            "doi": f"10.1/{q_suffix}b",
                            "pmid": "",
                            "title": f"Direct B {q_suffix}",
                            "score": 0.4,
                            "origin": "question",
                            "selected_for_context": False,
                        },
                        {
                            "chunk_id": f"direct-{q_suffix}-a",
                            "doc_id": f"doc-{q_suffix}-a",
                            "doi": f"10.1/{q_suffix}a",
                            "pmid": "",
                            "title": f"Direct A {q_suffix}",
                            "score": 0.9,
                            "origin": "question",
                            "selected_for_context": True,
                        },
                    ],
                    "selected_candidates": [
                        {
                            "chunk_id": f"final-direct-{q_suffix}",
                            "doc_id": f"doc-final-direct-{q_suffix}",
                            "doi": f"10.1/fd{q_suffix}",
                            "pmid": "",
                            "title": f"Final Direct {q_suffix}",
                            "score": 0.95,
                            "origin": "question",
                            "selected_for_context": True,
                        }
                    ],
                }
            }

        return AskResponse(answer=None, citations=[], raw=raw)


@pytest.mark.anyio
async def test_run_retrieval_export_writes_expected_artifacts(tmp_path: Path) -> None:
    queries = tmp_path / "queries.jsonl"
    _write_queries(queries)
    rag = _FakeRagClient()
    ctx = RunContext(run_id="run-123", run_dir=str(tmp_path / "run"))

    outputs = await run_retrieval_export(
        ctx=ctx,
        rag=rag,
        queries_jsonl=queries,
        hyde_header_name="X-HyDE-Enabled",
        hyde_header_value="true",
        retrieval_pool_depth=50,
        max_queries=2,
    )

    direct_rows = _read_jsonl(outputs["direct"])
    hyde_rows = _read_jsonl(outputs["hyde"])
    final_rows = _read_jsonl(outputs["final_context"])

    assert [row["query_id"] for row in direct_rows] == ["q1", "q1", "q2", "q2"]
    assert [row["rank"] for row in direct_rows] == [1, 2, 1, 2]
    assert [row["chunk_id"] for row in direct_rows[:2]] == ["direct-1-b", "direct-1-a"]
    assert direct_rows[1]["selected_for_context"] is True
    assert direct_rows[1]["candidate_origin"] == "question"
    assert direct_rows[1]["run_id"] == "run-123"
    assert direct_rows[1]["corpus_version"] == ""

    assert [row["query_id"] for row in hyde_rows] == ["q1", "q1", "q2", "q2"]
    assert [row["rank"] for row in hyde_rows] == [1, 2, 1, 2]
    assert hyde_rows[1]["selected_for_context"] is True
    assert hyde_rows[1]["candidate_origin"] == "hyde"

    assert [row["query_id"] for row in final_rows] == ["q1", "q1", "q2", "q2"]
    assert [row["retrieval_mode"] for row in final_rows] == ["direct", "hyde", "direct", "hyde"]
    assert [row["rank"] for row in final_rows] == [1, 1, 1, 1]
    assert all(row["selected_for_context"] is True for row in final_rows)
    assert final_rows[1]["candidate_origin"] == "both"

    assert len(rag.calls) == 4
    assert all(call["debug_enabled"] is True for call in rag.calls)
    assert all(call["debug_ask_max_chunks_candidate"] == 50 for call in rag.calls)
