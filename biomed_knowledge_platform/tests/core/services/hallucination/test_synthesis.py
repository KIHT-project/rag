from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pytest

from biomed_platform.api.models.generated import schemas
from biomed_platform.core.domains.llm import LlmChatMessage
from biomed_platform.core.domains.retrieval import ChunkCandidate
from biomed_platform.core.ports.llm import LlmClientPort
from biomed_platform.core.services.hallucination.synthesis import (
    MAX_RISK_FACTORS,
    synthesize_answer,
)


@dataclass(slots=True)
class _FakeLlm(LlmClientPort):
    responses: list[str]

    async def chat(
        self,
        *,
        model_id: str,
        messages: Sequence[LlmChatMessage],
        options: Mapping[str, Any] | None = None,
    ) -> str:
        if not self.responses:
            return ""
        return self.responses.pop(0)

    async def generate_text(
        self,
        *,
        model_id: str,
        prompt: str,
        options: Mapping[str, Any] | None = None,
        system: str | None = None,
    ) -> str:
        raise NotImplementedError


def _chunk(*, chunk_id: str, text: str) -> ChunkCandidate:
    return ChunkCandidate(
        chunk_id=chunk_id,
        doc_id="d1",
        doi="10.1/x",
        title="t",
        year=2020,
        section="s",
        source_type=schemas.SourceType.pubmed_abstract,
        score=0.9,
        chunk_text=text,
    )


def _lim_texts(limitations: Sequence[object]) -> list[str]:
    out: list[str] = []
    for it in limitations or []:
        if isinstance(it, str):
            s = it.strip()
            if s:
                out.append(s)
            continue

        root = getattr(it, "root", None)
        if isinstance(root, str):
            s = root.strip()
            if s:
                out.append(s)
            continue

        s = str(it).strip()
        if s:
            out.append(s)
    return out


@pytest.mark.anyio
async def test_synthesize_answer_parses_valid_json() -> None:
    llm_output = {
        "answer": {
            "summary": "s",
            "risk_factors": [
                {
                    "rank": 1,
                    "normalized_name": "hypertension",
                    "aliases": ["high blood pressure"],
                    "confidence": 0.8,
                    "rationale": "supported by chunk",
                    "citations": ["c1"],
                }
            ],
            "limitations": ["limited evidence"],
        },
        "citations": [
            {
                "chunk_id": "c1",
                "doi": "10.1/x",
                "title": "t",
                "year": 2020,
                "snippet": "excerpt",
            }
        ],
    }

    llm = _FakeLlm(responses=[json.dumps(llm_output)])
    chunks = [_chunk(chunk_id="c1", text="some evidence")]

    res = await synthesize_answer(
        llm=llm,
        model_id="m",
        question="q",
        selected_chunks=chunks,
        max_context_chars=1000,
        max_json_retries=0,
    )

    assert res.answer.summary == "s"
    assert len(res.answer.risk_factors) == 1
    assert res.answer.risk_factors[0].citations == ["c1"]
    assert len(res.citations) == 1
    assert res.citations[0].chunk_id == "c1"


@pytest.mark.anyio
async def test_synthesize_answer_invalid_json_retries_then_returns_fallback() -> None:
    llm = _FakeLlm(responses=["not json", "still not json"])
    chunks = [_chunk(chunk_id="c1", text="some evidence")]

    res = await synthesize_answer(
        llm=llm,
        model_id="m",
        question="q",
        selected_chunks=chunks,
        max_context_chars=1000,
        max_json_retries=1,
    )

    assert res.answer.summary
    assert res.answer.risk_factors == []
    assert res.citations == []

    lim = _lim_texts(res.answer.limitations)
    assert any("invalid_json" in x for x in lim)


@pytest.mark.anyio
async def test_synthesize_answer_unknown_chunk_id_in_risk_factor_is_filtered_and_backfilled() -> None:
    llm_output = {
        "answer": {
            "summary": "s",
            "risk_factors": [
                {
                    "rank": 1,
                    "normalized_name": "rf",
                    "aliases": ["a"],
                    "confidence": 0.5,
                    "rationale": "r",
                    "citations": ["c_missing"],
                }
            ],
            "limitations": [],
        },
        "citations": [],
    }

    llm = _FakeLlm(responses=[json.dumps(llm_output)])
    chunks = [_chunk(chunk_id="c1", text="some evidence")]

    res = await synthesize_answer(
        llm=llm,
        model_id="m",
        question="q",
        selected_chunks=chunks,
        max_context_chars=1000,
        max_json_retries=0,
    )

    assert len(res.answer.risk_factors) == 1
    assert res.answer.risk_factors[0].citations == ["c1"]

    assert len(res.citations) == 1
    assert res.citations[0].chunk_id == "c1"

    lim = _lim_texts(res.answer.limitations)
    assert any("assigned from the overall cited evidence" in x for x in lim)


@pytest.mark.anyio
async def test_synthesize_answer_caps_risk_factors() -> None:
    rfs: list[dict[str, Any]] = []
    for i in range(MAX_RISK_FACTORS + 3):
        rfs.append(
            {
                "rank": i + 1,
                "normalized_name": f"rf{i}",
                "aliases": [],
                "confidence": 0.5,
                "rationale": "r",
                "citations": ["c1"],
            }
        )

    llm_output = {
        "answer": {"summary": "s", "risk_factors": rfs, "limitations": []},
        "citations": [
            {"chunk_id": "c1", "doi": "10.1/x", "title": "t", "year": 2020, "snippet": "e"}
        ],
    }

    llm = _FakeLlm(responses=[json.dumps(llm_output)])
    chunks = [_chunk(chunk_id="c1", text="some evidence")]

    res = await synthesize_answer(
        llm=llm,
        model_id="m",
        question="q",
        selected_chunks=chunks,
        max_context_chars=1000,
        max_json_retries=0,
    )

    assert len(res.answer.risk_factors) == MAX_RISK_FACTORS


@pytest.mark.anyio
async def test_synthesize_answer_backfills_missing_risk_factor_citations() -> None:
    llm_output = {
        "answer": {
            "summary": "s",
            "risk_factors": [
                {
                    "rank": 1,
                    "normalized_name": "rf",
                    "aliases": [],
                    "confidence": 0.5,
                    "rationale": "r",
                    "citations": [],
                }
            ],
            "limitations": [],
        },
        "citations": [{"chunk_id": "c1", "doi": "", "title": "", "year": 0, "snippet": ""}],
    }

    llm = _FakeLlm(responses=[json.dumps(llm_output)])
    chunks = [_chunk(chunk_id="c1", text="some evidence")]

    res = await synthesize_answer(
        llm=llm,
        model_id="m",
        question="q",
        selected_chunks=chunks,
        max_context_chars=1000,
        max_json_retries=0,
    )

    assert len(res.answer.risk_factors) == 1
    assert res.answer.risk_factors[0].citations == ["c1"]

    lim = _lim_texts(res.answer.limitations)
    assert any("assigned from the overall cited evidence" in x for x in lim)


@pytest.mark.anyio
async def test_synthesize_answer_enforces_min_top_level_citations_limitation() -> None:
    llm_output = {
        "answer": {"summary": "s", "risk_factors": [], "limitations": []},
        "citations": [{"chunk_id": "c1", "doi": "", "title": "", "year": 0, "snippet": ""}],
    }

    llm = _FakeLlm(responses=[json.dumps(llm_output)])
    chunks = [_chunk(chunk_id="c1", text="some evidence")]

    res = await synthesize_answer(
        llm=llm,
        model_id="m",
        question="q",
        selected_chunks=chunks,
        max_context_chars=1000,
        max_json_retries=0,
    )

    lim = _lim_texts(res.answer.limitations)
    assert any("Evidence base is limited" in x for x in lim)
    assert any("only 1 unique citations" in x for x in lim)
