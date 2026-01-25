from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pytest
from fastapi.exceptions import RequestValidationError

from biomed_platform.api.models.generated import schemas
from biomed_platform.core.domains.llm import LlmChatMessage
from biomed_platform.core.domains.retrieval import ChunkCandidate
from biomed_platform.core.ports.llm import LlmClientPort
from biomed_platform.core.services.hallucination.synthesis import synthesize_answer


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


@pytest.mark.anyio
async def test_synthesize_answer_parses_valid_json() -> None:
    # Given
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

    # When
    res = await synthesize_answer(
        llm=llm,
        model_id="m",
        question="q",
        selected_chunks=chunks,
        max_context_chars=1000,
        max_json_retries=0,
    )

    # Then
    assert res.answer.summary == "s"
    assert len(res.answer.risk_factors) == 1
    assert res.answer.risk_factors[0].citations == ["c1"]
    assert len(res.citations) == 1
    assert res.citations[0].chunk_id == "c1"


@pytest.mark.anyio
async def test_synthesize_answer_invalid_json_retries_then_raises_request_validation_error() -> None:
    # Given, invalid JSON twice
    llm = _FakeLlm(responses=["not json", "still not json"])
    chunks = [_chunk(chunk_id="c1", text="some evidence")]

    # When, Then
    with pytest.raises(RequestValidationError):
        await synthesize_answer(
            llm=llm,
            model_id="m",
            question="q",
            selected_chunks=chunks,
            max_context_chars=1000,
            max_json_retries=1,
        )


@pytest.mark.anyio
async def test_synthesize_answer_rejects_unknown_chunk_id_citations() -> None:
    # Given, model references c_missing which is not in selected chunks
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

    # When, Then
    with pytest.raises(RequestValidationError):
        await synthesize_answer(
            llm=llm,
            model_id="m",
            question="q",
            selected_chunks=chunks,
            max_context_chars=1000,
            max_json_retries=0,
        )
