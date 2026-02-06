from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest.mock import AsyncMock

import pytest

from biomed_platform.api.models.generated import schemas
from biomed_platform.core.domains.retrieval import ChunkCandidate
from biomed_platform.core.errors.errors import BusinessError
from biomed_platform.core.services.hallucination.synthesis import synthesize_answer
from biomed_platform.core.use_cases.ask import AskUseCase

if not hasattr(schemas, "RerankerMode"):
    class RerankerMode(StrEnum):
        off = "off"

    schemas.RerankerMode = RerankerMode

def _cand(*, chunk_id: str, score: float, text: str) -> ChunkCandidate:
    return ChunkCandidate(
        chunk_id=chunk_id,
        doc_id="d",
        doi="10.1/x",
        title="t",
        year=2020,
        section="s",
        source_type=schemas.SourceType.pubmed_abstract,
        score=float(score),
        chunk_text=text,
    )


def _synthesis_result(*, chunk_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        answer=schemas.AnswerPayload(summary="ok", risk_factors=[], limitations=[]),
        citations=[
            schemas.Citation(
                chunk_id=chunk_id,
                doi="10.1/x",
                title="t",
                year=2020,
                snippet="snip",
            )
        ],
    )


@pytest.mark.anyio
async def test_hyde_disabled_only_question_path_runs() -> None:
    embedder = AsyncMock()
    embedder.embed_texts = AsyncMock(return_value=[[0.1, 0.2]])

    vector_index = AsyncMock()
    vector_index.search_chunks = AsyncMock(
        return_value=[
            _cand(chunk_id="c1", score=0.9, text="aaa"),
            _cand(chunk_id="c2", score=0.8, text="bbb"),
        ]
    )

    hyde_generator = AsyncMock()
    synthesizer = AsyncMock(return_value=_synthesis_result(chunk_id="c1"))

    uc = AskUseCase(
        embedder=embedder,
        vector_index=vector_index,
        llm=AsyncMock(),
        hyde_generator=hyde_generator,
        synthesizer=synthesizer,
    )

    res = await uc.execute(
        request_id="r",
        question="  q ",
        filters={"year_min": 2000},
        embedding_model_id="e",
        generator_model_id="g",
        hyde_model_id="h",
        hyde_enabled=False,
        hyde_max_chars=100,
        ask_max_question_chars=50,
        ask_max_chunks_candidate=5,
        ask_max_chunks_final=2,
        ask_max_context_chars=1000,
        ask_llm_max_retries=0,
        debug_enabled=True,
    )

    hyde_generator.assert_not_called()
    assert vector_index.search_chunks.call_count == 1

    _, kwargs = vector_index.search_chunks.call_args
    assert kwargs["qfilter"] == {"year_min": 2000}

    assert res.effective_hyde_enabled is False
    assert res.answer.summary == "ok"
    assert res.debug is not None
    assert res.debug["hyde_candidates"] == []

    merged_all = res.debug["merged_candidates_all"]
    merged_usable = res.debug["merged_candidates_usable"]

    assert merged_all[0]["origin"] == "question"
    assert merged_usable[0]["origin"] == "question"


@pytest.mark.anyio
async def test_hyde_enabled_merge_dedupes_and_keeps_best_score() -> None:
    embedder = AsyncMock()
    embedder.embed_texts = AsyncMock(return_value=[[0.1]])

    vector_index = AsyncMock()
    vector_index.search_chunks = AsyncMock(
        side_effect=[
            [
                _cand(chunk_id="c1", score=0.5, text="q1"),
                _cand(chunk_id="c2", score=0.4, text="q2"),
            ],
            [
                _cand(chunk_id="c1", score=0.9, text="h1"),
                _cand(chunk_id="c3", score=0.7, text="h3"),
            ],
        ]
    )

    hyde_generator = AsyncMock(return_value="hyde text")
    synthesizer = AsyncMock(return_value=_synthesis_result(chunk_id="c1"))

    uc = AskUseCase(
        embedder=embedder,
        vector_index=vector_index,
        llm=AsyncMock(),
        hyde_generator=hyde_generator,
        synthesizer=synthesizer,
    )

    res = await uc.execute(
        request_id="r",
        question="q",
        filters=None,
        embedding_model_id="e",
        generator_model_id="g",
        hyde_model_id="h",
        hyde_enabled=True,
        hyde_max_chars=100,
        ask_max_question_chars=None,
        ask_max_chunks_candidate=10,
        ask_max_chunks_final=5,
        ask_max_context_chars=1000,
        ask_llm_max_retries=0,
        debug_enabled=True,
    )

    assert vector_index.search_chunks.call_count == 2
    assert res.debug is not None

    merged_all = res.debug["merged_candidates_all"]
    merged_usable = res.debug["merged_candidates_usable"]

    assert merged_all[0]["chunk_id"] == "c1"
    assert merged_all[0]["origin"] == "both"
    assert merged_all[0]["score"] == pytest.approx(0.9)

    assert merged_usable[0]["chunk_id"] == "c1"
    assert merged_usable[0]["origin"] == "both"
    assert merged_usable[0]["score"] > 0


@pytest.mark.anyio
async def test_filters_are_passed_to_both_retrieval_calls() -> None:
    embedder = AsyncMock()
    embedder.embed_texts = AsyncMock(return_value=[[0.1]])

    vector_index = AsyncMock()
    vector_index.search_chunks = AsyncMock(
        side_effect=[
            [_cand(chunk_id="c1", score=0.1, text="q")],
            [_cand(chunk_id="c2", score=0.2, text="h")],
        ]
    )

    hyde_generator = AsyncMock(return_value="hyde")
    synthesizer = AsyncMock(return_value=_synthesis_result(chunk_id="c1"))

    uc = AskUseCase(
        embedder=embedder,
        vector_index=vector_index,
        llm=AsyncMock(),
        hyde_generator=hyde_generator,
        synthesizer=synthesizer,
    )

    filters = schemas.SearchFilters(
        disease=schemas.Disease.thrombosis,
        year_min=2001,
        year_max=2024,
        source_type=schemas.SourceType.pubmed_abstract,
    )

    await uc.execute(
        request_id="r",
        question="q",
        filters=filters,
        embedding_model_id="e",
        generator_model_id="g",
        hyde_model_id="h",
        hyde_enabled=True,
        hyde_max_chars=100,
        ask_max_question_chars=None,
        ask_max_chunks_candidate=10,
        ask_max_chunks_final=5,
        ask_max_context_chars=1000,
        ask_llm_max_retries=0,
        debug_enabled=False,
    )

    assert vector_index.search_chunks.call_count == 2
    first = vector_index.search_chunks.call_args_list[0].kwargs
    second = vector_index.search_chunks.call_args_list[1].kwargs

    expected = {
        "disease": schemas.Disease.thrombosis.value,
        "source_type": schemas.SourceType.pubmed_abstract.value,
        "year_min": 2001,
        "year_max": 2024,
    }
    assert first["qfilter"] == expected
    assert second["qfilter"] == expected


@pytest.mark.anyio
async def test_empty_chunk_text_candidates_are_dropped_before_selection() -> None:
    embedder = AsyncMock()
    embedder.embed_texts = AsyncMock(return_value=[[0.1]])

    vector_index = AsyncMock()
    vector_index.search_chunks = AsyncMock(
        return_value=[
            _cand(chunk_id="c1", score=0.9, text=""),
            _cand(chunk_id="c2", score=0.8, text="ok"),
        ]
    )

    synthesizer = AsyncMock(return_value=_synthesis_result(chunk_id="c2"))

    uc = AskUseCase(
        embedder=embedder,
        vector_index=vector_index,
        llm=AsyncMock(),
        hyde_generator=AsyncMock(),
        synthesizer=synthesizer,
    )

    res = await uc.execute(
        request_id="r",
        question="q",
        filters=None,
        embedding_model_id="e",
        generator_model_id="g",
        hyde_model_id="h",
        hyde_enabled=False,
        hyde_max_chars=100,
        ask_max_question_chars=None,
        ask_max_chunks_candidate=10,
        ask_max_chunks_final=10,
        ask_max_context_chars=1000,
        ask_llm_max_retries=0,
        debug_enabled=True,
    )

    assert res.debug is not None

    all_ids = [c["chunk_id"] for c in res.debug["merged_candidates_all"]]
    usable_ids = [c["chunk_id"] for c in res.debug["merged_candidates_usable"]]

    assert "c1" in all_ids
    assert "c1" not in usable_ids
    assert "c2" in usable_ids


@dataclass(slots=True)
class _FakeLlm:
    output: str

    async def chat(
        self,
        *,
        model_id: str,
        messages: Sequence[Any],
        options: Mapping[str, Any] | None,
    ) -> str:  # type: ignore[override]
        return self.output


@pytest.mark.anyio
async def test_context_selection_limits_allowed_citations() -> None:
    embedder = AsyncMock()
    embedder.embed_texts = AsyncMock(return_value=[[0.1]])

    vector_index = AsyncMock()
    vector_index.search_chunks = AsyncMock(
        return_value=[
            _cand(chunk_id="c1", score=0.9, text="a" * 500),
            _cand(chunk_id="c2", score=0.8, text="b" * 500),
        ]
    )

    llm = _FakeLlm(
        output=(
            '{"answer":{"summary":"x","risk_factors":['
            '{"rank":1,"normalized_name":"n","aliases":["a"],"confidence":0.5,'
            '"rationale":"r","citations":["c2"]}],"limitations":[]},'
            '"citations":[{"chunk_id":"c2","doi":"10.1/x","title":"t","year":2020,'
            '"snippet":"s"}]}'
        )
    )

    uc = AskUseCase(
        embedder=embedder,
        vector_index=vector_index,
        llm=llm,  # type: ignore[arg-type]
        hyde_generator=AsyncMock(),
        synthesizer=synthesize_answer,
    )

    with pytest.raises(BusinessError) as exc:
        await uc.execute(
            request_id="r",
            question="q",
            filters=None,
            embedding_model_id="emb",
            generator_model_id="gen",
            hyde_model_id="hyde",
            hyde_enabled=False,
            hyde_max_chars=256,
            ask_max_question_chars=None,
            ask_max_chunks_candidate=30,
            ask_max_chunks_final=8,
            ask_max_context_chars=1,
            ask_llm_max_retries=0,
            debug_enabled=False,
        )

    assert exc.value.code == "validation_error"
    assert exc.value.message == "no_context_available"
