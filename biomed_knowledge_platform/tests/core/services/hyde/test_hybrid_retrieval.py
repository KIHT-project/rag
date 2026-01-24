from __future__ import annotations

from biomed_platform.core.domains.hyde import RetrievalOrigin
from biomed_platform.core.domains.retrieval import ChunkCandidate
from biomed_platform.core.services.hyde.hybrid_retrieval import union_dedupe_order_candidates


def _c(
    *,
    chunk_id: str,
    score: float,
    doc_id: str = "d1",
    doi: str = "10.1/xyz",
    title: str | None = None,
    year: int | None = None,
    section: str | None = None,
    chunk_text: str | None = None,
) -> ChunkCandidate:
    return ChunkCandidate(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doi=doi,
        title=title,
        year=year,
        section=section,
        source_type=None,
        score=score,
        chunk_text=chunk_text,
    )


class TestHybridRetrievalUnionDedupeUnit:
    def test_given_disjoint_lists_when_union_then_returns_all_with_origin(self) -> None:
        # Given
        q = [_c(chunk_id="c1", score=0.9), _c(chunk_id="c2", score=0.5)]
        h = [_c(chunk_id="c3", score=0.7), _c(chunk_id="c4", score=0.2)]

        # When
        out = union_dedupe_order_candidates(question=q, hyde=h)

        # Then
        assert [c.chunk_id for c in out] == ["c1", "c3", "c2", "c4"]
        assert {c.chunk_id: c.origin for c in out} == {
            "c1": RetrievalOrigin.QUESTION,
            "c2": RetrievalOrigin.QUESTION,
            "c3": RetrievalOrigin.HYDE,
            "c4": RetrievalOrigin.HYDE,
        }

    def test_given_same_chunk_id_in_both_lists_when_dedupe_then_keeps_best_score_and_sets_both_origin(self) -> None:
        # Given
        q = [_c(chunk_id="c1", score=0.6, title="q")]
        h = [_c(chunk_id="c1", score=0.8, title="h")]

        # When
        out = union_dedupe_order_candidates(question=q, hyde=h)

        # Then
        assert len(out) == 1
        assert out[0].chunk_id == "c1"
        assert out[0].score == 0.8
        assert out[0].title == "h"
        assert out[0].origin is RetrievalOrigin.BOTH

    def test_given_duplicate_chunk_id_with_equal_score_when_dedupe_then_keeps_existing_fields_and_sets_both_origin(self) -> None:
        # Given
        q = [_c(chunk_id="c1", score=0.8, title="q")]
        h = [_c(chunk_id="c1", score=0.8, title="h")]

        # When
        out = union_dedupe_order_candidates(question=q, hyde=h)

        # Then
        assert len(out) == 1
        assert out[0].chunk_id == "c1"
        assert out[0].score == 0.8
        assert out[0].title == "q"
        assert out[0].origin is RetrievalOrigin.BOTH

    def test_given_ties_in_score_when_ordering_then_sorts_by_chunk_id_for_determinism(self) -> None:
        # Given
        q = [_c(chunk_id="c2", score=0.5), _c(chunk_id="c1", score=0.5)]
        h = []

        # When
        out = union_dedupe_order_candidates(question=q, hyde=h)

        # Then
        assert [c.chunk_id for c in out] == ["c1", "c2"]

    def test_given_same_inputs_different_input_order_when_union_then_output_is_deterministic(self) -> None:
        # Given
        q1 = [_c(chunk_id="c1", score=0.9), _c(chunk_id="c2", score=0.7)]
        h1 = [_c(chunk_id="c3", score=0.8)]

        q2 = list(reversed(q1))
        h2 = list(reversed(h1))

        # When
        out1 = union_dedupe_order_candidates(question=q1, hyde=h1)
        out2 = union_dedupe_order_candidates(question=q2, hyde=h2)

        # Then
        assert [(c.chunk_id, c.score, c.origin) for c in out1] == [
            ("c1", 0.9, RetrievalOrigin.QUESTION),
            ("c3", 0.8, RetrievalOrigin.HYDE),
            ("c2", 0.7, RetrievalOrigin.QUESTION),
        ]
        assert [(c.chunk_id, c.score, c.origin) for c in out2] == [
            ("c1", 0.9, RetrievalOrigin.QUESTION),
            ("c3", 0.8, RetrievalOrigin.HYDE),
            ("c2", 0.7, RetrievalOrigin.QUESTION),
        ]
