from __future__ import annotations

from biomed_platform.core.domains.retrieval import ChunkCandidate, VectorSearchHit
from biomed_platform.core.services.retrieval.hybrid_ranker import (
    rerank_hybrid_candidates,
    rerank_vector_hits,
    section_weight,
)


def _cand(*, cid: str, score: float, text: str, section: str | None = None) -> ChunkCandidate:
    return ChunkCandidate(
        chunk_id=cid,
        doc_id="d",
        doi="10.1/x",
        title="t",
        year=2020,
        section=section,
        source_type=None,
        score=float(score),
        chunk_text=text,
    )


def test_section_weight_respects_boosts() -> None:
    assert section_weight("Results") > 1.0
    assert section_weight("Methods") < 1.0
    assert section_weight("Unknown") == 1.0
    assert section_weight(None) == 1.0


def test_rerank_vector_hits_uses_lexical_signal() -> None:
    hits = [
        VectorSearchHit(
            point_id="c1",
            score=0.9,
            payload={"text": "alpha", "section": "Introduction", "doc_id": "d1"},
        ),
        VectorSearchHit(
            point_id="c2",
            score=0.8,
            payload={"text": "alpha beta beta", "section": "Results", "doc_id": "d1"},
        ),
    ]

    ranked = rerank_vector_hits(
        query="alpha beta",
        hits=hits,
        weights={"dense": 0.1, "lexical": 1.0},
    )

    assert ranked[0].point_id == "c2"
    assert ranked[0].score > ranked[1].score


def test_rerank_hybrid_candidates_prefers_section_boost() -> None:
    question_candidates = [
        _cand(cid="c1", score=0.9, text="alpha beta", section="Methods"),
        _cand(cid="c2", score=0.8, text="alpha beta", section="Results"),
    ]

    hyde_candidates = [
        _cand(cid="c3", score=0.7, text="alpha", section="Discussion"),
    ]

    ranked = rerank_hybrid_candidates(
        question="alpha beta",
        question_candidates=question_candidates,
        hyde_candidates=hyde_candidates,
    )

    ids = [c.chunk_id for c in ranked]
    assert "c2" in ids
    assert ranked[0].chunk_id == "c2"
    assert ranked[0].score >= ranked[-1].score
