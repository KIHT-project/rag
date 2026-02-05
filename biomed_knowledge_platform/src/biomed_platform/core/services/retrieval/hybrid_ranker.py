from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.hyde import HybridChunkCandidate
from biomed_platform.core.domains.retrieval import ChunkCandidate, VectorSearchHit
from biomed_platform.core.services.hyde.hybrid_retrieval import union_dedupe_order_candidates

log = get_logger(__name__)

DEFAULT_RRF_K = 60
DEFAULT_RRF_WEIGHTS: Mapping[str, float] = {
    "dense": 1.0,
    "hyde": 0.6,
    "lexical": 1.0,
}

SECTION_BOOSTS: Mapping[str, float] = {
    "results": 1.35,
    "discussion": 1.2,
    "conclusion": 1.3,
    "conclusions": 1.3,
    "abstract": 1.1,
    "introduction": 1.0,
    "background": 1.0,
    "methods": 0.85,
    "materials_and_methods": 0.85,
    "methodology": 0.85,
    "supplementary": 0.75,
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _normalize_section(section: str | None) -> str | None:
    if not section:
        return None
    raw = section.strip().lower()
    if not raw:
        return None
    cleaned = re.sub(r"[^a-z0-9 ]", " ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.replace(" ", "_") if cleaned else None


def section_weight(section: str | None, *, boosts: Mapping[str, float] | None = None) -> float:
    normalized = _normalize_section(section)
    if not normalized:
        return 1.0
    lookup = boosts or SECTION_BOOSTS
    return float(lookup.get(normalized, 1.0))


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= 2]


def _bm25_scores(
    *,
    query_tokens: Sequence[str],
    docs: Mapping[str, Sequence[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    if not query_tokens or not docs:
        return {}

    doc_lens = {cid: len(tokens) for cid, tokens in docs.items()}
    avgdl = sum(doc_lens.values()) / max(1, len(doc_lens))

    df: dict[str, int] = {}
    unique_query = set(query_tokens)
    for term in unique_query:
        df[term] = sum(1 for tokens in docs.values() if term in tokens)

    scores: dict[str, float] = {}
    for cid, tokens in docs.items():
        if not tokens:
            continue
        tf: dict[str, int] = {}
        for t in tokens:
            if t in unique_query:
                tf[t] = tf.get(t, 0) + 1

        score = 0.0
        dl = doc_lens[cid]
        for term in unique_query:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            denom = freq + k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
            idf = math.log((len(docs) - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1.0)
            score += idf * (freq * (k1 + 1.0) / denom)

        if score > 0:
            scores[cid] = score

    return scores


def _rank_map(ids: Iterable[str]) -> dict[str, int]:
    return {cid: idx + 1 for idx, cid in enumerate(ids) if cid}


def _rrf_score(*, ranks: Mapping[str, int], k: int, weights: Mapping[str, float]) -> float:
    score = 0.0
    for name, rank in ranks.items():
        if rank <= 0:
            continue
        weight = float(weights.get(name, 1.0))
        score += weight / (k + rank)
    return score


def _lexical_rank(
    *, query: str, texts: Mapping[str, str]
) -> tuple[dict[str, float], dict[str, int]]:
    tokens = _tokenize(query)
    docs = {cid: _tokenize(text) for cid, text in texts.items() if text}
    scores = _bm25_scores(query_tokens=tokens, docs=docs)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return scores, _rank_map([cid for cid, _ in ranked])


@dataclass(slots=True)
class RankedHybrid:
    candidate: HybridChunkCandidate
    score: float


def rerank_hybrid_candidates(
    *,
    question: str,
    question_candidates: Sequence[ChunkCandidate],
    hyde_candidates: Sequence[ChunkCandidate],
    rrf_k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
    section_boosts: Mapping[str, float] | None = None,
) -> list[HybridChunkCandidate]:
    merged = union_dedupe_order_candidates(question=question_candidates, hyde=hyde_candidates)
    if not merged:
        return []

    weights_final = weights or DEFAULT_RRF_WEIGHTS

    dense_sorted = sorted(question_candidates, key=lambda c: (-float(c.score), c.chunk_id))
    hyde_sorted = sorted(hyde_candidates, key=lambda c: (-float(c.score), c.chunk_id))
    dense_rank = _rank_map([c.chunk_id for c in dense_sorted])
    hyde_rank = _rank_map([c.chunk_id for c in hyde_sorted])

    text_map = {c.chunk_id: (c.chunk_text or "") for c in merged if c.chunk_id}
    _, lexical_rank = _lexical_rank(query=question, texts=text_map)

    ranked: list[HybridChunkCandidate] = []
    for cand in merged:
        ranks: dict[str, int] = {}
        if cand.chunk_id in dense_rank:
            ranks["dense"] = dense_rank[cand.chunk_id]
        if cand.chunk_id in hyde_rank:
            ranks["hyde"] = hyde_rank[cand.chunk_id]
        if cand.chunk_id in lexical_rank:
            ranks["lexical"] = lexical_rank[cand.chunk_id]

        rrf_score = _rrf_score(ranks=ranks, k=int(rrf_k), weights=weights_final)
        boosted = rrf_score * section_weight(cand.section, boosts=section_boosts)
        ranked.append(
            HybridChunkCandidate(
                chunk_id=cand.chunk_id,
                doc_id=cand.doc_id,
                doi=cand.doi,
                title=cand.title,
                year=cand.year,
                section=cand.section,
                source_type=cand.source_type,
                score=boosted,
                chunk_text=cand.chunk_text,
                origin=cand.origin,
            )
        )

    ranked.sort(key=lambda c: (-float(c.score), c.chunk_id))
    return ranked


def rerank_vector_hits(
    *,
    query: str,
    hits: Sequence[VectorSearchHit],
    rrf_k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
    section_boosts: Mapping[str, float] | None = None,
) -> list[VectorSearchHit]:
    if not hits:
        return []

    weights_final = weights or DEFAULT_RRF_WEIGHTS
    dense_sorted = sorted(hits, key=lambda h: (-float(h.score), h.point_id))
    dense_rank = _rank_map([h.point_id for h in dense_sorted])

    texts = {h.point_id: str((h.payload or {}).get("text") or "") for h in hits if h.point_id}
    _, lexical_rank = _lexical_rank(query=query, texts=texts)

    out: list[VectorSearchHit] = []
    for h in hits:
        payload = dict(h.payload or {})
        section = payload.get("section")
        section_str = str(section) if isinstance(section, str) else None

        ranks: dict[str, int] = {}
        if h.point_id in dense_rank:
            ranks["dense"] = dense_rank[h.point_id]
        if h.point_id in lexical_rank:
            ranks["lexical"] = lexical_rank[h.point_id]

        rrf_score = _rrf_score(ranks=ranks, k=int(rrf_k), weights=weights_final)
        boosted = rrf_score * section_weight(section_str, boosts=section_boosts)

        out.append(
            VectorSearchHit(
                point_id=h.point_id,
                score=float(boosted),
                payload=payload,
            )
        )

    out.sort(key=lambda h: (-float(h.score), h.point_id))
    log.info(
        "Hybrid rerank applied | hits=%s | query_len=%s",
        len(out),
        len(query or ""),
    )
    return out
