from __future__ import annotations

from biomed_platform.api.models.generated import schemas
from biomed_platform.api.mappers.retrieval_mapper import to_domain_search_filters
from biomed_platform.core.domains.retrieval import SearchFilters
from biomed_platform.core.domains.synthesis import (
    AnswerPayload,
    AskResponseEnvelope,
    Citation,
    RiskFactor,
)


def to_domain_ask_filters(filters: schemas.SearchFilters | None) -> SearchFilters | None:
    return to_domain_search_filters(filters)


def _to_api_risk_factor(item: RiskFactor) -> schemas.RiskFactor:
    return schemas.RiskFactor(
        rank=int(item.rank),
        normalized_name=item.normalized_name,
        aliases=[schemas.Alias(root=a) for a in (item.aliases or [])],
        confidence=float(item.confidence),
        rationale=item.rationale,
        citations=list(item.citations or []),
    )


def _to_api_answer_payload(answer: AnswerPayload) -> schemas.AnswerPayload:
    return schemas.AnswerPayload(
        summary=answer.summary,
        risk_factors=[_to_api_risk_factor(rf) for rf in answer.risk_factors],
        limitations=[schemas.Limitation(root=v) for v in answer.limitations],
    )


def _to_api_citation(citation: Citation) -> schemas.Citation:
    return schemas.Citation(
        chunk_id=citation.chunk_id,
        doi=citation.doi,
        title=citation.title,
        year=citation.year,
        section=citation.section,
        snippet=citation.snippet,
    )


def to_api_ask_response(response: AskResponseEnvelope) -> schemas.AskResponseEnvelope:
    return schemas.AskResponseEnvelope(
        request_id=response.request_id,
        effective_hyde_enabled=response.effective_hyde_enabled,
        answer=_to_api_answer_payload(response.answer),
        citations=[_to_api_citation(c) for c in response.citations],
        debug=response.debug,
    )
