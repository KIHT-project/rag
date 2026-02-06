from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RiskFactor:
    rank: int
    normalized_name: str
    aliases: list[str]
    confidence: float
    rationale: str
    citations: list[str]


@dataclass(frozen=True, slots=True)
class AnswerPayload:
    summary: str
    risk_factors: list[RiskFactor]
    limitations: list[str]


@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: str
    doi: str
    title: str | None
    year: int | None
    section: str | None
    snippet: str


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    answer: AnswerPayload
    citations: list[Citation]


@dataclass(frozen=True, slots=True)
class SynthesisOutput:
    answer: AnswerPayload
    citations: list[Citation]


@dataclass(frozen=True, slots=True)
class AskResponseEnvelope:
    request_id: str
    effective_hyde_enabled: bool
    answer: AnswerPayload
    citations: list[Citation]
    debug: dict[str, Any] | None = None
