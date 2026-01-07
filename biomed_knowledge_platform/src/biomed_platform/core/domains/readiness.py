from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CheckStatus(StrEnum):
    ok = "ok"
    missing_config = "missing_config"
    unhealthy = "unhealthy"
    degraded = "degraded"
    error = "error"


class ReadinessStatus(StrEnum):
    ready = "ready"
    not_ready = "not_ready"


@dataclass(frozen=True, slots=True)
class ReadinessChecks:
    qdrant: CheckStatus
    llm: CheckStatus


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    status: ReadinessStatus
    checks: ReadinessChecks


@dataclass(frozen=True, slots=True)
class ReadinessHttpDecision:
    is_ready: bool
    log_message: str
    log_args: tuple