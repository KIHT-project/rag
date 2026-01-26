from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CheckStatus(str, Enum):
    ok = "ok"
    degraded = "degraded"
    unhealthy = "unhealthy"
    error = "error"
    missing_config = "missing_config"


class ReadinessStatus(str, Enum):
    ready = "ready"
    not_ready = "not_ready"


@dataclass(frozen=True)
class ReadinessChecks:
    qdrant: CheckStatus
    llm: CheckStatus
    rdbms: CheckStatus


@dataclass(frozen=True)
class ReadinessResult:
    status: ReadinessStatus
    checks: ReadinessChecks
    errors: dict[str, dict[str, object]] | None = None
