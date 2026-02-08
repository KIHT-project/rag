from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AppError(Exception):
    code: str
    message: str
    details: dict[str, object] | None = None
    retryable: bool = False


@dataclass(frozen=True)
class BusinessError(AppError):
    retryable: bool = False


@dataclass(frozen=True)
class SystemError(AppError):
    retryable: bool = True


def business_error(
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> BusinessError:
    return BusinessError(code=code, message=message, details=details)


def system_error(
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    retryable: bool = True,
) -> SystemError:
    return SystemError(code=code, message=message, details=details, retryable=retryable)
