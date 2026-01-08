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


@dataclass(frozen=True)
class DependencyError(SystemError):
    dependency: str = ""


def dependency_connection_failed(
    *,
    dependency: str,
    base_url: str,
    reason: str,
    code: str | None = None,
    message: str | None = None,
    retryable: bool = True,
    extra_details: dict[str, Any] | None = None,
) -> DependencyError:
    details: dict[str, Any] = {"base_url": base_url, "reason": reason}
    if extra_details:
        details.update(extra_details)

    resolved_code = code or f"{dependency}_connection_failed"
    resolved_message = message or f"Connection with {dependency} failed"

    return DependencyError(
        dependency=dependency,
        code=resolved_code,
        message=resolved_message,
        details=details,
        retryable=retryable,
    )


def business_error(
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> BusinessError:
    return BusinessError(code=code, message=message, details=details)


def duplicate_doi_error(*, doi_normalized: str, embedding_model_id: str) -> BusinessError:
    return business_error(
        code="duplicate_doi",
        message="Duplicate DOI not allowed",
        details={
            "doi_normalized": doi_normalized,
            "embedding_model_id": embedding_model_id,
        },
    )


def idempotency_conflict_error(*, idempotency_key: str) -> BusinessError:
    return business_error(
        code="validation_error",
        message="Idempotency Key reused with different payload",
        details={"idempotency_key": idempotency_key},
    )


def job_not_found_error(*, job_id: str) -> BusinessError:
    return business_error(
        code="not_found",
        message="Job not found",
        details={"job_id": job_id},
    )


def queue_full_error(*, queue_max_size: int, retry_after_seconds: int) -> BusinessError:
    return business_error(
        code="too_many_requests",
        message="Too many requests",
        details={
            "queue_max_size": queue_max_size,
            "retry_after_seconds": retry_after_seconds,
        },
    )
