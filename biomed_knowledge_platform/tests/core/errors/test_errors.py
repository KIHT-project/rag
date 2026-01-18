from __future__ import annotations

from biomed_platform.core.errors import errors as err


def test_error_dataclasses_have_expected_defaults() -> None:
    # Given AppError and derived error types
    app = err.AppError(code="c", message="m")
    biz = err.BusinessError(code="c", message="m")
    sys = err.SystemError(code="c", message="m")
    dep = err.DependencyError(code="c", message="m", dependency="dep")

    # When reading retryable defaults
    # Then defaults match class intent
    assert app.retryable is False
    assert biz.retryable is False
    assert sys.retryable is True
    assert dep.retryable is True
    assert dep.dependency == "dep"


def test_dependency_connection_failed_builds_expected_error() -> None:
    # Given dependency parameters
    # When building a dependency error
    exc = err.dependency_connection_failed(
        dependency="qdrant",
        base_url="http://localhost:6333",
        reason="timeout",
        extra_details={"k": "v"},
    )

    # Then the error is typed and carries merged details
    assert isinstance(exc, err.DependencyError)
    assert exc.code == "qdrant_connection_failed"
    assert "Connection with" in exc.message
    assert exc.details == {"base_url": "http://localhost:6333", "reason": "timeout", "k": "v"}


def test_business_error_helpers_use_expected_codes_and_details() -> None:
    # Given helper functions
    # When building typed errors
    dup = err.duplicate_doi_error(doi_normalized="10.1/abc", embedding_model_id="m")
    idem = err.idempotency_conflict_error(idempotency_key="k")
    nf = err.job_not_found_error(job_id="j")
    qf = err.queue_full_error(queue_max_size=3, retry_after_seconds=7)
    nvi = err.no_valid_items_error(embedding_model_id="m")

    # Then codes and details are stable
    assert dup.code == "duplicate_doi"
    assert dup.details == {"doi_normalized": "10.1/abc", "embedding_model_id": "m"}

    assert idem.code == "validation_error"
    assert idem.details == {"idempotency_key": "k"}

    assert nf.code == "not_found"
    assert nf.details == {"job_id": "j"}

    assert qf.code == "too_many_requests"
    assert qf.details == {"queue_max_size": 3, "retry_after_seconds": 7}

    assert nvi.code == "validation_error"
    assert nvi.details == {"embedding_model_id": "m"}
