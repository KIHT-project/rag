from __future__ import annotations

from biomed_platform.core.services.ingestion.backpressure import SimpleBackpressurePolicy


def test_backpressure_retry_after_low_when_not_near_full() -> None:
    # Given a policy
    pol = SimpleBackpressurePolicy(worker_count=5, near_full_ratio=0.9, retry_after_low_seconds=3, retry_after_high_seconds=10)

    # When queue depth is below the near full threshold
    hint = pol.retry_after(queue_depth=7, queue_max_size=10, worker_count=5)

    # Then a low retry hint is returned
    assert hint.seconds == 3


def test_backpressure_retry_after_high_when_near_full() -> None:
    # Given a policy
    pol = SimpleBackpressurePolicy(worker_count=5, near_full_ratio=0.9, retry_after_low_seconds=3, retry_after_high_seconds=10)

    # When queue depth is at or above threshold
    hint = pol.retry_after(queue_depth=9, queue_max_size=10, worker_count=5)

    # Then a high retry hint is returned
    assert hint.seconds == 10


def test_backpressure_retry_after_high_when_max_size_invalid() -> None:
    # Given a policy
    pol = SimpleBackpressurePolicy(worker_count=5)

    # When queue max size is invalid
    hint = pol.retry_after(queue_depth=1, queue_max_size=0, worker_count=5)

    # Then fallback uses high value
    assert hint.seconds == pol.retry_after_high_seconds
