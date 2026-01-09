from __future__ import annotations

import pytest

from biomed_platform.core.domains.ingestion import RetryAfterHint
from biomed_platform.core.services.ingestion import SimpleBackpressurePolicy


class TestSimpleBackpressurePolicy:
    def test_retry_after_returns_high_when_queue_max_size_non_positive(self) -> None:
        policy = SimpleBackpressurePolicy(
            worker_count=4,
            near_full_ratio=0.9,
            retry_after_low_seconds=3,
            retry_after_high_seconds=10,
        )

        got = policy.retry_after(queue_depth=0, queue_max_size=0, worker_count=4)

        assert got == RetryAfterHint(seconds=10)

    @pytest.mark.parametrize("queue_max_size", [-1, -100])
    def test_retry_after_returns_high_when_queue_max_size_negative(self, queue_max_size: int) -> None:
        policy = SimpleBackpressurePolicy(worker_count=2)

        got = policy.retry_after(queue_depth=0, queue_max_size=queue_max_size, worker_count=2)

        assert got == RetryAfterHint(seconds=policy.retry_after_high_seconds)

    def test_retry_after_returns_low_when_below_threshold(self) -> None:
        policy = SimpleBackpressurePolicy(
            worker_count=4,
            near_full_ratio=0.9,
            retry_after_low_seconds=3,
            retry_after_high_seconds=10,
        )

        queue_max_size = 100
        threshold = int(queue_max_size * policy.near_full_ratio)
        assert threshold == 90  # sanity check for this test case

        got = policy.retry_after(queue_depth=threshold - 1, queue_max_size=queue_max_size, worker_count=4)

        assert got == RetryAfterHint(seconds=policy.retry_after_low_seconds)

    def test_retry_after_returns_high_when_at_threshold(self) -> None:
        policy = SimpleBackpressurePolicy(worker_count=4, near_full_ratio=0.9)

        queue_max_size = 100
        threshold = int(queue_max_size * policy.near_full_ratio)
        assert threshold == 90

        got = policy.retry_after(queue_depth=threshold, queue_max_size=queue_max_size, worker_count=4)

        assert got == RetryAfterHint(seconds=policy.retry_after_high_seconds)

    def test_retry_after_returns_high_when_above_threshold(self) -> None:
        policy = SimpleBackpressurePolicy(worker_count=4, near_full_ratio=0.9)

        queue_max_size = 100
        threshold = int(queue_max_size * policy.near_full_ratio)
        assert threshold == 90

        got = policy.retry_after(queue_depth=threshold + 1, queue_max_size=queue_max_size, worker_count=4)

        assert got == RetryAfterHint(seconds=policy.retry_after_high_seconds)

    def test_retry_after_ignores_worker_count_parameter(self) -> None:
        policy = SimpleBackpressurePolicy(worker_count=4, near_full_ratio=0.9)

        got_a = policy.retry_after(queue_depth=10, queue_max_size=100, worker_count=1)
        got_b = policy.retry_after(queue_depth=10, queue_max_size=100, worker_count=999)

        assert got_a == got_b

    def test_retry_after_respects_custom_retry_after_seconds(self) -> None:
        policy = SimpleBackpressurePolicy(
            worker_count=4,
            near_full_ratio=0.9,
            retry_after_low_seconds=1,
            retry_after_high_seconds=42,
        )

        # below threshold
        got_low = policy.retry_after(queue_depth=0, queue_max_size=100, worker_count=4)
        assert got_low == RetryAfterHint(seconds=1)

        # at or above threshold
        got_high = policy.retry_after(queue_depth=90, queue_max_size=100, worker_count=4)
        assert got_high == RetryAfterHint(seconds=42)

    def test_retry_after_threshold_rounding_int_truncation_behavior(self) -> None:
        policy = SimpleBackpressurePolicy(worker_count=1, near_full_ratio=0.9)

        queue_max_size = 11
        threshold = int(queue_max_size * policy.near_full_ratio)  # int(9.9) == 9
        assert threshold == 9

        got_low = policy.retry_after(queue_depth=8, queue_max_size=queue_max_size, worker_count=1)
        got_high = policy.retry_after(queue_depth=9, queue_max_size=queue_max_size, worker_count=1)

        assert got_low == RetryAfterHint(seconds=policy.retry_after_low_seconds)
        assert got_high == RetryAfterHint(seconds=policy.retry_after_high_seconds)
