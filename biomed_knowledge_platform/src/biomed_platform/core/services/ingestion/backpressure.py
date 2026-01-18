from __future__ import annotations

from dataclasses import dataclass

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.ingestion import RetryAfterHint
from biomed_platform.core.ports.ingestion import BackpressurePolicy

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SimpleBackpressurePolicy(BackpressurePolicy):
    worker_count: int
    near_full_ratio: float = 0.9
    retry_after_low_seconds: int = 3
    retry_after_high_seconds: int = 10

    def retry_after(
        self, *, queue_depth: int, queue_max_size: int, worker_count: int
    ) -> RetryAfterHint:
        if queue_max_size <= 0:
            log.warning(
                "Backpressure fallback triggered due to invalid queue_max_size, "
                "queue_depth=%d, queue_max_size=%d",
                queue_depth,
                queue_max_size,
            )
            return RetryAfterHint(seconds=self.retry_after_high_seconds)

        threshold = int(queue_max_size * self.near_full_ratio)
        is_near_full = queue_depth >= threshold

        seconds = (
            self.retry_after_low_seconds if not is_near_full else self.retry_after_high_seconds
        )

        log.debug(
            "Backpressure decision computed, "
            "queue_depth=%d, "
            "queue_max_size=%d, "
            "threshold=%d, "
            "near_full=%s, "
            "retry_after_seconds=%d",
            queue_depth,
            queue_max_size,
            threshold,
            is_near_full,
            seconds,
        )

        return RetryAfterHint(seconds=seconds)
