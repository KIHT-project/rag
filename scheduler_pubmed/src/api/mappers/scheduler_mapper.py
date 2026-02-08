from __future__ import annotations

from scheduler_pubmed.src.api.models import schemas
from scheduler_pubmed.src.core.domains.scheduler import SchedulerRunCreated, SchedulerStatus


def to_api_scheduler_run_created(model: SchedulerRunCreated) -> schemas.SchedulerRunCreated:
    return schemas.SchedulerRunCreated(
        run_id=model.run_id,
        status=model.status.value,
        started_at=model.started_at,
    )


def to_api_scheduler_status(model: SchedulerStatus) -> schemas.SchedulerStatus:
    return schemas.SchedulerStatus(
        enabled=model.enabled,
        utc_schedule=model.utc_schedule,
        next_run_at=model.next_run_at,
        last_run_at=model.last_run_at,
        last_run_status=model.last_run_status.value if model.last_run_status else None,
    )
