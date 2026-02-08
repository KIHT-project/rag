from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from scheduler_pubmed.src.api.mappers.scheduler_mapper import (
    to_api_scheduler_run_created,
    to_api_scheduler_status,
)
from scheduler_pubmed.src.core.domains.scheduler import RunStatus, SchedulerRunCreated, SchedulerStatus


def test_to_api_scheduler_run_created_maps_fields() -> None:
    model = SchedulerRunCreated(
        run_id=uuid4(),
        status=RunStatus.RUNNING,
        started_at=datetime.now(UTC),
    )

    response = to_api_scheduler_run_created(model)

    assert response.run_id == model.run_id
    assert response.status == "RUNNING"
    assert response.started_at == model.started_at


def test_to_api_scheduler_status_maps_fields() -> None:
    model = SchedulerStatus(
        enabled=True,
        utc_schedule=["02:00", "14:00"],
        next_run_at=datetime.now(UTC),
        last_run_at=datetime.now(UTC),
        last_run_status=RunStatus.SUCCESS,
    )

    response = to_api_scheduler_status(model)

    assert response.enabled is True
    assert response.utc_schedule == ["02:00", "14:00"]
    assert response.last_run_status == "SUCCESS"
