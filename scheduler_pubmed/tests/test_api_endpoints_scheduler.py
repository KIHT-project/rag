from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from scheduler_pubmed.src.api.endpoints import scheduler as endpoint_mod
from scheduler_pubmed.src.api.error_handlers import install_error_handlers
from scheduler_pubmed.src.core.domains.scheduler import (
    RunStatus,
    SchedulerRunCreated,
    SchedulerStatus,
)


class _FakeRuntime:
    def __init__(self) -> None:
        self.trigger_calls = 0
        self.last_reldate_days: int | None = None
        self.status_calls = 0
        self.run = SchedulerRunCreated(
            run_id=uuid4(),
            status=RunStatus.RUNNING,
            started_at=datetime.now(UTC),
        )
        self.status = SchedulerStatus(
            enabled=True,
            utc_schedule=["02:00", "14:00"],
            next_run_at=datetime.now(UTC),
            last_run_at=None,
            last_run_status=None,
        )

    async def trigger_manual_run(
        self, *, reldate_days: int | None = None
    ) -> SchedulerRunCreated:
        self.trigger_calls += 1
        self.last_reldate_days = reldate_days
        return self.run

    async def get_status(self) -> SchedulerStatus:
        self.status_calls += 1
        return self.status


@pytest.fixture
def scheduler_client(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(endpoint_mod.router)

    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(endpoint_mod, "_get_runtime", lambda _request: fake_runtime)

    with TestClient(app) as client:
        yield client, fake_runtime


def test_trigger_scheduler_run_returns_202(scheduler_client) -> None:
    client, runtime = scheduler_client

    response = client.post("/v1/pubmed/scheduler/run")

    assert response.status_code == 202
    assert response.json()["run_id"] == str(runtime.run.run_id)
    assert response.json()["status"] == "RUNNING"
    assert runtime.trigger_calls == 1
    assert runtime.last_reldate_days is None


def test_trigger_scheduler_run_accepts_reldate_override(scheduler_client) -> None:
    client, runtime = scheduler_client

    response = client.post("/v1/pubmed/scheduler/run", json={"reldate_days": 36500})

    assert response.status_code == 202
    assert runtime.trigger_calls == 1
    assert runtime.last_reldate_days == 36500


def test_get_scheduler_status_returns_200(scheduler_client) -> None:
    client, runtime = scheduler_client

    response = client.get("/v1/pubmed/scheduler/status")

    assert response.status_code == 200
    assert response.json()["enabled"] is True
    assert response.json()["utc_schedule"] == ["02:00", "14:00"]
    assert runtime.status_calls == 1


def test_scheduler_endpoints_have_scheduler_tag() -> None:
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(endpoint_mod.router)

    schema = app.openapi()
    assert schema["paths"]["/v1/pubmed/scheduler/run"]["post"]["tags"] == ["Scheduler"]
    assert schema["paths"]["/v1/pubmed/scheduler/status"]["get"]["tags"] == [
        "Scheduler"
    ]
