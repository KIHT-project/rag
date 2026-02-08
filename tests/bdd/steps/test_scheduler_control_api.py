from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pytest_bdd import given, scenario, then, when

from scheduler_pubmed.src.api.endpoints import scheduler as endpoint_mod
from scheduler_pubmed.src.core.domains.scheduler import (
    RunStatus,
    SchedulerRunCreated,
    SchedulerStatus,
)
from tests.bdd.helpers.scheduler_control_api import (
    get_scheduler_status,
    trigger_scheduler_run,
)


class _FakeSchedulerRuntime:
    async def trigger_manual_run(
        self, *, reldate_days: int | None = None
    ) -> SchedulerRunCreated:
        _ = reldate_days
        return SchedulerRunCreated(
            run_id=uuid4(),
            status=RunStatus.RUNNING,
            started_at=datetime.now(UTC),
        )

    async def get_status(self) -> SchedulerStatus:
        return SchedulerStatus(
            enabled=True,
            utc_schedule=["02:00", "08:00", "14:00"],
            next_run_at=datetime.now(UTC),
            last_run_at=None,
            last_run_status=None,
        )


@pytest.fixture(autouse=True)
def fake_scheduler_runtime(monkeypatch: pytest.MonkeyPatch, bdd_target: str) -> None:
    if bdd_target != "scheduler":
        pytest.skip("Scheduler control BDD runs only on scheduler target")
    runtime = _FakeSchedulerRuntime()
    monkeypatch.setattr(endpoint_mod, "_get_runtime", lambda _request: runtime)


@scenario(
    "../features/scheduler_control.feature",
    "Trigger scheduler run returns accepted response",
)
def test_trigger_scheduler_run_returns_accepted() -> None:
    pass


@scenario(
    "../features/scheduler_control.feature",
    "Trigger scheduler run with reldate override returns accepted response",
)
def test_trigger_scheduler_run_with_reldate_override_returns_accepted() -> None:
    pass


@scenario(
    "../features/scheduler_control.feature",
    "Scheduler status endpoint returns configured status",
)
def test_scheduler_status_endpoint_returns_configured_status() -> None:
    pass


@given("the scheduler control API is ready")
def given_scheduler_control_api_is_ready() -> None:
    return


@when("I trigger the scheduler run")
def when_i_trigger_scheduler_run(client, ctx: dict) -> None:
    ctx["res"] = trigger_scheduler_run(client)


@when("I trigger the scheduler run with 365 days lookback")
def when_i_trigger_scheduler_run_with_reldate_override(client, ctx: dict) -> None:
    ctx["res"] = trigger_scheduler_run(client, payload={"reldate_days": 365})


@when("I get scheduler control status")
def when_i_get_scheduler_control_status(client, ctx: dict) -> None:
    ctx["res"] = get_scheduler_status(client)


@then("the scheduler control response status is 202")
def then_scheduler_control_status_202(ctx: dict) -> None:
    assert ctx["res"].status_code == 202


@then("the scheduler control response status is 200")
def then_scheduler_control_status_200(ctx: dict) -> None:
    assert ctx["res"].status_code == 200


@then("the scheduler run status is RUNNING")
def then_scheduler_run_status_running(ctx: dict) -> None:
    assert ctx["res"].json()["status"] == "RUNNING"


@then("the scheduler run id exists")
def then_scheduler_run_id_exists(ctx: dict) -> None:
    run_id = ctx["res"].json().get("run_id")
    UUID(run_id)


@then("scheduler status contains utc schedule")
def then_scheduler_status_contains_utc_schedule(ctx: dict) -> None:
    assert ctx["res"].json()["utc_schedule"] == ["02:00", "08:00", "14:00"]


@then("scheduler status enabled flag is true")
def then_scheduler_status_enabled_flag_true(ctx: dict) -> None:
    assert ctx["res"].json()["enabled"] is True
