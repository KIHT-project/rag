from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pytest_bdd import given, parsers, scenario, then, when

from scheduler_pubmed.src.api.endpoints import runs as endpoint_mod
from scheduler_pubmed.src.core.domains.scheduler import (
    DoiExecutionStatus,
    QueryExecution,
    RunDoiResult,
    RunStatus,
    SchedulerRun,
)
from scheduler_pubmed.src.core.errors.errors import business_error
from tests.bdd.helpers.scheduler_runs_api import (
    get_scheduler_run,
    list_run_dois,
    list_scheduler_runs,
)


class _FakeSchedulerRunsRuntime:
    def __init__(self) -> None:
        self.run_id = uuid4()
        self.run = SchedulerRun(
            run_id=self.run_id,
            status=RunStatus.SUCCESS,
            started_at=datetime(2026, 2, 8, 13, 0, tzinfo=UTC),
            finished_at=datetime(2026, 2, 8, 13, 5, tzinfo=UTC),
            queries=[
                QueryExecution(
                    query_execution_id=uuid4(),
                    query_id=uuid4(),
                    status=RunStatus.SUCCESS,
                    pubmed_result_count=1,
                    doi_resolved_count=1,
                    doi_skipped_exists_count=0,
                    doi_enqueued_count=1,
                    doi_failed_count=0,
                    ingest_job_id=None,
                )
            ],
        )
        self.failed_run = SchedulerRun(
            run_id=uuid4(),
            status=RunStatus.FAILED,
            started_at=datetime(2026, 2, 8, 15, 0, tzinfo=UTC),
            finished_at=datetime(2026, 2, 8, 15, 2, tzinfo=UTC),
            queries=[],
        )
        self.run_dois = [
            RunDoiResult(
                doi="10.1000/a",
                status=DoiExecutionStatus.INGESTED,
                error_message=None,
            )
        ]
        self._runs = [self.run, self.failed_run]

    async def list_runs(self, *, status, from_at, to_at):  # noqa: ANN001
        if from_at is not None and to_at is not None and from_at > to_at:
            raise business_error(
                code="validation_error",
                message="Invalid runs time range",
                details={
                    "from": from_at.isoformat(),
                    "to": to_at.isoformat(),
                },
            )

        if status is None and from_at is None and to_at is None:
            return [self.run]

        out = list(self._runs)
        if status is not None:
            out = [item for item in out if item.status == status]
        if from_at is not None:
            out = [item for item in out if item.started_at >= from_at]
        if to_at is not None:
            out = [item for item in out if item.started_at <= to_at]
        return out

    async def get_run(self, *, run_id):  # noqa: ANN001
        if run_id != self.run_id:
            raise business_error(
                code="not_found",
                message="Scheduler run not found",
                details={"run_id": str(run_id)},
            )
        return self.run

    async def list_run_dois(self, *, run_id):  # noqa: ANN001
        if run_id != self.run_id:
            raise business_error(
                code="not_found",
                message="Scheduler run not found",
                details={"run_id": str(run_id)},
            )
        return self.run_dois


@pytest.fixture(autouse=True)
def fake_scheduler_runs_runtime(
    monkeypatch: pytest.MonkeyPatch,
    bdd_target: str,
) -> _FakeSchedulerRunsRuntime:
    if bdd_target != "scheduler":
        pytest.skip("Scheduler runs BDD runs only on scheduler target")
    fake = _FakeSchedulerRunsRuntime()
    monkeypatch.setattr(endpoint_mod, "_get_runtime", lambda _request: fake)
    return fake


@scenario("../features/scheduler_runs.feature", "List scheduler runs")
def test_list_scheduler_runs() -> None:
    pass


@scenario("../features/scheduler_runs.feature", "Get scheduler run by id")
def test_get_scheduler_run_by_id() -> None:
    pass


@scenario("../features/scheduler_runs.feature", "List scheduler runs filtered by status")
def test_list_scheduler_runs_filtered_by_status() -> None:
    pass


@scenario("../features/scheduler_runs.feature", "List scheduler runs filtered by from-to range")
def test_list_scheduler_runs_filtered_by_from_to_range() -> None:
    pass


@scenario("../features/scheduler_runs.feature", "List scheduler runs with invalid from-to range")
def test_list_scheduler_runs_invalid_from_to_range() -> None:
    pass


@scenario("../features/scheduler_runs.feature", "List scheduler run doi results")
def test_list_scheduler_run_doi_results() -> None:
    pass


@scenario("../features/scheduler_runs.feature", "Missing scheduler run returns not found")
def test_missing_scheduler_run_returns_not_found() -> None:
    pass


@given("the scheduler runs API is ready")
def given_scheduler_runs_api_is_ready() -> None:
    return


@given("a scheduler run id is available")
def given_scheduler_run_id_is_available(
    ctx: dict,
    fake_scheduler_runs_runtime: _FakeSchedulerRunsRuntime,
) -> None:
    ctx["run_id"] = str(fake_scheduler_runs_runtime.run_id)


@when("I list scheduler runs")
def when_i_list_scheduler_runs(client, ctx: dict) -> None:
    ctx["res"] = list_scheduler_runs(client)


@when("I list scheduler runs with status FAILED")
def when_i_list_scheduler_runs_with_status_failed(client, ctx: dict) -> None:
    ctx["res"] = list_scheduler_runs(client, status="FAILED")


@when(parsers.parse('I list scheduler runs from "{from_at}" to "{to_at}"'))
def when_i_list_scheduler_runs_from_to(
    client,
    ctx: dict,
    from_at: str,
    to_at: str,
) -> None:
    ctx["from_at"] = from_at
    ctx["to_at"] = to_at
    ctx["res"] = list_scheduler_runs(client, from_at=from_at, to_at=to_at)


@when("I get the scheduler run by id")
def when_i_get_the_scheduler_run_by_id(client, ctx: dict) -> None:
    ctx["res"] = get_scheduler_run(client, run_id=ctx["run_id"])


@when("I list scheduler run dois")
def when_i_list_scheduler_run_dois(client, ctx: dict) -> None:
    ctx["res"] = list_run_dois(client, run_id=ctx["run_id"])


@when("I get a missing scheduler run")
def when_i_get_a_missing_scheduler_run(client, ctx: dict) -> None:
    ctx["res"] = get_scheduler_run(client, run_id=str(uuid4()))


@then("the scheduler runs response status is 200")
def then_scheduler_runs_response_status_200(ctx: dict) -> None:
    assert ctx["res"].status_code == 200


@then("the scheduler runs response status is 404")
def then_scheduler_runs_response_status_404(ctx: dict) -> None:
    assert ctx["res"].status_code == 404


@then("the scheduler runs response status is 400")
def then_scheduler_runs_response_status_400(ctx: dict) -> None:
    assert ctx["res"].status_code == 400


@then("scheduler runs list contains one run")
def then_scheduler_runs_list_contains_one_run(ctx: dict) -> None:
    body = ctx["res"].json()
    assert isinstance(body, list)
    assert len(body) == 1


@then("the scheduler run id matches the requested id")
def then_scheduler_run_id_matches_requested_id(ctx: dict) -> None:
    assert ctx["res"].json()["run_id"] == ctx["run_id"]


@then("scheduler run doi results contain ingested status")
def then_scheduler_run_doi_results_contain_ingested_status(ctx: dict) -> None:
    body = ctx["res"].json()
    assert isinstance(body, list)
    assert body[0]["status"] == "INGESTED"
    UUID(ctx["run_id"])


@then("all listed scheduler runs have status FAILED")
def then_all_listed_scheduler_runs_have_failed_status(ctx: dict) -> None:
    body = ctx["res"].json()
    assert isinstance(body, list)
    assert len(body) >= 1
    assert all(item["status"] == "FAILED" for item in body)


@then("scheduler runs are within the requested range")
def then_scheduler_runs_are_within_requested_range(ctx: dict) -> None:
    body = ctx["res"].json()
    assert isinstance(body, list)
    from_at = datetime.fromisoformat(ctx["from_at"].replace("Z", "+00:00"))
    to_at = datetime.fromisoformat(ctx["to_at"].replace("Z", "+00:00"))
    assert len(body) >= 1
    for item in body:
        started_at = datetime.fromisoformat(item["started_at"].replace("Z", "+00:00"))
        assert from_at <= started_at <= to_at


@then("scheduler runs error code is validation_error")
def then_scheduler_runs_error_code_validation_error(ctx: dict) -> None:
    assert ctx["res"].json()["error"] == "validation_error"
