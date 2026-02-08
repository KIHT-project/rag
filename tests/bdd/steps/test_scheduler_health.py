from __future__ import annotations

from pytest_bdd import given, scenario, then, when


@scenario("../features/scheduler_health.feature", "Scheduler health endpoint responds ok")
def test_scheduler_health_endpoint() -> None:
    pass


@scenario("../features/scheduler_health.feature", "Scheduler live endpoint responds alive")
def test_scheduler_live_endpoint() -> None:
    pass


@scenario("../features/scheduler_health.feature", "Scheduler ready endpoint responds ready")
def test_scheduler_ready_endpoint() -> None:
    pass


@given("the scheduler API is running")
def given_scheduler_api_running() -> None:
    return


@when("I call the scheduler health endpoint")
def when_call_scheduler_health(client, ctx: dict) -> None:
    ctx["res"] = client.get("/health")


@when("I call the scheduler live endpoint")
def when_call_scheduler_live(client, ctx: dict) -> None:
    ctx["res"] = client.get("/health/live")


@when("I call the scheduler ready endpoint")
def when_call_scheduler_ready(client, ctx: dict) -> None:
    ctx["res"] = client.get("/health/ready")


@then("the scheduler response status is 200")
def then_scheduler_status_200(ctx: dict) -> None:
    assert ctx["res"].status_code == 200


@then("the scheduler health status is ok")
def then_scheduler_health_ok(ctx: dict) -> None:
    assert ctx["res"].json() == {"status": "ok"}


@then("the scheduler live status is alive")
def then_scheduler_live_alive(ctx: dict) -> None:
    assert ctx["res"].json() == {"status": "alive"}


@then("the scheduler ready status is ready")
def then_scheduler_ready_ready(ctx: dict) -> None:
    assert ctx["res"].json() == {"status": "ready"}


@then("the scheduler OpenAPI title is BDD Testing - PubMed Scheduler")
def then_scheduler_openapi_title(client) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    body = response.json()
    assert body["info"]["title"] == "BDD Testing - PubMed Scheduler"
