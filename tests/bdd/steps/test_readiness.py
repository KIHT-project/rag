from __future__ import annotations

from pytest_bdd import given, scenario, then, when


@scenario("../features/readiness.feature", "Readiness check path")
def test_readiness_check_path():
    pass


@given("the API is running")
def given_api_running():
    return


@when("I call the readiness endpoint")
def when_call_ready(client, ctx):
    ctx["res"] = client.get("/ready")


@then("the response status is 200")
def then_status_200(ctx):
    assert ctx["res"].status_code == 200


@then("the readiness status is ready")
def then_ready(ctx):
    body = ctx["res"].json()
    assert body["status"] == "ready"


@then("qdrant is ok and llm is ok")
def then_checks_ok(ctx):
    body = ctx["res"].json()
    assert body["checks"]["qdrant"] == "ok"
    assert body["checks"]["llm"] == "ok"
