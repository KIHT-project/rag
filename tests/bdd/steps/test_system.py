from __future__ import annotations

from pytest_bdd import given, scenario, then, when


@scenario("../features/system.feature", "Health check path")
def test_health_check_path():
    pass


@given("the API is running")
def given_api_running():
    return


@when("I call the health endpoint")
def when_call_health(client, ctx):
    ctx["res"] = client.get("/health")


@then("the response status is 200")
def then_status_200(ctx):
    assert ctx["res"].status_code == 200


@then("the response body contains status ok")
def then_body_ok(ctx):
    assert ctx["res"].json() == {"status": "ok"}
