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


@then("the core OpenAPI title is BDD Testing - Biomedical Knowledge Platform")
def then_core_openapi_title(client):
    res = client.get("/openapi.json")
    assert res.status_code == 200
    body = res.json()
    assert body["info"]["title"] == "BDD Testing - Biomedical Knowledge Platform"
