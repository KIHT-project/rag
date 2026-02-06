from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pytest_bdd import given, scenario, then, when

from biomed_platform.api.error_handlers import install_error_handlers


@scenario("../features/error_handling.feature", "Request validation errors are normalized")
def test_request_validation_errors_are_normalized():
    pass


@scenario("../features/error_handling.feature", "Unexpected exceptions are normalized")
def test_unexpected_exceptions_are_normalized():
    pass


@scenario("../features/error_handling.feature", "Not found routes are normalized")
def test_not_found_routes_are_normalized():
    pass


@given("a local test app with error handlers")
def given_local_test_app_with_error_handlers(ctx):
    app = FastAPI()
    install_error_handlers(app)
    ctx["local_app"] = app
    ctx["local_client"] = TestClient(app, raise_server_exceptions=False)


@given("an endpoint with strict payload schema")
def given_endpoint_with_strict_payload_schema(ctx):
    app: FastAPI = ctx["local_app"]

    class _Payload(BaseModel):
        count: int

    @app.post("/strict")
    async def strict(payload: _Payload):
        return {"count": payload.count}


@given("an endpoint that raises an unexpected exception")
def given_endpoint_raises_unexpected_exception(ctx):
    app: FastAPI = ctx["local_app"]

    @app.get("/explode")
    async def explode():
        raise ValueError("boom")


@when("I POST invalid payload to the strict endpoint")
def when_post_invalid_payload(ctx):
    client: TestClient = ctx["local_client"]
    ctx["res"] = client.post("/strict", json={"count": "x"})


@when("I call the exploding endpoint")
def when_call_exploding_endpoint(ctx):
    client: TestClient = ctx["local_client"]
    ctx["res"] = client.get("/explode")


@when("I call a missing route on the local app")
def when_call_missing_route(ctx):
    client: TestClient = ctx["local_client"]
    ctx["res"] = client.get("/missing-route")


@then("the response status is 422")
def then_status_422(ctx):
    assert ctx["res"].status_code == 422


@then("the response status is 500")
def then_status_500(ctx):
    assert ctx["res"].status_code == 500


@then("the response status is 404")
def then_status_404(ctx):
    assert ctx["res"].status_code == 404


@then("the error code is validation_error")
def then_error_validation_error(ctx):
    assert ctx["res"].json().get("error") == "validation_error"


@then("the error code is system_error")
def then_error_system_error(ctx):
    assert ctx["res"].json().get("error") == "system_error"


@then("the error code is not_found")
def then_error_not_found(ctx):
    assert ctx["res"].json().get("error") == "not_found"


@then("request id is present or none")
def then_request_id_present_or_none(ctx):
    rid = ctx["res"].json().get("request_id")
    assert isinstance(rid, str) and rid
