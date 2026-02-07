from __future__ import annotations

import json

from pytest_bdd import given, scenario, then, when

from tests.bdd.helpers.ask_api import post_ask
from tests.bdd.helpers.postgres_audit import fetch_one, fetch_rows


@scenario("../features/audit.feature", "Successful API request is audited in postgres")
def test_successful_api_request_is_audited_in_postgres():
    pass


@scenario("../features/audit.feature", "Validation error is audited with stacktrace")
def test_validation_error_is_audited_with_stacktrace():
    pass


@scenario("../features/audit.feature", "Duplicate request ids are audited as separate rows")
def test_duplicate_request_ids_are_audited_as_separate_rows():
    pass


@scenario("../features/audit.feature", "Not found route is audited as error")
def test_not_found_route_is_audited_as_error():
    pass


@scenario("../features/audit.feature", "System error path is audited with stacktrace")
def test_system_error_path_is_audited_with_stacktrace():
    pass


@scenario("../features/audit.feature", "Search request stores request and response raw payloads")
def test_search_request_stores_request_and_response_raw_payloads():
    pass


@scenario("../features/audit.feature", "Duplicate request id across endpoints is tracked separately")
def test_duplicate_request_id_across_endpoints_is_tracked_separately():
    pass


@given("audit ask request is invalid")
def given_ask_request_is_invalid(ctx):
    ctx["ask_payload"] = {"filters": None}


@when("I GET health with request id audit-success-1")
def when_get_health(client, ctx):
    ctx["request_id"] = "audit-success-1"
    ctx["res"] = client.get("/health", headers={"X-Request-Id": ctx["request_id"]})


@when("I POST ask with request id audit-validation-1")
def when_post_invalid_ask(client, ctx):
    ctx["request_id"] = "audit-validation-1"
    payload = ctx["ask_payload"]
    ctx["res"] = post_ask(client, payload=payload, request_id=ctx["request_id"])


@when("I GET health twice with request id audit-dup-1")
def when_get_health_twice(client, ctx):
    ctx["request_id"] = "audit-dup-1"
    headers = {"X-Request-Id": ctx["request_id"]}
    r1 = client.get("/health", headers=headers)
    r2 = client.get("/health", headers=headers)
    assert r1.status_code == 200
    assert r2.status_code == 200


@when("I GET missing route with request id audit-404-1")
def when_get_missing_route(client, ctx):
    ctx["request_id"] = "audit-404-1"
    ctx["res"] = client.get("/missing-audit-path", headers={"X-Request-Id": ctx["request_id"]})


@when("I POST search with broken service and request id audit-500-1")
def when_post_search_with_broken_service(client, ctx):
    ctx["request_id"] = "audit-500-1"
    app = client.app
    prev = getattr(app.state, "search_use_case", None)
    app.state.search_use_case = None
    try:
        ctx["res"] = client.post(
            "/v1/search",
            json={"query": "thrombosis risk factors", "filters": None},
            headers={"X-Request-Id": ctx["request_id"]},
        )
    finally:
        app.state.search_use_case = prev


@when("I POST search with request id audit-search-200-1")
def when_post_search_success(client, ctx):
    ctx["request_id"] = "audit-search-200-1"
    ctx["search_payload"] = {"query": "thrombosis risk factors", "filters": None}
    ctx["res"] = client.post(
        "/v1/search",
        json=ctx["search_payload"],
        headers={"X-Request-Id": ctx["request_id"]},
    )


@when("I call health then missing route with request id audit-dup-cross-1")
def when_call_cross_endpoint_duplicate(client):
    rid = "audit-dup-cross-1"
    r1 = client.get("/health", headers={"X-Request-Id": rid})
    r2 = client.get("/missing-audit-path", headers={"X-Request-Id": rid})
    assert r1.status_code == 200
    assert r2.status_code == 404


@then("audit scenario response status is 200")
def then_status_200(ctx):
    assert ctx["res"].status_code == 200


@then("audit scenario response status is 422")
def then_status_422(ctx):
    assert ctx["res"].status_code == 422


@then("audit scenario response status is 404")
def then_status_404(ctx):
    assert ctx["res"].status_code == 404


@then("audit scenario response status is 500")
def then_status_500(ctx):
    assert ctx["res"].status_code == 500


@then("audit request record exists for audit-success-1 with status SUCCESS")
def then_audit_success():
    row = fetch_one(
        sql=(
            "SELECT request_id, status, response_status_code, response_headers_raw "
            "FROM audit_request WHERE request_id = $1 ORDER BY id DESC LIMIT 1"
        ),
        args=("audit-success-1",),
    )
    assert row["request_id"] == "audit-success-1"
    assert row["status"] == "SUCCESS"
    assert row["response_status_code"] == 200
    assert isinstance(row["response_headers_raw"], str) and row["response_headers_raw"]


@then("audit request record exists for audit-validation-1 with status ERROR")
def then_audit_error_request():
    row = fetch_one(
        sql=(
            "SELECT request_id, status, response_status_code FROM audit_request "
            "WHERE request_id = $1 ORDER BY id DESC LIMIT 1"
        ),
        args=("audit-validation-1",),
    )
    assert row["request_id"] == "audit-validation-1"
    assert row["status"] == "ERROR"
    assert row["response_status_code"] == 422


@then("audit request record exists for audit-404-1 with status ERROR")
def then_audit_404_request():
    row = fetch_one(
        sql=(
            "SELECT request_id, status, response_status_code FROM audit_request "
            "WHERE request_id = $1 ORDER BY id DESC LIMIT 1"
        ),
        args=("audit-404-1",),
    )
    assert row["request_id"] == "audit-404-1"
    assert row["status"] == "ERROR"
    assert row["response_status_code"] == 404


@then("audit request record exists for audit-500-1 with status ERROR")
def then_audit_500_request():
    row = fetch_one(
        sql=(
            "SELECT request_id, status, response_status_code FROM audit_request "
            "WHERE request_id = $1 ORDER BY id DESC LIMIT 1"
        ),
        args=("audit-500-1",),
    )
    assert row["request_id"] == "audit-500-1"
    assert row["status"] == "ERROR"
    assert row["response_status_code"] == 500


@then("audit events contain API_REQUEST_RECEIVED and API_RESPONSE_SENT for audit-success-1")
def then_audit_events_exist():
    rows = fetch_rows(
        sql=(
            "SELECT event_type FROM audit_event WHERE request_id = $1 "
            "ORDER BY created_at ASC"
        ),
        args=("audit-success-1",),
    )
    event_types = {str(r["event_type"]) for r in rows}
    assert "API_REQUEST_RECEIVED" in event_types
    assert "API_RESPONSE_SENT" in event_types


@then("audit error exists for audit-validation-1 with exception class RequestValidationError")
def then_audit_error_exists():
    row = fetch_one(
        sql=(
            "SELECT exception_class, stacktrace_raw FROM audit_error "
            "WHERE request_id = $1 ORDER BY created_at DESC LIMIT 1"
        ),
        args=("audit-validation-1",),
    )
    assert row["exception_class"] == "RequestValidationError"
    assert isinstance(row["stacktrace_raw"], str) and row["stacktrace_raw"].strip()


@then("audit error exists for audit-404-1 with exception class HTTPException")
def then_audit_404_error_exists():
    row = fetch_one(
        sql=(
            "SELECT exception_class, stacktrace_raw FROM audit_error "
            "WHERE request_id = $1 ORDER BY created_at DESC LIMIT 1"
        ),
        args=("audit-404-1",),
    )
    assert row["exception_class"] == "HTTPException"
    assert isinstance(row["stacktrace_raw"], str) and row["stacktrace_raw"].strip()


@then("audit error exists for audit-500-1 with exception class SystemError")
def then_audit_500_error_exists():
    row = fetch_one(
        sql=(
            "SELECT exception_class, stacktrace_raw FROM audit_error "
            "WHERE request_id = $1 ORDER BY created_at DESC LIMIT 1"
        ),
        args=("audit-500-1",),
    )
    assert row["exception_class"] == "SystemError"
    assert isinstance(row["stacktrace_raw"], str) and row["stacktrace_raw"].strip()


@then("audit request row count for audit-dup-1 is 2")
def then_duplicate_count_is_two():
    row = fetch_one(
        sql="SELECT COUNT(*) AS cnt FROM audit_request WHERE request_id = $1",
        args=("audit-dup-1",),
    )
    assert int(row["cnt"]) == 2


@then("audit request row count for audit-dup-cross-1 is 2")
def then_duplicate_cross_count_is_two():
    row = fetch_one(
        sql="SELECT COUNT(*) AS cnt FROM audit_request WHERE request_id = $1",
        args=("audit-dup-cross-1",),
    )
    assert int(row["cnt"]) == 2


@then("audit request body and response body are persisted for audit-search-200-1")
def then_request_response_body_persisted(ctx):
    row = fetch_one(
        sql=(
            "SELECT request_body_raw, response_body_raw, response_status_code FROM audit_request "
            "WHERE request_id = $1 ORDER BY id DESC LIMIT 1"
        ),
        args=("audit-search-200-1",),
    )
    assert row["response_status_code"] == 200
    assert isinstance(row["request_body_raw"], str) and row["request_body_raw"].strip()
    assert isinstance(row["response_body_raw"], str) and row["response_body_raw"].strip()
    request_json = json.loads(row["request_body_raw"])
    response_json = json.loads(row["response_body_raw"])
    assert request_json == ctx["search_payload"]
    assert isinstance(response_json, dict)
    assert "request_id" in response_json


@then("audit event sequence is complete for audit-search-200-1")
def then_event_sequence_complete():
    rows = fetch_rows(
        sql=(
            "SELECT event_type FROM audit_event WHERE request_id = $1 "
            "ORDER BY created_at ASC"
        ),
        args=("audit-search-200-1",),
    )
    event_types = [str(r["event_type"]) for r in rows]
    assert "API_REQUEST_RECEIVED" in event_types
    assert "API_HANDLER_STARTED" in event_types
    assert "API_HANDLER_COMPLETED" in event_types
    assert "API_RESPONSE_SENT" in event_types


@then("audit request rows for audit-dup-cross-1 include paths /health and /missing-audit-path")
def then_cross_endpoint_paths():
    rows = fetch_rows(
        sql="SELECT path FROM audit_request WHERE request_id = $1 ORDER BY id ASC",
        args=("audit-dup-cross-1",),
    )
    paths = [str(r["path"]) for r in rows]
    assert "/health" in paths
    assert "/missing-audit-path" in paths
