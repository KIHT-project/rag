from __future__ import annotations

import json

import pytest
from pytest_bdd import given, scenario, then, when

from tests.bdd.helpers.ingestion_api import (
    extract_job_id,
    extract_request_id,
    json_body,
    poll_job_until_terminal,
    post_ingest,
)

# -----------------------------------------------------------------------------
# Scenarios
# -----------------------------------------------------------------------------

@scenario("../features/ingestion.feature", "Ingest happy path")
def test_ingest_happy_path():
    pass


@scenario("../features/ingestion.feature", "Duplicate DOI in same request")
def test_duplicate_doi_in_same_request():
    pass


@scenario("../features/ingestion.feature", "Idempotency happy path")
def test_idempotency_happy_path():
    pass


@scenario("../features/ingestion.feature", "Idempotency conflict path")
def test_idempotency_conflict_path():
    pass

@scenario("../features/ingestion.feature", "Not found path")
def test_not_found_path():
    pass


@scenario("../features/ingestion.feature", "Schema validation path")
def test_schema_validation_path():
    pass



# -----------------------------------------------------------------------------
# Payload builders
# -----------------------------------------------------------------------------

def _base_ingest_payload() -> dict:
    return {
        "items": [
            {
                "doi": "10.1000/xyz123",
                "disease": "thrombosis",
                "year": 2020,
                "source_type": "pubmed_abstract",
                "title": "Example title",
                "journal": "Example journal",
                "authors": ["Author One", "Author Two"],
                "content_text": "Example abstract text.",
            }
        ],
    }


# -----------------------------------------------------------------------------
# Given
# -----------------------------------------------------------------------------

@given("ingestion payload is valid")
def given_payload_valid(ctx):
    ctx["payload"] = _base_ingest_payload()
    ctx["expected_total"] = 1


@given("ingestion payload has duplicate doi in same request")
def given_payload_duplicate_doi(ctx):
    p = _base_ingest_payload()
    p["items"] = [p["items"][0], dict(p["items"][0])]
    ctx["payload"] = p
    ctx["expected_counts_duplicate"] = {"total": 2}


@given("ingestion payload is invalid")
def given_payload_invalid(ctx):
    ctx["payload"] = {"items": "not a list"}


@given("idempotency key is set")
def given_idempotency_key(ctx):
    ctx["idempotency_key"] = "bdd-idem-key-1"


# -----------------------------------------------------------------------------
# When
# -----------------------------------------------------------------------------

@when("I POST ingest")
def when_post_ingest(client, ctx):
    request_id = "bdd-request-1"
    ctx["request_id"] = request_id

    payload = ctx.get("payload") or _base_ingest_payload()
    ctx["payload"] = payload

    headers = {"X-Request-Id": request_id}
    if "idempotency_key" in ctx:
        headers["Idempotency-Key"] = ctx["idempotency_key"]

    res = client.post("/v1/ingest/items", json=payload, headers=headers)
    ctx["post_res"] = res


@when("I POST ingest again with same idempotency key and same body")
def when_post_ingest_idem_same(client, ctx):
    res = post_ingest(
        client,
        payload=ctx["payload"],
        idempotency_key=ctx["idempotency_key"],
        request_id="bdd-request-2",
    )
    ctx["post_res_2"] = res


@when("I POST ingest again with same idempotency key but different body")
def when_post_ingest_idem_conflict(client, ctx):
    p2 = json.loads(json.dumps(ctx["payload"]))
    p2["items"][0]["doi"] = "10.1000/DIFFERENT"

    res = post_ingest(
        client,
        payload=p2,
        idempotency_key=ctx["idempotency_key"],
        request_id="bdd-request-3",
    )
    ctx["post_res_2"] = res


@when("I GET an unknown job id")
def when_get_unknown_job_id(client, ctx):
    res = client.get(
        "/v1/ingest/jobs/unknown-job-id-zzz",
        headers={"X-Request-Id": "bdd-request-404"},
    )
    ctx["res"] = res


# -----------------------------------------------------------------------------
# Then
# -----------------------------------------------------------------------------

@then("the response status is 202")
def then_202(ctx):
    assert ctx["post_res"].status_code == 202


@then("the response status is 400")
def then_400(ctx):
    assert ctx["post_res_2"].status_code == 400


@then("the response status is 409")
def then_409(ctx):
    res = ctx.get("post_res_2") or ctx.get("post_res")
    assert res.status_code == 409


@then("the response status is 404")
def then_404(ctx):
    assert ctx["res"].status_code == 404


@then("the response status is 422")
def then_422(ctx):
    assert ctx["post_res"].status_code == 422


@then("I capture the job id")
def then_capture_job_id(ctx):
    ctx["job_id_1"] = extract_job_id(ctx["post_res"])


@then("the job id is the same as before")
def then_job_id_same(ctx):
    jid1 = ctx["job_id_1"]
    jid2 = extract_job_id(ctx["post_res_2"])
    assert jid1 == jid2


@then("I can poll the job until terminal state")
def then_poll_terminal(client, ctx):
    job_id = extract_job_id(ctx["post_res"])
    poll = poll_job_until_terminal(client, job_id=job_id)
    ctx["poll"] = poll.job


@then("the job state is succeeded")
def then_job_state_succeeded(ctx):
    state = (ctx["poll"].get("state") or ctx["poll"].get("status") or "").lower()
    assert state == "succeeded"


@then("the item status counts are internally consistent")
def then_counts_internally_consistent(ctx):
    counts = ctx["poll"]["counts"]

    total = counts.get("total")
    assert isinstance(total, int) and total >= 0

    terminal = (
        counts.get("succeeded", 0)
        + counts.get("skipped_duplicate", 0)
        + counts.get("failed", 0)
    )

    assert terminal == total


@then("the error code is validation_error")
def then_error_validation(ctx):
    res = ctx.get("post_res_2") or ctx.get("post_res")
    assert _extract_error_code(res) == "validation_error"


@then("the error code is not_found")
def then_error_404(ctx):
    assert _extract_error_code(ctx["res"]) == "not_found"


@then("the error code is duplicate_doi")
def then_error_duplicate_doi(ctx):
    res = ctx.get("post_res_2") or ctx.get("post_res")
    assert _extract_error_code(res) == "duplicate_doi"


@then("request id is present")
def then_request_id_present(ctx):
    res = ctx.get("res") or ctx.get("post_res_2") or ctx.get("post_res")
    assert extract_request_id(res)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _extract_error_code(res) -> str:
    body = json_body(res)

    if isinstance(body.get("error"), str):
        return body["error"]

    if isinstance(body.get("detail"), list):
        return "validation_error"

    raise AssertionError(f"Missing error code, status={res.status_code}, body={body}")
