from __future__ import annotations

from typing import Any

from pytest_bdd import given, scenario, then, when

from tests.bdd.helpers.ask_api import post_ask
from tests.bdd.helpers.ingestion_api import (
    extract_job_id,
    extract_request_id,
    json_body,
    poll_job_until_terminal,
    post_ingest,
)


@scenario("../features/ask.feature", "Ask happy path")
def test_ask_happy_path():
    pass


@scenario("../features/ask.feature", "Ask with HyDE enabled")
def test_ask_with_hyde_enabled():
    pass


@scenario("../features/ask.feature", "Schema validation path")
def test_ask_schema_validation_path():
    pass


@scenario("../features/ask.feature", "Filters exclude non matching documents")
def test_ask_filters_exclude_non_matching_documents():
    pass


def _make_item(
    *,
    doi: str,
    disease: str,
    year: int,
    source_type: str,
    content_text: str,
    title: str = "Example title",
    journal: str = "Example journal",
    authors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "doi": doi,
        "disease": disease,
        "year": year,
        "source_type": source_type,
        "title": title,
        "journal": journal,
        "authors": authors or ["Author One", "Author Two"],
        "content_text": content_text,
    }


def _make_ingest_payload(*, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"items": items}


def _ingest_one(client, *, item: dict[str, Any], request_id: str) -> dict[str, Any]:
    payload = _make_ingest_payload(items=[item])
    res = post_ingest(client, payload=payload, request_id=request_id)
    assert res.status_code == 202, f"Expected 202, got {res.status_code}, body={json_body(res)}"

    job_id = extract_job_id(res)
    poll = poll_job_until_terminal(client, job_id=job_id, request_id=request_id)

    state = (poll.job.get("state") or poll.job.get("status") or "").lower()
    assert state == "succeeded", f"Ingest did not succeed, job={poll.job}"

    return poll.job


@given("a document is ingested")
def given_document_is_ingested(client, ctx):
    item = _make_item(
        doi="10.1000/xyz123",
        disease="thrombosis",
        year=2020,
        source_type="pubmed_abstract",
        content_text="Example abstract text about thrombosis and risk factors.",
    )
    ctx["expected_doi"] = item["doi"]
    ctx["ingest_job"] = _ingest_one(client, item=item, request_id="bdd-ask-ingest-1")


@given("two documents are ingested")
def given_two_documents_are_ingested(client, ctx):
    item1 = _make_item(
        doi="10.1000/xyz123",
        disease="thrombosis",
        year=2020,
        source_type="pubmed_abstract",
        content_text="shared token alpha",
    )
    item2 = _make_item(
        doi="10.1000/xyz124",
        disease="thrombosis",
        year=2021,
        source_type="pubmed_abstract",
        content_text="shared token beta",
    )

    _ingest_one(client, item=item1, request_id="bdd-ask-ingest-1a")
    _ingest_one(client, item=item2, request_id="bdd-ask-ingest-1b")

    ctx["excluded_doi"] = item1["doi"]
    ctx["expected_doi"] = item2["doi"]


@given("ask request is valid")
def given_ask_request_is_valid(ctx):
    ctx["ask_payload"] = {
        "question": "What are the key thrombosis risk factors mentioned in the evidence?",
        "filters": None,
    }


@given("ask request is valid with filters year_min 2021")
def given_ask_request_is_valid_with_filters(ctx):
    ctx["ask_payload"] = {
        "question": "shared token",
        "filters": {"year_min": 2021},
    }


@given("ask request is invalid")
def given_ask_request_is_invalid(ctx):
    ctx["ask_payload"] = {"filters": None}


@when("I POST ask")
def when_i_post_ask(client, ctx):
    payload = ctx.get("ask_payload")
    assert isinstance(payload, dict), "Missing ask_payload in ctx"
    res = post_ask(client, payload=payload, request_id="bdd-ask-req-1")
    ctx["res"] = res


@when("I POST ask with hyde enabled")
def when_i_post_ask_with_hyde_enabled(client, ctx):
    payload = ctx.get("ask_payload")
    assert isinstance(payload, dict), "Missing ask_payload in ctx"
    res = post_ask(
        client,
        payload=payload,
        request_id="bdd-ask-req-2",
        hyde_enabled=True,
    )
    ctx["res"] = res


@then("the response status is 200")
def then_status_200(ctx):
    assert ctx["res"].status_code == 200, f"body={ctx['res'].text}"


@then("the response status is 422")
def then_status_422(ctx):
    assert ctx["res"].status_code == 422, f"body={ctx['res'].text}"


@then("request id is present")
def then_request_id_is_present(ctx):
    rid = extract_request_id(ctx["res"])
    assert isinstance(rid, str) and rid, f"Missing request id, headers={dict(ctx['res'].headers)}"


@then("effective hyde enabled is false")
def then_effective_hyde_enabled_is_false(ctx):
    body = json_body(ctx["res"])
    assert body.get("effective_hyde_enabled") is False


@then("effective hyde enabled is true")
def then_effective_hyde_enabled_is_true(ctx):
    body = json_body(ctx["res"])
    assert body.get("effective_hyde_enabled") is True


@then("answer summary is present")
def then_answer_summary_is_present(ctx):
    body = json_body(ctx["res"])
    answer = body.get("answer")
    assert isinstance(answer, dict), f"Expected answer object, body={body}"
    summary = answer.get("summary")
    assert isinstance(summary, str) and summary.strip(), "Expected non empty answer.summary"


@then("citations include the ingested doi")
def then_citations_include_the_ingested_doi(ctx):
    body = json_body(ctx["res"])
    citations = body.get("citations") or []
    expected = ctx.get("expected_doi")
    assert isinstance(expected, str) and expected
    assert any(isinstance(c, dict) and c.get("doi") == expected for c in citations), (
        f"Expected citations to include doi {expected}, citations={citations}"
    )


@then("citations include the expected doi")
def then_citations_include_the_expected_doi(ctx):
    body = json_body(ctx["res"])
    citations = body.get("citations") or []
    expected = ctx.get("expected_doi")
    assert isinstance(expected, str) and expected
    assert any(isinstance(c, dict) and c.get("doi") == expected for c in citations), (
        f"Expected citations to include doi {expected}, citations={citations}"
    )


@then("citations do not include the excluded doi")
def then_citations_do_not_include_the_excluded_doi(ctx):
    body = json_body(ctx["res"])
    citations = body.get("citations") or []
    excluded = ctx.get("excluded_doi")
    assert isinstance(excluded, str) and excluded
    assert all(not isinstance(c, dict) or c.get("doi") != excluded for c in citations), (
        f"Expected citations to not include doi {excluded}, citations={citations}"
    )
