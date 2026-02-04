from __future__ import annotations

from typing import Any

from pytest_bdd import given, scenario, then, when

from tests.bdd.helpers.documents_api import delete_document, extract_error_code
from tests.bdd.helpers.ingestion_api import (
    extract_job_id,
    extract_request_id,
    json_body,
    poll_job_until_terminal,
    post_ingest,
)


@scenario("../features/documents.feature", "Delete happy path")
def test_delete_happy_path():
    pass


@scenario("../features/documents.feature", "Not found path")
def test_delete_not_found_path():
    pass


@scenario("../features/documents.feature", "Invalid DOI path")
def test_delete_invalid_doi_path():
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
    ctx["doi"] = item["doi"]
    ctx["ingest_job"] = _ingest_one(client, item=item, request_id="bdd-doc-delete-ingest-1")


@when("I DELETE the document by doi")
def when_delete_document(client, ctx):
    res = delete_document(client, doi=ctx["doi"], request_id="bdd-doc-delete-1")
    ctx["res"] = res


@when("I DELETE the document by doi again")
def when_delete_document_again(client, ctx):
    res = delete_document(client, doi=ctx["doi"], request_id="bdd-doc-delete-2")
    ctx["res"] = res


@when("I DELETE an unknown document")
def when_delete_unknown_document(client, ctx):
    res = delete_document(client, doi="10.9999/unknown", request_id="bdd-doc-delete-404")
    ctx["res"] = res


@when("I DELETE a document with invalid doi")
def when_delete_invalid_doi(client, ctx):
    res = delete_document(client, doi="not-a-doi", request_id="bdd-doc-delete-400")
    ctx["res"] = res


@then("the response status is 204")
def then_204(ctx):
    assert ctx["res"].status_code == 204


@then("the response status is 404")
def then_404(ctx):
    assert ctx["res"].status_code == 404


@then("the response status is 400")
def then_400(ctx):
    assert ctx["res"].status_code == 400


@then("request id is present")
def then_request_id_present(ctx):
    rid = extract_request_id(ctx["res"])
    assert isinstance(rid, str) and rid, f"Missing request id, headers={dict(ctx['res'].headers)}"


@then("the error code is not_found")
def then_error_not_found(ctx):
    assert extract_error_code(ctx["res"]) == "not_found"


@then("the error code is validation_error")
def then_error_validation(ctx):
    assert extract_error_code(ctx["res"]) == "validation_error"
