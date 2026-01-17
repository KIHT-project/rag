from __future__ import annotations

import json
from typing import Any

from pytest_bdd import given, scenario, then, when

from tests.bdd.helpers.ingestion_api import (
    extract_job_id,
    extract_request_id,
    json_body,
    poll_job_until_terminal,
    post_ingest,
)
from tests.bdd.helpers.search_api import post_search


@scenario("../features/search.feature", "Search happy path")
def test_search_happy_path():
    pass


@scenario("../features/search.feature", "Search with filters")
def test_search_with_filters():
    pass


@scenario("../features/search.feature", "Schema validation path")
def test_search_schema_validation_path():
    pass


@scenario("../features/search.feature", "Empty result set")
def test_empty_result_set():
    pass


@scenario("../features/search.feature", "top_k limits results")
def test_top_k_limits_results():
    pass


@scenario("../features/search.feature", "Filters exclude non matching documents")
def test_filters_exclude_non_matching_documents():
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
    return {
        "embedding_model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "items": items,
    }


def _ingest_one(client, *, item: dict[str, Any], request_id: str) -> dict[str, Any]:
    payload = _make_ingest_payload(items=[item])
    res = post_ingest(client, payload=payload, request_id=request_id)
    assert res.status_code == 202, f"Expected 202, got {res.status_code}, body={json_body(res)}"

    job_id = extract_job_id(res)
    poll = poll_job_until_terminal(client, job_id=job_id)

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
        content_text="Example abstract text.",
    )
    ctx["expected_doi"] = item["doi"]
    ctx["ingest_job"] = _ingest_one(client, item=item, request_id="bdd-search-ingest-1")


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

    _ingest_one(client, item=item1, request_id="bdd-search-ingest-1a")
    _ingest_one(client, item=item2, request_id="bdd-search-ingest-1b")

    ctx["expected_dois"] = [item1["doi"], item2["doi"]]


@given("two documents with different diseases are ingested")
def given_two_documents_with_different_diseases_are_ingested(client, ctx):
    item1 = _make_item(
        doi="10.1000/xyz123",
        disease="thrombosis",
        year=2020,
        source_type="pubmed_abstract",
        content_text="shared token alpha",
    )
    item2 = _make_item(
        doi="10.1000/xyz999",
        disease="stroke",
        year=2020,
        source_type="pubmed_abstract",
        content_text="shared token beta",
    )

    _ingest_one(client, item=item1, request_id="bdd-search-ingest-2a")
    _ingest_one(client, item=item2, request_id="bdd-search-ingest-2b")

    ctx["thrombosis_doi"] = item1["doi"]
    ctx["non_thrombosis_doi"] = item2["doi"]


@given("no documents are ingested")
def given_no_documents_are_ingested(ctx):
    ctx["no_ingest"] = True


@given("search request is valid")
def given_search_request_valid(ctx):
    ctx["search_payload"] = {
        "query": "Example abstract text",
        "top_k": 5,
    }


@given("top_k is set to 1")
def given_top_k_is_set_to_1(ctx):
    p = json.loads(json.dumps(ctx.get("search_payload") or {}))
    p["top_k"] = 1
    ctx["search_payload"] = p


@given("search request matches both documents")
def given_search_request_matches_both_documents(ctx):
    ctx["search_payload"] = {
        "query": "shared token",
        "top_k": 5,
    }


@given("search filters are set")
def given_search_filters_set(ctx):
    p = json.loads(json.dumps(ctx.get("search_payload") or {}))
    p["filters"] = {
        "disease": "thrombosis",
        "year_min": 2019,
        "year_max": 2021,
        "source_type": "pubmed_abstract",
    }
    ctx["search_payload"] = p


@given("search filters are set to disease thrombosis only")
def given_search_filters_set_to_disease_thrombosis_only(ctx):
    p = json.loads(json.dumps(ctx.get("search_payload") or {}))
    p["filters"] = {"disease": "thrombosis"}
    ctx["search_payload"] = p


@given("search request is invalid")
def given_search_request_invalid(ctx):
    ctx["search_payload"] = {"query": ""}


@when("I POST search")
def when_post_search(client, ctx):
    res = post_search(client, payload=ctx["search_payload"], request_id="bdd-search-1")
    ctx["res"] = res


@then("the response status is 200")
def then_200(ctx):
    assert ctx["res"].status_code == 200


@then("the response status is 422")
def then_422(ctx):
    assert ctx["res"].status_code == 422


@then("request id is present")
def then_request_id_present(ctx):
    rid = extract_request_id(ctx["res"])
    assert isinstance(rid, str) and rid, f"Missing request id, headers={dict(ctx['res'].headers)}"


@then("effective embedding model id is present")
def then_effective_embedding_model_id_present(ctx):
    body = json_body(ctx["res"])
    val = body.get("effective_embedding_model_id")
    assert isinstance(val, str) and val


@then("search hits include the ingested doi")
def then_hits_include_doi(ctx):
    body = json_body(ctx["res"])
    hits = body.get("hits")
    assert isinstance(hits, list)

    expected = ctx["expected_doi"]
    assert any(isinstance(h, dict) and h.get("doi") == expected for h in hits), (
        f"Expected to find doi={expected} in hits, hits={hits}"
    )


@then("search hits are empty")
def then_hits_are_empty(ctx):
    body = json_body(ctx["res"])
    hits = body.get("hits")
    assert isinstance(hits, list)
    assert len(hits) == 0, f"Expected empty hits, hits={hits}"


@then("search hits count is 1")
def then_hits_count_is_one(ctx):
    body = json_body(ctx["res"])
    hits = body.get("hits")
    assert isinstance(hits, list)
    assert len(hits) == 1, f"Expected 1 hit, hits={hits}"


@then("only thrombosis documents are returned")
def then_only_thrombosis_documents_are_returned(ctx):
    body = json_body(ctx["res"])
    hits = body.get("hits")
    assert isinstance(hits, list)

    thrombosis_doi = ctx["thrombosis_doi"]
    non_thrombosis_doi = ctx["non_thrombosis_doi"]

    hit_dois = [h.get("doi") for h in hits if isinstance(h, dict)]
    assert thrombosis_doi in hit_dois, f"Expected thrombosis doi in hits, hits={hits}"
    assert non_thrombosis_doi not in hit_dois, f"Expected non thrombosis doi excluded, hits={hits}"


@then("the error code is validation_error")
def then_error_validation(ctx):
    res = ctx["res"]
    assert _extract_error_code(res) == "validation_error"


def _extract_error_code(res) -> str:
    body = json_body(res)

    if isinstance(body.get("error"), str):
        return body["error"]

    if isinstance(body.get("detail"), list):
        return "validation_error"

    raise AssertionError(f"Missing error code, status={res.status_code}, body={body}")


@given("two documents with different years are ingested")
def given_two_documents_with_different_years_are_ingested(client, ctx):
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

    _ingest_one(client, item=item1, request_id="bdd-search-ingest-y1")
    _ingest_one(client, item=item2, request_id="bdd-search-ingest-y2")

    ctx["year2020_doi"] = item1["doi"]
    ctx["year2021_doi"] = item2["doi"]


@given("search filters are set to year 2020 only")
def given_search_filters_set_to_year_2020_only(ctx):
    p = json.loads(json.dumps(ctx.get("search_payload") or {}))
    p["filters"] = {"year_min": 2020, "year_max": 2020}
    ctx["search_payload"] = p


@then("only year 2020 documents are returned")
def then_only_year_2020_documents_are_returned(ctx):
    body = json_body(ctx["res"])
    hits = body.get("hits")
    assert isinstance(hits, list)

    year2020_doi = ctx["year2020_doi"]
    year2021_doi = ctx["year2021_doi"]

    hit_dois = [h.get("doi") for h in hits if isinstance(h, dict)]
    assert year2020_doi in hit_dois, f"Expected year 2020 doi in hits, hits={hits}"
    assert year2021_doi not in hit_dois, f"Expected year 2021 doi excluded, hits={hits}"
