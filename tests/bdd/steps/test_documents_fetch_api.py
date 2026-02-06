from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
from pytest_bdd import given, scenario, then, when

from biomed_platform.core.domains.ingestion import IngestBatchAccepted, JobState
from biomed_platform.core.domains.pubmed import PubMedDocument
from biomed_platform.adapters.pubmed.pubmed_client import PubMedClientAdapter
from tests.bdd.helpers.documents_api import post_fetch_batch, post_fetch_document
from tests.bdd.helpers.ingestion_api import extract_request_id, json_body


@scenario("../features/documents_fetch.feature", "Fetch by DOI with ingest")
def test_fetch_by_doi_with_ingest():
    pass


@scenario("../features/documents_fetch.feature", "Fetch by PMID without ingest")
def test_fetch_by_pmid_without_ingest():
    pass


@scenario("../features/documents_fetch.feature", "Fetch by PMID with namespaced PMC without ingest")
def test_fetch_by_pmid_with_namespaced_pmc_without_ingest():
    pass


@scenario("../features/documents_fetch.feature", "Fetch batch accepted")
def test_fetch_batch_accepted():
    pass


class _FakePubMedClient:
    def __init__(self, docs_by_doi: dict[str, PubMedDocument], docs_by_pmid: dict[str, PubMedDocument]):
        self._by_doi = docs_by_doi
        self._by_pmid = docs_by_pmid

    async def fetch_document(self, *, doi: str | None, pmid: str | None) -> PubMedDocument | None:
        if doi and doi in self._by_doi:
            return self._by_doi[doi]
        if pmid and pmid in self._by_pmid:
            return self._by_pmid[pmid]
        return None


class _FakeIngestionService:
    def __init__(self):
        self.commands: list[Any] = []

    async def ingest_batch(self, cmd):
        self.commands.append(cmd)
        return IngestBatchAccepted(job_id="job123", state=JobState.queued)


def _install_stubs(client, *, doc: PubMedDocument) -> None:
    pubmed = _FakePubMedClient(
        docs_by_doi={doc.doi or "": doc},
        docs_by_pmid={doc.pmid or "": doc},
    )
    client.app.state.pubmed_client = pubmed
    client.app.state.ingestion_service = _FakeIngestionService()


@given("PubMed has a document for doi")
def given_pubmed_has_document_for_doi(client, ctx):
    doc = PubMedDocument(
        doi="10.1000/xyz123",
        pmid="123",
        pmcid=None,
        title="Example title",
        journal="Example journal",
        year=2024,
        authors=["A Author"],
        mesh_terms=["Thrombosis"],
        abstract="Example abstract text.",
        full_text=None,
    )
    _install_stubs(client, doc=doc)
    ctx["doi"] = doc.doi


@given("PubMed has a document for pmid")
def given_pubmed_has_document_for_pmid(client, ctx):
    doc = PubMedDocument(
        doi="10.2000/abc456",
        pmid="999",
        pmcid="PMC123",
        title="Example title",
        journal="Example journal",
        year=2024,
        authors=["A Author"],
        mesh_terms=["Thrombosis"],
        abstract="Example abstract text.",
        full_text="Full text content.",
    )
    _install_stubs(client, doc=doc)
    ctx["pmid"] = doc.pmid


@given("PubMed has a namespaced PMC document for pmid")
def given_pubmed_has_namespaced_pmc_document_for_pmid(client, ctx):
    pubmed_xml = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>18648623</PMID>
      <Article>
        <ArticleTitle>Example title</ArticleTitle>
        <Journal>
          <JournalIssue>
            <PubDate>
              <Year>2008</Year>
            </PubDate>
          </JournalIssue>
          <Title>Example journal</Title>
        </Journal>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pmc">PMC2475953</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

    pmc_xml_with_default_ns = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns="http://jats.nlm.nih.gov">
  <body>
    <sec>
      <title>Intro</title>
      <p>First paragraph.</p>
      <sec>
        <title>Methods</title>
        <p>Second paragraph.</p>
      </sec>
    </sec>
  </body>
</article>
"""

    class _StubHttpxClient:
        async def get(self, url: str, *, params: dict[str, object]) -> httpx.Response:
            request = httpx.Request("GET", url)

            if params.get("db") == "pubmed":
                return httpx.Response(200, text=pubmed_xml, request=request)
            if params.get("db") == "pmc":
                return httpx.Response(200, text=pmc_xml_with_default_ns, request=request)

            return httpx.Response(400, text="unexpected request", request=request)

    client.app.state.pubmed_client = PubMedClientAdapter(
        client=_StubHttpxClient(),  # type: ignore[arg-type]
        base_url="http://example",
        max_retries=0,
    )
    client.app.state.ingestion_service = _FakeIngestionService()
    ctx["pmid"] = "18648623"


@when("I POST fetch by doi")
def when_post_fetch_by_doi(client, ctx):
    res = post_fetch_document(
        client,
        payload={"doi": ctx["doi"]},
        request_id="bdd-fetch-1",
    )
    ctx["res"] = res


@when("I POST fetch by pmid without ingest")
def when_post_fetch_by_pmid_without_ingest(client, ctx):
    res = post_fetch_document(
        client,
        payload={"pmid": ctx["pmid"]},
        ingest_enabled=False,
        request_id="bdd-fetch-2",
    )
    ctx["res"] = res


@when("I POST fetch batch")
def when_post_fetch_batch(client, ctx):
    res = post_fetch_batch(
        client,
        payload={"items": [{"doi": ctx["doi"]}]},
        request_id="bdd-fetch-3",
    )
    ctx["res"] = res


@then("the response status is 200")
def then_200(ctx):
    assert ctx["res"].status_code == 200


@then("the response status is 202")
def then_202(ctx):
    assert ctx["res"].status_code == 202


@then("request id is present")
def then_request_id_present(ctx):
    rid = extract_request_id(ctx["res"])
    assert isinstance(rid, str) and rid


@then("content text source is abstract")
def then_content_source_abstract(ctx):
    body = json_body(ctx["res"])
    assert body.get("content_text_source") == "abstract"


@then("content text source is pmc")
def then_content_source_pmc(ctx):
    body = json_body(ctx["res"])
    assert body.get("content_text_source") == "pmc"


@then("full text available is false")
def then_full_text_false(ctx):
    body = json_body(ctx["res"])
    assert body.get("full_text_available") is False


@then("full text available is true")
def then_full_text_true(ctx):
    body = json_body(ctx["res"])
    assert body.get("full_text_available") is True


@then("ingest job id is present")
def then_ingest_job_id_present(ctx):
    body = json_body(ctx["res"])
    ingest = body.get("ingest") or {}
    job_id = ingest.get("job_id")
    assert isinstance(job_id, str) and job_id


@then("batch job id is present")
def then_batch_job_id_present(ctx):
    body = json_body(ctx["res"])
    job_id = body.get("job_id")
    assert isinstance(job_id, str) and job_id
