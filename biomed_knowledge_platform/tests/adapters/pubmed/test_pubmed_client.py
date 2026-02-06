from __future__ import annotations

import httpx
import pytest

from biomed_platform.adapters.pubmed.pubmed_client import (
    PubMedClientAdapter,
    _extract_pmc_text,
    _parse_pubmed_xml,
)
from biomed_platform.core.errors.errors import SystemError


PUBMED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>123</PMID>
      <Article>
        <ArticleTitle>Sample title</ArticleTitle>
        <Journal>
          <JournalIssue>
            <PubDate>
              <Year>2020</Year>
            </PubDate>
          </JournalIssue>
          <Title>Journal Name</Title>
        </Journal>
        <Abstract>
          <AbstractText Label="BACKGROUND">Background text.</AbstractText>
          <AbstractText>More text.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>Jane</ForeName>
          </Author>
          <Author>
            <CollectiveName>Group</CollectiveName>
          </Author>
        </AuthorList>
        <ELocationID EIdType="doi">10.1/abc</ELocationID>
      </Article>
      <MeshHeadingList>
        <MeshHeading>
          <DescriptorName>Thrombosis</DescriptorName>
          <QualifierName>therapy</QualifierName>
        </MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1/xyz</ArticleId>
        <ArticleId IdType="pmc">PMC12345</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

PUBMED_MEDLINE_DATE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>456</PMID>
      <Article>
        <ArticleTitle>Title</ArticleTitle>
        <Journal>
          <JournalIssue>
            <PubDate>
              <MedlineDate>2021 Jan-Feb</MedlineDate>
            </PubDate>
          </JournalIssue>
          <Title>Journal</Title>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

PMC_XML = """<?xml version="1.0" encoding="UTF-8"?>
<article>
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

PMC_XML_WITH_DEFAULT_NS = """<?xml version="1.0" encoding="UTF-8"?>
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
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self._queue: list[object] = []

    def queue(self, item: object) -> None:
        self._queue.append(item)

    async def get(self, url: str, *, params: dict[str, object]) -> httpx.Response:
        self.calls.append((url, params))
        if not self._queue:
            raise RuntimeError("No queued response")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, httpx.Response)
        return item


def _response(*, status_code: int, url: str, text: str | None = None, json: object | None = None) -> httpx.Response:
    request = httpx.Request("GET", url)
    if json is not None:
        return httpx.Response(status_code, json=json, request=request)
    return httpx.Response(status_code, text=text or "", request=request)


def test_parse_pubmed_xml_extracts_fields() -> None:
    doc = _parse_pubmed_xml(PUBMED_XML)
    assert doc is not None
    assert doc.doi == "10.1/xyz"
    assert doc.pmid == "123"
    assert doc.pmcid == "PMC12345"
    assert doc.title == "Sample title"
    assert doc.journal == "Journal Name"
    assert doc.year == 2020
    assert doc.authors == ["Jane Smith", "Group"]
    assert doc.mesh_terms == ["Thrombosis", "therapy"]
    assert doc.abstract == "BACKGROUND: Background text.\nMore text."


def test_parse_pubmed_xml_uses_medline_date_when_year_missing() -> None:
    doc = _parse_pubmed_xml(PUBMED_MEDLINE_DATE_XML)
    assert doc is not None
    assert doc.year == 2021


def test_parse_pubmed_xml_returns_none_on_invalid_xml() -> None:
    assert _parse_pubmed_xml("<not-xml") is None


def test_extract_pmc_text_collects_sections() -> None:
    text = _extract_pmc_text(PMC_XML)
    assert text == "Intro\n\nFirst paragraph.\n\nMethods\n\nSecond paragraph."


def test_extract_pmc_text_collects_sections_with_default_namespace() -> None:
    text = _extract_pmc_text(PMC_XML_WITH_DEFAULT_NS)
    assert text == "Intro\n\nFirst paragraph.\n\nMethods\n\nSecond paragraph."


@pytest.mark.asyncio
async def test_fetch_document_resolves_pmid_and_fetches_pmc() -> None:
    http = _StubHttpxClient()
    http.queue(
        _response(
            status_code=200,
            url="http://example/esearch.fcgi",
            json={"esearchresult": {"idlist": ["123"]}},
        )
    )
    http.queue(
        _response(
            status_code=200,
            url="http://example/efetch.fcgi",
            text=PUBMED_XML,
        )
    )
    http.queue(
        _response(
            status_code=200,
            url="http://example/efetch.fcgi",
            text=PMC_XML,
        )
    )

    client = PubMedClientAdapter(client=http, base_url="http://example", max_retries=0)  # type: ignore[arg-type]

    doc = await client.fetch_document(doi="10.1/xyz", pmid=None)
    assert doc is not None
    assert doc.pmid == "123"
    assert doc.doi == "10.1/xyz"
    assert doc.full_text == "Intro\n\nFirst paragraph.\n\nMethods\n\nSecond paragraph."
    assert len(http.calls) == 3


@pytest.mark.asyncio
async def test_fetch_document_returns_none_when_esearch_empty() -> None:
    http = _StubHttpxClient()
    http.queue(
        _response(
            status_code=200,
            url="http://example/esearch.fcgi",
            json={"esearchresult": {"idlist": []}},
        )
    )

    client = PubMedClientAdapter(client=http, base_url="http://example", max_retries=0)  # type: ignore[arg-type]

    doc = await client.fetch_document(doi="10.1/xyz", pmid=None)
    assert doc is None


@pytest.mark.asyncio
async def test_fetch_document_raises_system_error_on_http_error() -> None:
    http = _StubHttpxClient()
    http.queue(
        _response(
            status_code=500,
            url="http://example/esearch.fcgi",
            text="error",
        )
    )

    client = PubMedClientAdapter(client=http, base_url="http://example", max_retries=0)  # type: ignore[arg-type]

    with pytest.raises(SystemError) as exc:
        await client.fetch_document(doi="10.1/xyz", pmid=None)

    assert exc.value.code == "pubmed_unavailable"
    assert exc.value.retryable is True


@pytest.mark.asyncio
async def test_fetch_document_raises_system_error_on_request_error() -> None:
    http = _StubHttpxClient()
    request = httpx.Request("GET", "http://example/esearch.fcgi")
    http.queue(httpx.ConnectError("boom", request=request))

    client = PubMedClientAdapter(client=http, base_url="http://example", max_retries=0)  # type: ignore[arg-type]

    with pytest.raises(SystemError) as exc:
        await client.fetch_document(doi="10.1/xyz", pmid=None)

    assert exc.value.code == "pubmed_unavailable"
    assert exc.value.retryable is True
