from __future__ import annotations

import re
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.pubmed import PubMedDocument
from biomed_platform.core.errors.errors import SystemError

log = get_logger(__name__)

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _first_text(elem: ET.Element | None) -> str | None:
    if elem is None:
        return None
    text = "".join(elem.itertext()).strip()
    return text or None


def _parse_year(value: str | None) -> int | None:
    if not value:
        return None
    m = re.search(r"\b(\d{4})\b", value)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_authors(article: ET.Element | None) -> list[str] | None:
    if article is None:
        return None
    authors: list[str] = []
    author_list = article.find("AuthorList")
    if author_list is None:
        return None
    for author in author_list.findall("Author"):
        collective = _first_text(author.find("CollectiveName"))
        if collective:
            authors.append(collective)
            continue
        last = _first_text(author.find("LastName")) or ""
        fore = _first_text(author.find("ForeName")) or _first_text(author.find("Initials")) or ""
        name = " ".join(p for p in (fore.strip(), last.strip()) if p)
        if name:
            authors.append(name)
    return authors or None


def _extract_abstract(article: ET.Element | None) -> str | None:
    if article is None:
        return None
    abstract = article.find("Abstract")
    if abstract is None:
        return None
    parts: list[str] = []
    for part in abstract.findall("AbstractText"):
        label = part.attrib.get("Label")
        text = _first_text(part) or ""
        if not text:
            continue
        if label:
            parts.append(f"{label}: {text}")
        else:
            parts.append(text)
    return "\n".join(parts).strip() or None


def _extract_mesh_terms(citation: ET.Element | None) -> list[str] | None:
    if citation is None:
        return None
    mesh_list = citation.find("MeshHeadingList")
    if mesh_list is None:
        return None
    terms: list[str] = []
    for heading in mesh_list.findall("MeshHeading"):
        descriptor = _first_text(heading.find("DescriptorName"))
        if descriptor:
            terms.append(descriptor)
        for qualifier in heading.findall("QualifierName"):
            qual = _first_text(qualifier)
            if qual:
                terms.append(qual)
    return terms or None


def _extract_ids(
    article: ET.Element | None,
    pubmed_data: ET.Element | None,
) -> tuple[str | None, str | None]:
    doi = None
    pmcid = None

    if article is not None:
        for eloc in article.findall("ELocationID"):
            if eloc.attrib.get("EIdType") == "doi":
                doi = _first_text(eloc) or doi

    if pubmed_data is not None:
        id_list = pubmed_data.find("ArticleIdList")
        if id_list is not None:
            for aid in id_list.findall("ArticleId"):
                id_type = (aid.attrib.get("IdType") or "").lower()
                val = _first_text(aid)
                if not val:
                    continue
                if id_type == "doi":
                    doi = val
                elif id_type == "pmc":
                    pmcid = val

    return doi, pmcid


def _parse_pubmed_xml(xml_text: str) -> PubMedDocument | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        log.exception("Failed to parse PubMed XML response")
        return None

    article_node = root.find(".//PubmedArticle")
    if article_node is None:
        return None

    citation = article_node.find("MedlineCitation")
    pubmed_data = article_node.find("PubmedData")
    if citation is None:
        return None

    pmid = _first_text(citation.find("PMID"))
    article = citation.find("Article")
    title = _first_text(article.find("ArticleTitle")) if article is not None else None
    journal = _first_text(article.find("Journal/Title")) if article is not None else None

    year = None
    if article is not None:
        pub_date = article.find("Journal/JournalIssue/PubDate")
        if pub_date is not None:
            year = _parse_year(_first_text(pub_date.find("Year")))
            if year is None:
                year = _parse_year(_first_text(pub_date.find("MedlineDate")))

    authors = _extract_authors(article)
    abstract = _extract_abstract(article)
    mesh_terms = _extract_mesh_terms(citation)
    doi, pmcid = _extract_ids(article, pubmed_data)

    return PubMedDocument(
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        title=title,
        journal=journal,
        year=year,
        authors=authors,
        mesh_terms=mesh_terms,
        abstract=abstract,
        full_text=None,
    )


def _extract_section_blocks(sec: ET.Element) -> list[str]:
    blocks: list[str] = []

    def _localname(elem: ET.Element) -> str:
        tag = str(elem.tag)
        return tag.rsplit("}", 1)[-1] if "}" in tag else tag

    def _find_child(parent: ET.Element, name: str) -> ET.Element | None:
        for child in list(parent):
            if _localname(child) == name:
                return child
        return None

    title = _first_text(_find_child(sec, "title"))
    if title:
        blocks.append(title.strip())

    paras: list[str] = []
    for child in list(sec):
        if _localname(child) in {"title", "sec"}:
            continue
        text = _first_text(child)
        if text:
            paras.append(text)

    if paras:
        blocks.append("\n\n".join(paras).strip())

    return blocks


def _collect_section_blocks(body: ET.Element) -> list[str]:
    sections: list[str] = []

    def visit(sec: ET.Element) -> None:
        sections.extend(_extract_section_blocks(sec))
        for sub in list(sec):
            tag = str(sub.tag)
            name = tag.rsplit("}", 1)[-1] if "}" in tag else tag
            if name == "sec":
                visit(sub)

    for sec in list(body):
        tag = str(sec.tag)
        name = tag.rsplit("}", 1)[-1] if "}" in tag else tag
        if name == "sec":
            visit(sec)

    return sections


def _extract_pmc_text(xml_text: str) -> str | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        log.exception("Failed to parse PMC XML response")
        return None

    body = None
    for el in root.iter():
        tag = str(el.tag)
        name = tag.rsplit("}", 1)[-1] if "}" in tag else tag
        if name == "body":
            body = el
            break
    if body is None:
        return None

    sections = _collect_section_blocks(body)
    if sections:
        return "\n\n".join(s for s in sections if s).strip() or None

    text = _first_text(body)
    return text or None


@dataclass(frozen=True, slots=True)
class PubMedClientAdapter:
    client: httpx.AsyncClient
    base_url: str = _EUTILS_BASE
    max_retries: int = 2
    backoff_seconds: float = 0.5

    def _retry_delay_seconds(self, *, attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(float(retry_after), 0.0)
                except ValueError:
                    try:
                        dt = datetime.fromisoformat(retry_after)
                    except ValueError:
                        dt = None
                    if dt is not None:
                        now = datetime.now(timezone.utc)
                        return max((dt - now).total_seconds(), 0.0)
        return self.backoff_seconds * (2**attempt)

    def _should_retry(self, *, response: httpx.Response) -> bool:
        if response.status_code == 429:
            return True
        return 500 <= response.status_code <= 599

    async def _get(self, *, url: str, params: dict[str, Any]) -> httpx.Response:
        last_response: httpx.Response | None = None
        for attempt in range(max(0, self.max_retries) + 1):
            try:
                response = await self.client.get(url, params=params)
            except httpx.RequestError as exc:
                if attempt >= self.max_retries:
                    raise SystemError(
                        code="pubmed_unavailable",
                        message="Failed to reach PubMed",
                        details={"error": str(exc)},
                        retryable=True,
                    ) from exc
                await asyncio.sleep(self._retry_delay_seconds(attempt=attempt, response=None))
                continue

            last_response = response
            if self._should_retry(response=response) and attempt < self.max_retries:
                await asyncio.sleep(self._retry_delay_seconds(attempt=attempt, response=response))
                continue

            return response

        return last_response  # type: ignore[return-value]

    async def _resolve_pmid(self, *, doi: str) -> str | None:
        if not doi:
            return None
        url = f"{self.base_url}/esearch.fcgi"
        params = {"db": "pubmed", "term": f"{doi}[doi]", "retmode": "json"}
        res = await self._get(url=url, params=params)
        try:
            res.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise SystemError(
                code="pubmed_unavailable",
                message="PubMed search failed",
                details={"status_code": exc.response.status_code},
                retryable=True,
            ) from exc
        data = res.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        if isinstance(ids, list) and ids:
            return str(ids[0])
        return None

    async def _fetch_pubmed(self, *, pmid: str) -> PubMedDocument | None:
        if not pmid:
            return None
        url = f"{self.base_url}/efetch.fcgi"
        params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
        res = await self._get(url=url, params=params)
        try:
            res.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise SystemError(
                code="pubmed_unavailable",
                message="PubMed fetch failed",
                details={"status_code": exc.response.status_code},
                retryable=True,
            ) from exc
        doc = _parse_pubmed_xml(res.text)
        return doc

    async def _fetch_pmc_text(self, *, pmcid: str) -> str | None:
        if not pmcid:
            return None
        url = f"{self.base_url}/efetch.fcgi"
        params = {"db": "pmc", "id": pmcid, "retmode": "xml"}
        res = await self._get(url=url, params=params)
        try:
            res.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise SystemError(
                code="pubmed_unavailable",
                message="PMC fetch failed",
                details={"status_code": exc.response.status_code},
                retryable=True,
            ) from exc
        return _extract_pmc_text(res.text)

    async def fetch_document(self, *, doi: str | None, pmid: str | None) -> PubMedDocument | None:
        resolved_pmid = pmid
        if not resolved_pmid and doi:
            resolved_pmid = await self._resolve_pmid(doi=doi)
        if not resolved_pmid:
            return None

        doc = await self._fetch_pubmed(pmid=resolved_pmid)
        if doc is None:
            return None

        pmcid = doc.pmcid
        full_text = None
        if pmcid:
            full_text = await self._fetch_pmc_text(pmcid=pmcid)

        return PubMedDocument(
            doi=doc.doi,
            pmid=doc.pmid or resolved_pmid,
            pmcid=pmcid,
            title=doc.title,
            journal=doc.journal,
            year=doc.year,
            authors=doc.authors,
            mesh_terms=doc.mesh_terms,
            abstract=doc.abstract,
            full_text=full_text,
        )
