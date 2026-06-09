# src/app/python/download_data/find_doi.py
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import requests


PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
EUROPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"

REQUEST_TIMEOUT_S = int(os.getenv("NCBI_TIMEOUT_S", "1200"))

SLEEP_S = float(os.getenv("NCBI_SLEEP_S", "0.12"))
MIN_TITLE_SIMILARITY = float(os.getenv("DOI_MIN_TITLE_SIMILARITY", "0.93"))

EFETCH_MAX_PMIDS_PER_CALL = int(os.getenv("EFETCH_MAX_PMIDS_PER_CALL", "200"))
EFETCH_MAX_ID_CHARS = int(os.getenv("EFETCH_MAX_ID_CHARS", "8000"))

IDCONV_MAX_PMIDS_PER_CALL = int(os.getenv("IDCONV_MAX_PMIDS_PER_CALL", "200"))

CROSSREF_ROWS = int(os.getenv("CROSSREF_ROWS", "5"))
OPENALEX_PER_PAGE = int(os.getenv("OPENALEX_PER_PAGE", "1"))
EUROPMC_PAGE_SIZE = int(os.getenv("EUROPMC_PAGE_SIZE", "1"))

CACHE_PATH = os.getenv("DOI_CACHE_PATH", "").strip()
CACHE_FLUSH_EVERY = int(os.getenv("DOI_CACHE_FLUSH_EVERY", "200"))

NCBI_TOOL = (os.getenv("NCBI_TOOL") or "labelstudio-tools").strip()
NCBI_EMAIL = (os.getenv("NCBI_EMAIL") or "").strip()
NCBI_API_KEY = (os.getenv("NCBI_API_KEY") or "").strip()


@dataclass
class _Cache:
    path: Path
    data: dict[str, dict[str, Any]]
    dirty: bool = False

    @classmethod
    def load(cls) -> "_Cache" | None:
        p = CACHE_PATH
        if not p:
            return None
        path = Path(p).expanduser()
        if not path.exists():
            return cls(path=path, data={}, dirty=False)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                norm: dict[str, dict[str, Any]] = {}
                for k, v in raw.items():
                    if isinstance(k, str) and isinstance(v, dict):
                        norm[k] = v
                return cls(path=path, data=norm, dirty=False)
        except Exception:
            return cls(path=path, data={}, dirty=False)
        return cls(path=path, data={}, dirty=False)

    def get(self, pubmed_id: str) -> dict[str, Any] | None:
        return self.data.get(pubmed_id)

    def put(self, pubmed_id: str, rec: dict[str, Any]) -> None:
        self.data[pubmed_id] = rec
        self.dirty = True

    def flush(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
        self.dirty = False


def _sleep() -> None:
    if SLEEP_S > 0:
        time.sleep(SLEEP_S)


def _request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout_s: int = REQUEST_TIMEOUT_S,
    max_attempts: int = 6,
    backoff_s: float = 0.6,
    **kwargs: Any,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = session.request(method, url, timeout=timeout_s, **kwargs)
            if r.status_code in (429, 500, 502, 503, 504):
                _sleep()
                time.sleep(backoff_s * attempt)
                continue
            return r
        except requests.RequestException as e:
            last_exc = e
            _sleep()
            time.sleep(backoff_s * attempt)
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("request failed without exception")


def _get_pubmed_id(d: dict[str, Any]) -> str | None:
    source = d.get("source")
    source_id = d.get("source_id")
    pmid = d.get("pmid")

    if isinstance(source, str) and source.strip().lower() == "pubmed" and source_id is not None:
        s = str(source_id).strip()
        return s if s else None

    if pmid is not None:
        s = str(pmid).strip()
        return s if s else None

    return None


def _extract_pubmed_ids(tasks: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tasks:
        d = t.get("data")
        if not isinstance(d, dict):
            continue
        pubmed_id = _get_pubmed_id(d)
        if not pubmed_id:
            continue
        if pubmed_id not in seen:
            seen.add(pubmed_id)
            out.append(pubmed_id)
    return out


def _pubmed_id_to_title(tasks: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for t in tasks:
        d = t.get("data")
        if not isinstance(d, dict):
            continue
        pubmed_id = _get_pubmed_id(d)
        if not pubmed_id:
            continue
        title = d.get("title")
        if isinstance(title, str) and title.strip():
            out[pubmed_id] = title.strip()
    return out


def _pubmed_id_to_year(tasks: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tasks:
        d = t.get("data")
        if not isinstance(d, dict):
            continue
        pubmed_id = _get_pubmed_id(d)
        if not pubmed_id:
            continue
        year = d.get("year")
        if isinstance(year, int):
            out[pubmed_id] = year
        elif isinstance(year, str) and year.strip().isdigit():
            out[pubmed_id] = int(year.strip())
    return out


def _norm_title(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_title(a), _norm_title(b)).ratio()


def _chunk_ids_for_efetch(pubmed_ids: list[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    cur: list[str] = []
    cur_chars = 0

    for pubmed_id in pubmed_ids:
        s = str(pubmed_id).strip()
        if not s:
            continue

        add_chars = len(s) + (1 if cur else 0)
        next_chars = cur_chars + add_chars
        next_count = len(cur) + 1

        if cur and (next_count > EFETCH_MAX_PMIDS_PER_CALL or next_chars > EFETCH_MAX_ID_CHARS):
            batches.append(cur)
            cur = [s]
            cur_chars = len(s)
        else:
            cur.append(s)
            cur_chars = next_chars

    if cur:
        batches.append(cur)

    return batches


def _chunk_ids(ids: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(ids), size):
        yield ids[i : i + size]


def _parse_pubmed_records(xml_text: str) -> dict[str, dict[str, Any]]:
    root = ET.fromstring(xml_text)
    out: dict[str, dict[str, Any]] = {}

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//MedlineCitation/PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        pubmed_id = pmid_el.text.strip()
        if not pubmed_id:
            continue

        record: dict[str, Any] = {}

        for aid in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if (aid.attrib.get("IdType") or "").lower() == "doi":
                if aid.text and aid.text.strip():
                    record["doi"] = aid.text.strip()
                    break

        journal_el = article.find(".//Journal/Title")
        if journal_el is not None and journal_el.text and journal_el.text.strip():
            record["journal"] = journal_el.text.strip()

        year: str | None = None
        year_el = article.find(".//JournalIssue/PubDate/Year")
        if year_el is not None and year_el.text:
            year = year_el.text.strip()
        if year is None:
            year_el = article.find(".//ArticleDate/Year")
            if year_el is not None and year_el.text:
                year = year_el.text.strip()
        if year and year.isdigit():
            record["year"] = int(year)

        authors: list[str] = []
        for a in article.findall(".//AuthorList/Author"):
            last = a.findtext("LastName")
            initials = a.findtext("Initials")
            collective = a.findtext("CollectiveName")
            if last and initials:
                authors.append(f"{last} {initials}")
            elif last:
                authors.append(last)
            elif collective:
                authors.append(collective)
        if authors:
            record["authors"] = authors

        if record:
            out[pubmed_id] = record

    return out


def _fetch_pubmed_xml(pubmed_ids: list[str], session: requests.Session) -> str:
    data: dict[str, str] = {
        "db": "pubmed",
        "retmode": "xml",
        "id": ",".join(pubmed_ids),
        "tool": NCBI_TOOL,
    }
    if NCBI_EMAIL:
        data["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        data["api_key"] = NCBI_API_KEY

    r = _request_with_retries(session, "POST", PUBMED_EFETCH_URL, data=data)
    r.raise_for_status()
    return r.text


def _parse_idconv_tsv(text: str) -> dict[str, dict[str, Any]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}
    header = lines[0].split("\t")
    idx_pmid = None
    idx_doi = None
    idx_pmcid = None
    for i, h in enumerate(header):
        hl = h.strip().lower()
        if hl == "pmid":
            idx_pmid = i
        elif hl == "doi":
            idx_doi = i
        elif hl == "pmcid":
            idx_pmcid = i
    if idx_pmid is None:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for ln in lines[1:]:
        cols = ln.split("\t")
        if idx_pmid >= len(cols):
            continue
        pubmed_id = cols[idx_pmid].strip()
        if not pubmed_id:
            continue
        rec: dict[str, Any] = {}
        if idx_doi is not None and idx_doi < len(cols):
            doi = cols[idx_doi].strip()
            if doi:
                rec["doi"] = doi
        if idx_pmcid is not None and idx_pmcid < len(cols):
            pmcid = cols[idx_pmcid].strip()
            if pmcid:
                rec["pmcid"] = pmcid
        if rec:
            out[pubmed_id] = rec
    return out


def _idconv_for_pubmed_ids(pubmed_ids: list[str], session: requests.Session) -> dict[str, dict[str, Any]]:
    params: dict[str, str] = {
        "tool": NCBI_TOOL,
        "format": "tsv",
        "ids": ",".join(pubmed_ids),
    }
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL

    r = _request_with_retries(session, "GET", NCBI_IDCONV_URL, params=params)
    if r.status_code >= 400:
        return {}
    try:
        return _parse_idconv_tsv(r.text)
    except Exception:
        return {}


def _openalex_metadata_for_pubmed_id(pubmed_id: str, session: requests.Session) -> dict[str, Any]:
    params = {"filter": f"pmid:{pubmed_id}", "per-page": str(OPENALEX_PER_PAGE)}
    r = _request_with_retries(session, "GET", OPENALEX_WORKS_URL, params=params)
    if r.status_code >= 400:
        return {}

    try:
        payload = r.json()
    except Exception:
        return {}

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return {}

    hit = results[0]
    out: dict[str, Any] = {}

    doi = hit.get("doi")
    if isinstance(doi, str) and doi.strip():
        out["doi"] = doi.replace("https://doi.org/", "").strip()

    year = hit.get("publication_year")
    if isinstance(year, int):
        out["year"] = year

    journal = ((hit.get("host_venue") or {}).get("display_name"))
    if isinstance(journal, str) and journal.strip():
        out["journal"] = journal.strip()

    authorships = hit.get("authorships")
    if isinstance(authorships, list):
        names: list[str] = []
        for a in authorships:
            name = ((a.get("author") or {}).get("display_name"))
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        if names:
            out["authors"] = names

    return out


def _europmc_by_pubmed_id(pubmed_id: str, session: requests.Session) -> dict[str, Any]:
    params = {
        "query": f"EXT_ID:{pubmed_id} AND SRC:MED",
        "format": "json",
        "pageSize": str(EUROPMC_PAGE_SIZE),
    }
    r = _request_with_retries(session, "GET", EUROPMC_SEARCH_URL, params=params)
    if r.status_code >= 400:
        return {}

    try:
        payload = r.json()
    except Exception:
        return {}

    hit = (((payload.get("resultList") or {}).get("result")) or [])
    if not isinstance(hit, list) or not hit:
        return {}

    rec0 = hit[0]
    out: dict[str, Any] = {}

    doi = rec0.get("doi")
    if isinstance(doi, str) and doi.strip():
        out["doi"] = doi.strip()

    year = rec0.get("pubYear")
    if isinstance(year, str) and year.strip().isdigit():
        out["year"] = int(year.strip())

    journal = rec0.get("journalTitle")
    if isinstance(journal, str) and journal.strip():
        out["journal"] = journal.strip()

    author_str = rec0.get("authorString")
    if isinstance(author_str, str) and author_str.strip():
        authors = [a.strip() for a in author_str.split(",") if a.strip()]
        if authors:
            out["authors"] = authors

    title = rec0.get("title")
    if isinstance(title, str) and title.strip():
        out["title"] = title.strip()

    return out


def _crossref_best_by_title(
    title: str,
    session: requests.Session,
    *,
    expected_year: int | None,
) -> dict[str, Any] | None:
    params: dict[str, str] = {"query.title": title, "rows": str(CROSSREF_ROWS)}
    r = _request_with_retries(session, "GET", CROSSREF_WORKS_URL, params=params)
    if r.status_code >= 400:
        return None

    try:
        payload = r.json()
    except Exception:
        return None

    items = (((payload.get("message") or {}).get("items")) or [])
    if not isinstance(items, list) or not items:
        return None

    best: dict[str, Any] | None = None
    best_score = 0.0

    for it in items:
        doi = it.get("DOI")
        titles = it.get("title")
        if not isinstance(doi, str) or not doi.strip():
            continue
        if not isinstance(titles, list) or not titles:
            continue
        cand_title = titles[0]
        if not isinstance(cand_title, str) or not cand_title.strip():
            continue

        score = _title_similarity(title, cand_title)

        if expected_year is not None:
            issued = it.get("issued") or {}
            parts = issued.get("date-parts") if isinstance(issued, dict) else None
            yr = None
            if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
                if isinstance(parts[0][0], int):
                    yr = parts[0][0]
            if yr is not None and abs(yr - expected_year) > 1:
                score *= 0.92

        if score > best_score:
            best_score = score
            best = it

    if best is None:
        return None
    if best_score < MIN_TITLE_SIMILARITY:
        return None

    out: dict[str, Any] = {"doi": str(best.get("DOI")).strip()}
    return out


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    if not src:
        return dst
    for k, v in src.items():
        if v is None:
            continue
        if k not in dst:
            dst[k] = v
    return dst


def enrich_tasks_with_doi(
    tasks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    pubmed_ids = _extract_pubmed_ids(tasks)
    if not pubmed_ids:
        return tasks, []

    pubmed_id_title = _pubmed_id_to_title(tasks)
    pubmed_id_year = _pubmed_id_to_year(tasks)

    cache = _Cache.load()
    session = requests.Session()
    pubmed_meta: dict[str, dict[str, Any]] = {}

    if cache is not None:
        for pubmed_id in pubmed_ids:
            c = cache.get(pubmed_id)
            if isinstance(c, dict) and c:
                pubmed_meta[pubmed_id] = dict(c)

    need_for_pubmed: list[str] = []
    for pubmed_id in pubmed_ids:
        rec = pubmed_meta.get(pubmed_id, {})
        if ("doi" not in rec) or ("year" not in rec) or ("journal" not in rec) or ("authors" not in rec):
            need_for_pubmed.append(pubmed_id)

    for batch in _chunk_ids_for_efetch(need_for_pubmed):
        xml = _fetch_pubmed_xml(batch, session)
        parsed = _parse_pubmed_records(xml)
        for k, v in parsed.items():
            pubmed_meta.setdefault(k, {})
            _merge(pubmed_meta[k], v)
        _sleep()

    need_for_idconv: list[str] = []
    for pubmed_id in pubmed_ids:
        rec = pubmed_meta.get(pubmed_id, {})
        if "doi" not in rec:
            need_for_idconv.append(pubmed_id)

    for batch in _chunk_ids(need_for_idconv, IDCONV_MAX_PMIDS_PER_CALL):
        parsed = _idconv_for_pubmed_ids(batch, session)
        for k, v in parsed.items():
            pubmed_meta.setdefault(k, {})
            _merge(pubmed_meta[k], v)
        _sleep()

    for pubmed_id in pubmed_ids:
        rec = pubmed_meta.get(pubmed_id, {})
        need = ("doi" not in rec) or ("year" not in rec) or ("journal" not in rec) or ("authors" not in rec)
        if not need:
            continue
        epmc = _europmc_by_pubmed_id(pubmed_id, session)
        if epmc:
            pubmed_meta.setdefault(pubmed_id, {})
            _merge(pubmed_meta[pubmed_id], epmc)
        _sleep()

    for pubmed_id in pubmed_ids:
        rec = pubmed_meta.get(pubmed_id, {})
        need = ("doi" not in rec) or ("year" not in rec) or ("journal" not in rec) or ("authors" not in rec)
        if not need:
            continue
        oa = _openalex_metadata_for_pubmed_id(pubmed_id, session)
        if oa:
            pubmed_meta.setdefault(pubmed_id, {})
            _merge(pubmed_meta[pubmed_id], oa)
        _sleep()

    for pubmed_id in pubmed_ids:
        rec = pubmed_meta.get(pubmed_id, {})
        if "doi" in rec:
            continue
        title = pubmed_id_title.get(pubmed_id)
        if not title:
            continue
        expected_year = pubmed_id_year.get(pubmed_id)
        cr = _crossref_best_by_title(title, session, expected_year=expected_year)
        if cr:
            pubmed_meta.setdefault(pubmed_id, {})
            _merge(pubmed_meta[pubmed_id], cr)
        _sleep()

    missing_source_ids: list[str] = []
    dirty_writes = 0

    for t in tasks:
        d = t.get("data")
        if not isinstance(d, dict):
            continue

        pubmed_id = _get_pubmed_id(d)
        if not pubmed_id:
            continue

        meta = pubmed_meta.get(pubmed_id) or {}
        doi = meta.get("doi")

        if not isinstance(doi, str) or not doi.strip():
            missing_source_ids.append(pubmed_id)
            continue

        doi_s = doi.strip()

        d["doi"] = doi_s
        d["doi_url"] = f"https://doi.org/{urllib.parse.quote(doi_s, safe=':/')}"

        d["source"] = "pubmed"
        d["source_id"] = pubmed_id
        d["source_url"] = f"https://pubmed.ncbi.nlm.nih.gov/{urllib.parse.quote(pubmed_id)}/"

        year = meta.get("year")
        if isinstance(year, int):
            d["year"] = year
        elif isinstance(year, str) and year.strip().isdigit():
            d["year"] = int(year.strip())

        journal = meta.get("journal")
        if isinstance(journal, str) and journal.strip():
            d["journal"] = journal.strip()

        authors = meta.get("authors")
        if isinstance(authors, list) and authors:
            d["authors"] = authors

        if cache is not None:
            cache.put(pubmed_id, {k: meta[k] for k in meta.keys()})
            dirty_writes += 1
            if dirty_writes % CACHE_FLUSH_EVERY == 0:
                cache.flush()

    if cache is not None:
        cache.flush()

    seen_m: set[str] = set()
    missing_unique: list[str] = []
    for m in missing_source_ids:
        if m not in seen_m:
            seen_m.add(m)
            missing_unique.append(m)

    return tasks, missing_unique
