import hashlib
import re
from typing import Iterable

from biomed_platform.api.models.generated.schemas import IngestItem
from biomed_platform.common.logging import get_logger

log = get_logger(__name__)
_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)


def canonicalize_doi(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""

    m = _DOI_RE.search(text)
    if not m:
        return ""

    doi = m.group(1).strip().lower()
    doi = doi.rstrip(").,;]}>\"'")
    return doi


def normalize_doi(doi: str) -> str:
    return canonicalize_doi(doi)


def compute_doc_id(*, doi_normalized: str) -> str:
    return hashlib.sha256(doi_normalized.encode("utf-8")).hexdigest()


def _author_to_str(author: object) -> str:
    if author is None:
        return ""
    if isinstance(author, str):
        return author
    root = getattr(author, "root", None)
    if isinstance(root, str):
        return root
    return ""


def compute_body_hash_from_items(
    *,
    effective_embedding_model_id: str,
    items: Iterable[IngestItem],
) -> str:
    """
    Stable semantic hash for idempotency.
    Uses canonicalized mapped values, not raw request fields.
    """
    items_list = list(items)

    parts: list[str] = []
    parts.append((effective_embedding_model_id or "").strip())

    for it in items_list:
        authors_flat = ",".join(
            a.strip() for a in (_author_to_str(x) for x in (it.authors or ())) if a and a.strip()
        )
        parts.append(
            "|".join(
                [
                    (getattr(it, "doi_normalized", "") or "").strip(),
                    str(it.disease or ""),
                    str(it.year or ""),
                    str(it.source_type or ""),
                    str(it.title or ""),
                    str(it.journal or ""),
                    authors_flat,
                    str(it.content_text or ""),
                ]
            )
        )

    joined = "\n".join(parts).encode("utf-8")
    digest = hashlib.sha256(joined).hexdigest()

    log.debug(
        "Computed ingest body hash from mapped items, items_count=%d,"
        "has_effective_embedding_model_id=%s",
        len(items_list),
        bool(effective_embedding_model_id and effective_embedding_model_id.strip()),
    )

    return digest
