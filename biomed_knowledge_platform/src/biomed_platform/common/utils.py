import hashlib

from biomed_platform.api.models.generated import schemas
from biomed_platform.common.logging import get_logger

log = get_logger(__name__)


def normalize_doi(doi: str) -> str:
    return doi.strip().lower()


def compute_doc_id(*, doi_normalized: str) -> str:
    return hashlib.sha256(doi_normalized.encode("utf-8")).hexdigest()


def compute_body_hash(request: schemas.IngestBatchRequest) -> str:
    parts: list[str] = []
    parts.append(str(request.embedding_model_id or ""))

    for item in request.items:
        authors = item.authors or []
        authors_flat = ",".join(a.root for a in authors)
        parts.append(
            "|".join(
                [
                    item.doi,
                    item.disease.value,
                    str(item.year or ""),
                    item.source_type.value,
                    str(item.title or ""),
                    str(item.journal or ""),
                    authors_flat,
                    item.content_text,
                ]
            )
        )

    joined = "\n".join(parts).encode("utf-8")
    digest = hashlib.sha256(joined).hexdigest()

    log.debug(
        "Computed ingest body hash, items_count=%d, has_embedding_model_id=%s",
        len(request.items),
        request.embedding_model_id is not None,
    )

    return digest
