from __future__ import annotations

from datetime import timezone

from biomed_platform.api.models.generated import schemas
from biomed_platform.common.utils import normalize_doi, compute_body_hash
from biomed_platform.core.domains.ingestion import (
    IngestBatchAccepted,
    IngestBatchCommand,
    IngestItem,
    IngestItemState,
    IngestionJob,
    JobCounts,
    JobState,
)


def to_ingest_batch_command(
    *,
    request: schemas.IngestBatchRequest,
    effective_embedding_model_id: str,
    idempotency_key: str | None,
) -> IngestBatchCommand:
    items: list[IngestItem] = []
    for it in request.items:
        authors = tuple(a.root for a in (it.authors or []))
        items.append(
            IngestItem(
                doi_original=it.doi,
                doi_normalized=normalize_doi(it.doi),
                disease=it.disease.value,
                source_type=it.source_type.value,
                content_text=it.content_text,
                year=it.year,
                title=it.title,
                journal=it.journal,
                authors=authors,
            )
        )

    body_hash = compute_body_hash(request)

    return IngestBatchCommand(
        effective_embedding_model_id=effective_embedding_model_id,
        items=tuple(items),
        idempotency_key=idempotency_key,
        body_hash=body_hash,
    )


def to_ingest_job_accepted_response(
    result: IngestBatchAccepted,
) -> schemas.IngestJobAcceptedResponse:
    return schemas.IngestJobAcceptedResponse(
        job_id=result.job_id,
        state=schemas.State.queued,
    )


def _to_api_job_state(state: JobState) -> schemas.JobState:
    return schemas.JobState(state.value)


def _to_api_item_state(state: IngestItemState) -> schemas.IngestItemState:
    return schemas.IngestItemState(state.value)


def to_ingest_job_status_response(job: IngestionJob) -> schemas.IngestJobStatusResponse:
    created_at = job.created_at
    updated_at = job.updated_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)

    counts = job.counts or JobCounts(
        total=len(job.items),
        succeeded=0,
        failed=0,
        skipped_duplicate=0,
    )

    api_items: list[schemas.IngestItemStatus] = []
    for it in job.items:
        api_items.append(
            schemas.IngestItemStatus(
                doi=it.doi_original,
                doc_id=it.doc_id,
                state=_to_api_item_state(it.state),
                message=it.message,
            )
        )

    return schemas.IngestJobStatusResponse(
        job_id=job.job_id,
        state=_to_api_job_state(job.state),
        created_at=created_at,
        updated_at=updated_at,
        effective_embedding_model_id=job.effective_embedding_model_id,
        counts=schemas.JobCounts(
            total=counts.total,
            succeeded=counts.succeeded,
            failed=counts.failed,
            skipped_duplicate=counts.skipped_duplicate,
        ),
        items=api_items,
    )
