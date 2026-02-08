from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from scheduler_pubmed.src.core.domains.pubmed_query import PubMedQuery
from scheduler_pubmed.src.core.domains.scheduler import (
    DoiExecutionStatus,
    IngestJobItemStatus,
    IngestJobStatus,
    PubMedSearchResult,
    RunStatus,
    SchedulerRunCreated,
    SchedulerStatus,
    TriggerType,
)
from scheduler_pubmed.src.core.ports.documents_client import DocumentsClient
from scheduler_pubmed.src.core.ports.pubmed_client import PubMedClient
from scheduler_pubmed.src.core.ports.pubmed_query_repository import (
    PubMedQueryRepository,
)
from scheduler_pubmed.src.core.ports.scheduler_repository import SchedulerRepository

_TERMINAL_JOB_STATES = {"succeeded", "failed", "partial"}
_INGESTED_ITEM_STATES = {"succeeded"}
_FAILED_ITEM_STATES = {"failed"}
_SKIPPED_EXISTS_ITEM_STATES = {"skipped_duplicate"}


def _normalize_doi(value: str) -> str:
    return value.strip()


def _parse_schedule_time(raw: str) -> tuple[int, int]:
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(raw)
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(raw)
    return hour, minute


def _compute_next_run_at(*, now: datetime, utc_schedule: list[str]) -> datetime:
    if not utc_schedule:
        return now

    candidates: list[datetime] = []
    for value in utc_schedule:
        hour, minute = _parse_schedule_time(value)
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        candidates.append(candidate)
    return min(candidates)


def _to_uuid_or_none(raw: str | None) -> UUID | None:
    if not raw:
        return None
    try:
        return UUID(raw)
    except ValueError:
        return None


def _derive_query_status(*, resolved_count: int, failed_count: int) -> RunStatus:
    if resolved_count == 0:
        return RunStatus.SUCCESS
    if failed_count == 0:
        return RunStatus.SUCCESS
    if failed_count >= resolved_count:
        return RunStatus.FAILED
    return RunStatus.PARTIAL_SUCCESS


def _derive_run_status(statuses: list[RunStatus]) -> RunStatus:
    if not statuses:
        return RunStatus.SUCCESS
    if all(item == RunStatus.SUCCESS for item in statuses):
        return RunStatus.SUCCESS
    if all(item == RunStatus.FAILED for item in statuses):
        return RunStatus.FAILED
    return RunStatus.PARTIAL_SUCCESS


def _filter_incremental_results(
    *,
    results: list[PubMedSearchResult],
    since: datetime | None,
) -> list[PubMedSearchResult]:
    if since is None:
        return results
    filtered: list[PubMedSearchResult] = []
    for item in results:
        if item.published_at is None:
            continue
        if item.published_at > since:
            filtered.append(item)
    return filtered


def _extract_unique_dois(*, results: list[PubMedSearchResult]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in results:
        if not item.doi:
            continue
        doi = _normalize_doi(item.doi)
        if not doi:
            continue
        key = doi.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(doi)
    return ordered


def _map_ingest_item_state(
    item: IngestJobItemStatus,
) -> tuple[DoiExecutionStatus, str | None]:
    state = item.state.strip().lower()
    if state in _INGESTED_ITEM_STATES:
        return DoiExecutionStatus.INGESTED, None
    if state in _FAILED_ITEM_STATES:
        return DoiExecutionStatus.FAILED, item.message
    if state in _SKIPPED_EXISTS_ITEM_STATES:
        return DoiExecutionStatus.SKIPPED_EXISTS, item.message
    return DoiExecutionStatus.ENQUEUED, item.message


@dataclass(slots=True)
class _QueryExecutionState:
    filtered_results: list[PubMedSearchResult]
    resolved_dois: list[str]
    doi_statuses: dict[str, tuple[DoiExecutionStatus, str | None]]
    enqueued_doi_count: int
    ingest_job_id: UUID | None
    query_error_message: str | None


class SchedulerOrchestrationUseCase:
    def __init__(
        self,
        *,
        query_repository: PubMedQueryRepository,
        scheduler_repository: SchedulerRepository,
        pubmed_client: PubMedClient,
        documents_client: DocumentsClient,
        ingest_poll_max_attempts: int = 20,
        ingest_poll_interval_seconds: float = 1.0,
    ) -> None:
        self._query_repository = query_repository
        self._scheduler_repository = scheduler_repository
        self._pubmed_client = pubmed_client
        self._documents_client = documents_client
        self._ingest_poll_max_attempts = max(1, ingest_poll_max_attempts)
        self._ingest_poll_interval_seconds = max(0.0, ingest_poll_interval_seconds)

    async def trigger_run(self, *, trigger_type: TriggerType) -> SchedulerRunCreated:
        return await self._scheduler_repository.create_run(trigger_type=trigger_type)

    async def get_status(self, *, enabled: bool, utc_schedule: list[str]) -> SchedulerStatus:
        now = datetime.now(UTC)
        next_run_at = _compute_next_run_at(now=now, utc_schedule=utc_schedule)
        last_run = await self._scheduler_repository.get_last_run()
        return SchedulerStatus(
            enabled=enabled,
            utc_schedule=utc_schedule,
            next_run_at=next_run_at,
            last_run_at=last_run.started_at if last_run else None,
            last_run_status=last_run.status if last_run else None,
        )

    async def execute_run(self, *, run_id: UUID, reldate_days: int | None = None) -> None:
        query_statuses: list[RunStatus] = []
        try:
            queries = await self._query_repository.list(enabled=True)
            for query in queries:
                status = await self._execute_query(
                    run_id=run_id,
                    query=query,
                    reldate_days=reldate_days,
                )
                query_statuses.append(status)

            run_status = _derive_run_status(query_statuses)
            await self._scheduler_repository.complete_run(run_id=run_id, status=run_status)
        except Exception:
            await self._scheduler_repository.complete_run(run_id=run_id, status=RunStatus.FAILED)

    async def _poll_ingest_job(self, *, job_id: str) -> IngestJobStatus | None:
        latest: IngestJobStatus | None = None
        for _ in range(self._ingest_poll_max_attempts):
            try:
                latest = await self._documents_client.get_ingest_job_status(job_id=job_id)
            except Exception:
                return None

            if latest.state.strip().lower() in _TERMINAL_JOB_STATES:
                return latest
            await asyncio.sleep(self._ingest_poll_interval_seconds)
        return latest

    async def _record_doi_status(
        self,
        *,
        run_id: UUID,
        query_execution_id: UUID,
        doi: str,
        status: DoiExecutionStatus,
        error_message: str | None,
        state: _QueryExecutionState,
    ) -> None:
        state.doi_statuses[doi] = (status, error_message)
        await self._scheduler_repository.upsert_doi_execution_result(
            query_execution_id=query_execution_id,
            run_id=run_id,
            doi=doi,
            status=status,
            error_message=error_message,
        )

    async def _resolve_query_results(
        self,
        *,
        query: PubMedQuery,
        state: _QueryExecutionState,
        reldate_days: int | None,
    ) -> None:
        search_results = await self._pubmed_client.search(
            query=query.pubmed_query,
            reldate_days=reldate_days,
        )
        state.filtered_results = _filter_incremental_results(
            results=search_results,
            since=query.last_successful_run_at,
        )
        state.resolved_dois = _extract_unique_dois(results=state.filtered_results)

    async def _resolve_existing_documents(
        self,
        *,
        run_id: UUID,
        query_execution_id: UUID,
        state: _QueryExecutionState,
    ) -> list[str]:
        to_enqueue: list[str] = []
        for doi in state.resolved_dois:
            try:
                exists = await self._documents_client.document_exists(doi=doi)
            except Exception as exc:
                await self._record_doi_status(
                    run_id=run_id,
                    query_execution_id=query_execution_id,
                    doi=doi,
                    status=DoiExecutionStatus.FAILED,
                    error_message=str(exc),
                    state=state,
                )
                continue

            if exists:
                await self._record_doi_status(
                    run_id=run_id,
                    query_execution_id=query_execution_id,
                    doi=doi,
                    status=DoiExecutionStatus.SKIPPED_EXISTS,
                    error_message=None,
                    state=state,
                )
                continue

            to_enqueue.append(doi)
        return to_enqueue

    async def _apply_ingest_job_updates(
        self,
        *,
        run_id: UUID,
        query_execution_id: UUID,
        to_enqueue: list[str],
        ingest_job: IngestJobStatus,
        state: _QueryExecutionState,
    ) -> None:
        mapped_by_doi = {_normalize_doi(item.doi).lower(): item for item in ingest_job.items}
        for doi in to_enqueue:
            item = mapped_by_doi.get(_normalize_doi(doi).lower())
            if item is None:
                continue
            mapped_status, mapped_error = _map_ingest_item_state(item)
            await self._record_doi_status(
                run_id=run_id,
                query_execution_id=query_execution_id,
                doi=doi,
                status=mapped_status,
                error_message=mapped_error,
                state=state,
            )

    async def _enqueue_new_documents(
        self,
        *,
        run_id: UUID,
        query_execution_id: UUID,
        to_enqueue: list[str],
        state: _QueryExecutionState,
    ) -> None:
        if not to_enqueue:
            return

        accepted = await self._documents_client.fetch_batch(dois=to_enqueue)
        state.ingest_job_id = _to_uuid_or_none(accepted.job_id)
        state.enqueued_doi_count = len(to_enqueue)

        for doi in to_enqueue:
            await self._record_doi_status(
                run_id=run_id,
                query_execution_id=query_execution_id,
                doi=doi,
                status=DoiExecutionStatus.ENQUEUED,
                error_message=None,
                state=state,
            )

        ingest_job = await self._poll_ingest_job(job_id=accepted.job_id)
        if ingest_job is None:
            return
        await self._apply_ingest_job_updates(
            run_id=run_id,
            query_execution_id=query_execution_id,
            to_enqueue=to_enqueue,
            ingest_job=ingest_job,
            state=state,
        )

    @staticmethod
    def _build_query_counts(
        *,
        state: _QueryExecutionState,
    ) -> tuple[int, int, int, int, int]:
        doi_skipped_exists_count = sum(
            1
            for status, _ in state.doi_statuses.values()
            if status == DoiExecutionStatus.SKIPPED_EXISTS
        )
        doi_failed_count = sum(
            1 for status, _ in state.doi_statuses.values() if status == DoiExecutionStatus.FAILED
        )
        return (
            len(state.filtered_results),
            len(state.resolved_dois),
            doi_skipped_exists_count,
            state.enqueued_doi_count,
            doi_failed_count,
        )

    async def _execute_query(
        self,
        *,
        run_id: UUID,
        query: PubMedQuery,
        reldate_days: int | None,
    ) -> RunStatus:
        query_execution_id = await self._scheduler_repository.create_query_execution(
            run_id=run_id,
            query_id=query.id,
        )

        now = datetime.now(UTC)
        state = _QueryExecutionState(
            filtered_results=[],
            resolved_dois=[],
            doi_statuses={},
            enqueued_doi_count=0,
            ingest_job_id=None,
            query_error_message=None,
        )

        try:
            await self._resolve_query_results(
                query=query,
                state=state,
                reldate_days=reldate_days,
            )
            to_enqueue = await self._resolve_existing_documents(
                run_id=run_id,
                query_execution_id=query_execution_id,
                state=state,
            )
            try:
                await self._enqueue_new_documents(
                    run_id=run_id,
                    query_execution_id=query_execution_id,
                    to_enqueue=to_enqueue,
                    state=state,
                )
            except Exception as exc:
                for doi in to_enqueue:
                    await self._record_doi_status(
                        run_id=run_id,
                        query_execution_id=query_execution_id,
                        doi=doi,
                        status=DoiExecutionStatus.FAILED,
                        error_message=str(exc),
                        state=state,
                    )
        except Exception as exc:
            state.query_error_message = str(exc)

        (
            pubmed_result_count,
            doi_resolved_count,
            doi_skipped_exists_count,
            doi_enqueued_count,
            doi_failed_count,
        ) = self._build_query_counts(state=state)

        status = _derive_query_status(
            resolved_count=doi_resolved_count, failed_count=doi_failed_count
        )
        if state.query_error_message is not None:
            status = RunStatus.FAILED
            if doi_resolved_count == 0:
                doi_failed_count = 0

        await self._scheduler_repository.complete_query_execution(
            query_execution_id=query_execution_id,
            status=status,
            pubmed_result_count=pubmed_result_count,
            doi_resolved_count=doi_resolved_count,
            doi_skipped_exists_count=doi_skipped_exists_count,
            doi_enqueued_count=doi_enqueued_count,
            doi_failed_count=doi_failed_count,
            ingest_job_id=state.ingest_job_id,
            error_message=state.query_error_message,
        )

        if status in {RunStatus.SUCCESS, RunStatus.PARTIAL_SUCCESS}:
            await self._scheduler_repository.set_query_last_successful_run_at(
                query_id=query.id, value=now
            )

        return status
