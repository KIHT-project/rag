from __future__ import annotations

import json
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from biomed_platform.db.models.audit import (
    AuditError,
    AuditEvaluationArtifact,
    AuditEvaluationRun,
    AuditEvent,
    AuditMetricResult,
    AuditRequest,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _exc_stacktrace(exc: BaseException | None) -> str:
    if exc is None:
        return ""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


class PostgresAuditService:
    def __init__(self, *, session_maker: async_sessionmaker[AsyncSession]) -> None:
        self._session_maker = session_maker

    async def create_request(
        self,
        *,
        request_id: str,
        method: str,
        path: str,
        query_string_raw: str,
        headers: dict[str, str],
        body_raw: str | None,
        client_ip: str | None,
        user_agent: str | None,
        api_key_id: str | None,
        session_id: str | None,
        user_id: str | None,
    ) -> None:
        async with self._session_maker() as session:
            rec = AuditRequest(
                request_id=request_id,
                received_at=_utc_now(),
                http_method=method,
                path=path,
                query_string_raw=query_string_raw or None,
                headers_raw=_dumps(headers),
                request_body_raw=body_raw,
                client_ip=client_ip,
                user_agent=user_agent,
                api_key_id=api_key_id,
                session_id=session_id,
                user_id=user_id,
                status="SUCCESS",
            )
            session.add(rec)
            await session.commit()

    async def complete_request(
        self,
        *,
        request_id: str,
        response_status_code: int,
        response_headers: dict[str, str],
        response_body_raw: str | None,
        completed_at: datetime,
    ) -> None:
        async with self._session_maker() as session:
            q = (
                select(AuditRequest)
                .where(AuditRequest.request_id == request_id)
                .order_by(desc(AuditRequest.id))
                .limit(1)
            )
            row = (await session.execute(q)).scalar_one()
            duration = max(
                0,
                int((completed_at - row.received_at).total_seconds() * 1000),
            )
            row.completed_at = completed_at
            row.duration_ms = duration
            row.response_status_code = int(response_status_code)
            row.response_headers_raw = _dumps(response_headers)
            row.response_body_raw = response_body_raw
            row.status = "ERROR" if int(response_status_code) >= 400 else "SUCCESS"
            await session.commit()

    async def create_event(
        self,
        *,
        request_id: str | None,
        run_id: str | None,
        event_type: str,
        component: str,
        status: str,
        message: str | None = None,
        phase: str | None = None,
        payload: Any | None = None,
        stacktrace_raw: str | None = None,
    ) -> str:
        event_id = uuid.uuid4().hex
        async with self._session_maker() as session:
            rec = AuditEvent(
                event_id=event_id,
                request_id=request_id,
                run_id=run_id,
                parent_event_id=None,
                event_type=event_type,
                component=component,
                phase=phase,
                status=status,
                message=message,
                payload_raw=_dumps(payload) if payload is not None else None,
                stacktrace_raw=stacktrace_raw,
                created_at=_utc_now(),
            )
            session.add(rec)
            await session.commit()
        return event_id

    async def create_error(
        self,
        *,
        request_id: str | None,
        run_id: str | None,
        event_id: str | None,
        exc: BaseException,
        component: str,
        phase: str | None,
        error_code: str | None = None,
        is_fatal: bool = True,
    ) -> str:
        error_id = uuid.uuid4().hex
        stacktrace_raw = _exc_stacktrace(exc)
        async with self._session_maker() as session:
            rec = AuditError(
                error_id=error_id,
                request_id=request_id,
                run_id=run_id,
                event_id=event_id,
                exception_class=type(exc).__name__,
                error_code=error_code,
                message=str(exc),
                stacktrace_raw=stacktrace_raw,
                component=component,
                phase=phase,
                is_fatal=is_fatal,
                created_at=_utc_now(),
            )
            session.add(rec)
            await session.commit()
        return error_id

    async def mark_request_error(self, *, request_id: str, error_id: str) -> None:
        async with self._session_maker() as session:
            q = (
                select(AuditRequest)
                .where(AuditRequest.request_id == request_id)
                .order_by(desc(AuditRequest.id))
                .limit(1)
            )
            row = (await session.execute(q)).scalar_one_or_none()
            if row is None:
                return
            row.status = "ERROR"
            row.error_id = error_id
            await session.commit()

    async def create_evaluation_run(
        self,
        *,
        run_id: str,
        request_id: str | None,
        run_type: str,
        trigger_source: str,
        status: str,
        dataset_name: str | None,
        dataset_version: str | None,
        config_snapshot: Any,
        model_provider: str,
        model_name: str,
        model_params: Any,
        seed: int | None,
    ) -> None:
        now = _utc_now()
        async with self._session_maker() as session:
            rec = AuditEvaluationRun(
                run_id=run_id,
                request_id=request_id,
                run_type=run_type,
                trigger_source=trigger_source,
                started_at=now,
                finished_at=None,
                duration_ms=None,
                status=status,
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                config_snapshot_raw=_dumps(config_snapshot),
                model_provider=model_provider,
                model_name=model_name,
                model_params_raw=_dumps(model_params),
                seed=seed,
                error_id=None,
                created_at=now,
                updated_at=now,
            )
            session.add(rec)
            await session.commit()

    async def complete_evaluation_run(
        self,
        *,
        run_id: str,
        status: str,
        error_id: str | None = None,
    ) -> None:
        now = _utc_now()
        async with self._session_maker() as session:
            q = select(AuditEvaluationRun).where(AuditEvaluationRun.run_id == run_id).limit(1)
            row = (await session.execute(q)).scalar_one()
            row.finished_at = now
            row.duration_ms = max(0, int((now - row.started_at).total_seconds() * 1000))
            row.status = status
            row.error_id = error_id
            row.updated_at = now
            await session.commit()

    async def create_artifact(
        self,
        *,
        run_id: str,
        request_id: str | None,
        artifact_type: str,
        artifact_name: str,
        mime_type: str | None,
        content_raw: str,
        content_sha256: str,
        metadata: Any | None = None,
    ) -> str:
        artifact_id = uuid.uuid4().hex
        async with self._session_maker() as session:
            rec = AuditEvaluationArtifact(
                artifact_id=artifact_id,
                run_id=run_id,
                request_id=request_id,
                artifact_type=artifact_type,
                artifact_name=artifact_name,
                mime_type=mime_type,
                content_raw=content_raw,
                content_sha256=content_sha256,
                size_bytes=len(content_raw.encode("utf-8")),
                metadata_raw=_dumps(metadata) if metadata is not None else None,
                created_at=_utc_now(),
            )
            session.add(rec)
            await session.commit()
        return artifact_id

    async def create_metric(
        self,
        *,
        run_id: str,
        request_id: str | None,
        phase: str,
        metric_name: str,
        metric_value: float,
        metric_unit: str | None = None,
        seed: int | None = None,
        aggregation: str | None = None,
        metadata: Any | None = None,
    ) -> str:
        metric_id = uuid.uuid4().hex
        async with self._session_maker() as session:
            rec = AuditMetricResult(
                metric_id=metric_id,
                run_id=run_id,
                request_id=request_id,
                phase=phase,
                metric_name=metric_name,
                metric_value=float(metric_value),
                metric_unit=metric_unit,
                seed=seed,
                aggregation=aggregation,
                metadata_raw=_dumps(metadata) if metadata is not None else None,
                created_at=_utc_now(),
            )
            session.add(rec)
            await session.commit()
        return metric_id
