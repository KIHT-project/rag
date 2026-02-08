from __future__ import annotations

import hashlib
import json
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class EvaluationPostgresAudit:
    def __init__(self, *, dsn: str) -> None:
        self._dsn = dsn
        self._conn: asyncpg.Connection | None = None

    async def start(self) -> None:
        self._conn = await asyncpg.connect(self._dsn, ssl=False)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def _require_conn(self) -> asyncpg.Connection:
        if self._conn is None:
            raise RuntimeError("EvaluationPostgresAudit is not started")
        return self._conn

    async def create_run(
        self,
        *,
        run_id: str,
        request_id: str | None,
        run_type: str,
        trigger_source: str,
        dataset_name: str | None,
        dataset_version: str | None,
        config_snapshot: dict[str, Any],
        model_provider: str,
        model_name: str,
        model_params: dict[str, Any],
        seed: int | None,
    ) -> None:
        now = _utc_now()
        conn = self._require_conn()
        await conn.execute(
            """
            INSERT INTO core_db.audit_evaluation_run (
                run_id, request_id, run_type, trigger_source, started_at, finished_at, duration_ms,
                status, dataset_name, dataset_version, config_snapshot_raw, model_provider,
                model_name, model_params_raw, seed, error_id, created_at, updated_at
            )
            VALUES (
                $1, $2, $3, $4, $5, NULL, NULL, 'RUNNING', $6, $7, $8, $9, $10, $11, $12, NULL, $5, $5
            )
            """,
            run_id,
            request_id,
            run_type,
            trigger_source,
            now,
            dataset_name,
            dataset_version,
            _json(config_snapshot),
            model_provider,
            model_name,
            _json(model_params),
            seed,
        )

    async def complete_run(self, *, run_id: str, status: str, error_id: str | None = None) -> None:
        now = _utc_now()
        conn = self._require_conn()
        row = await conn.fetchrow(
            "SELECT started_at FROM core_db.audit_evaluation_run WHERE run_id = $1",
            run_id,
        )
        started_at = row["started_at"] if row is not None else now
        duration_ms = max(0, int((now - started_at).total_seconds() * 1000))
        await conn.execute(
            """
            UPDATE core_db.audit_evaluation_run
            SET finished_at = $2,
                duration_ms = $3,
                status = $4,
                error_id = $5,
                updated_at = $2
            WHERE run_id = $1
            """,
            run_id,
            now,
            duration_ms,
            status,
            error_id,
        )

    async def create_event(
        self,
        *,
        run_id: str,
        request_id: str | None,
        event_type: str,
        status: str,
        message: str | None = None,
        phase: str | None = None,
        payload: Any | None = None,
        stacktrace_raw: str | None = None,
    ) -> str:
        event_id = uuid.uuid4().hex
        conn = self._require_conn()
        await conn.execute(
            """
            INSERT INTO core_db.audit_event (
                event_id, request_id, run_id, parent_event_id, event_type, component, phase, status,
                message, payload_raw, stacktrace_raw, created_at
            )
            VALUES ($1, $2, $3, NULL, $4, 'EVALUATION', $5, $6, $7, $8, $9, $10)
            """,
            event_id,
            request_id,
            run_id,
            event_type,
            phase,
            status,
            message,
            _json(payload) if payload is not None else None,
            stacktrace_raw,
            _utc_now(),
        )
        return event_id

    async def create_error(
        self,
        *,
        run_id: str,
        request_id: str | None,
        event_id: str | None,
        exc: Exception,
        phase: str | None,
        error_code: str | None = None,
    ) -> str:
        error_id = uuid.uuid4().hex
        stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        conn = self._require_conn()
        await conn.execute(
            """
            INSERT INTO core_db.audit_error (
                error_id, request_id, run_id, event_id, exception_class, error_code, message,
                stacktrace_raw, component, phase, is_fatal, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'EVALUATION', $9, true, $10)
            """,
            error_id,
            request_id,
            run_id,
            event_id,
            type(exc).__name__,
            error_code,
            str(exc),
            stack,
            phase,
            _utc_now(),
        )
        return error_id

    async def create_artifact(
        self,
        *,
        run_id: str,
        request_id: str | None,
        artifact_type: str,
        artifact_name: str,
        mime_type: str | None,
        content_raw: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        artifact_id = uuid.uuid4().hex
        conn = self._require_conn()
        await conn.execute(
            """
            INSERT INTO core_db.audit_evaluation_artifact (
                artifact_id, run_id, request_id, artifact_type, artifact_name, mime_type, content_raw,
                content_sha256, size_bytes, metadata_raw, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            artifact_id,
            run_id,
            request_id,
            artifact_type,
            artifact_name,
            mime_type,
            content_raw,
            _sha256_text(content_raw),
            len(content_raw.encode("utf-8")),
            _json(metadata) if metadata is not None else None,
            _utc_now(),
        )
        return artifact_id

    async def create_metric(
        self,
        *,
        run_id: str,
        request_id: str | None,
        phase: str,
        metric_name: str,
        metric_value: float,
        seed: int | None,
        aggregation: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        metric_id = uuid.uuid4().hex
        conn = self._require_conn()
        await conn.execute(
            """
            INSERT INTO core_db.audit_metric_result (
                metric_id, run_id, request_id, phase, metric_name, metric_value,
                metric_unit, seed, aggregation, metadata_raw, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, NULL, $7, $8, $9, $10)
            """,
            metric_id,
            run_id,
            request_id,
            phase,
            metric_name,
            float(metric_value),
            seed,
            aggregation,
            _json(metadata) if metadata is not None else None,
            _utc_now(),
        )
        return metric_id
