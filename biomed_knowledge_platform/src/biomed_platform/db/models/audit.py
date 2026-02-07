from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BIGINT,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from biomed_platform.db.base import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AuditRequest(Base):
    __tablename__ = "audit_request"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(BIGINT, nullable=True)

    http_method: Mapped[str] = mapped_column(String(16), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    query_string_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    headers_raw: Mapped[str] = mapped_column(Text, nullable=False)
    request_body_raw: Mapped[str | None] = mapped_column(Text, nullable=True)

    response_status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_headers_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_body_raw: Mapped[str | None] = mapped_column(Text, nullable=True)

    client_ip: Mapped[str | None] = mapped_column(String(128), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    api_key_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(256), nullable=True)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default="SUCCESS")
    error_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        Index("ix_audit_request_request_id", "request_id"),
        Index("ix_audit_request_received_at", "received_at"),
        Index("ix_audit_request_status_received_at", "status", "received_at"),
        Index("ix_audit_request_path_received_at", "path", "received_at"),
    )


class AuditEvaluationRun(Base):
    __tablename__ = "audit_evaluation_run"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    run_type: Mapped[str] = mapped_column(String(32), nullable=False)
    trigger_source: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(BIGINT, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    dataset_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    dataset_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    config_snapshot_raw: Mapped[str] = mapped_column(Text, nullable=False)
    model_provider: Mapped[str] = mapped_column(String(128), nullable=False)
    model_name: Mapped[str] = mapped_column(String(256), nullable=False)
    model_params_raw: Mapped[str] = mapped_column(Text, nullable=False)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now
    )

    __table_args__ = (
        Index("ix_audit_eval_run_request_id", "request_id"),
        Index("ix_audit_eval_run_status_started_at", "status", "started_at"),
        Index("ix_audit_eval_run_type_started_at", "run_type", "started_at"),
        Index(
            "ix_audit_eval_run_model_started_at",
            "model_provider",
            "model_name",
            "started_at",
        ),
    )


class AuditEvent(Base):
    __tablename__ = "audit_event"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    run_id: Mapped[str | None] = mapped_column(
        String(128),
        ForeignKey("audit_evaluation_run.run_id", ondelete="SET NULL"),
        nullable=True,
    )
    parent_event_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    component: Mapped[str] = mapped_column(String(64), nullable=False)
    phase: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    stacktrace_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now
    )

    __table_args__ = (
        Index("ix_audit_event_request_id_created_at", "request_id", "created_at"),
        Index("ix_audit_event_run_id_created_at", "run_id", "created_at"),
        Index("ix_audit_event_event_type_created_at", "event_type", "created_at"),
        Index("ix_audit_event_status_created_at", "status", "created_at"),
    )


class AuditEvaluationArtifact(Base):
    __tablename__ = "audit_evaluation_artifact"

    artifact_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("audit_evaluation_run.run_id", ondelete="CASCADE"), nullable=False
    )
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_name: Mapped[str] = mapped_column(String(256), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    content_raw: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BIGINT, nullable=False)
    metadata_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now
    )

    __table_args__ = (
        Index("ix_audit_eval_artifact_run_id_created_at", "run_id", "created_at"),
        Index("ix_audit_eval_artifact_request_id_created_at", "request_id", "created_at"),
        Index("ix_audit_eval_artifact_type_created_at", "artifact_type", "created_at"),
        Index("ix_audit_eval_artifact_sha256", "content_sha256"),
    )


class AuditMetricResult(Base):
    __tablename__ = "audit_metric_result"

    metric_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("audit_evaluation_run.run_id", ondelete="CASCADE"), nullable=False
    )
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    phase: Mapped[str] = mapped_column(String(128), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    metric_unit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    aggregation: Mapped[str | None] = mapped_column(String(32), nullable=True)
    metadata_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now
    )

    __table_args__ = (
        Index("ix_audit_metric_run_id_phase", "run_id", "phase"),
        Index("ix_audit_metric_name_created_at", "metric_name", "created_at"),
        Index("ix_audit_metric_seed_name", "seed", "metric_name"),
    )


class AuditError(Base):
    __tablename__ = "audit_error"

    error_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    run_id: Mapped[str | None] = mapped_column(
        String(128),
        ForeignKey("audit_evaluation_run.run_id", ondelete="SET NULL"),
        nullable=True,
    )
    event_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("audit_event.event_id", ondelete="SET NULL"),
        nullable=True,
    )
    exception_class: Mapped[str] = mapped_column(String(256), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    stacktrace_raw: Mapped[str] = mapped_column(Text, nullable=False)
    component: Mapped[str] = mapped_column(String(64), nullable=False)
    phase: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_fatal: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now
    )

    __table_args__ = (
        Index("ix_audit_error_request_id_created_at", "request_id", "created_at"),
        Index("ix_audit_error_run_id_created_at", "run_id", "created_at"),
        Index("ix_audit_error_exception_created_at", "exception_class", "created_at"),
        Index("ix_audit_error_component_created_at", "component", "created_at"),
    )
