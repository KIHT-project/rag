from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from scheduler_pubmed.src.db.base import Base


class PubMedQuery(Base):
    __tablename__ = "pubmed_query"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        nullable=False,
        server_default=text("gen_random_uuid()"),
    )
    pubmed_query: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    last_successful_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )

    __table_args__ = (
        Index("ix_pubmed_query_enabled", "enabled"),
        Index("ix_pubmed_query_last_successful_run_at", "last_successful_run_at"),
    )


class SchedulerRun(Base):
    __tablename__ = "scheduler_run"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        nullable=False,
        server_default=text("gen_random_uuid()"),
    )
    trigger_type: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('RUNNING','SUCCESS','PARTIAL_SUCCESS','FAILED')",
            name="ck_scheduler_run_status",
        ),
        CheckConstraint(
            "trigger_type IN ('SCHEDULED','MANUAL')",
            name="ck_scheduler_run_trigger_type",
        ),
        Index("ix_scheduler_run_started_at", "started_at"),
        Index("ix_scheduler_run_status_started_at", "status", "started_at"),
    )


class QueryExecution(Base):
    __tablename__ = "query_execution"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        nullable=False,
        server_default=text("gen_random_uuid()"),
    )
    run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("scheduler_run.id", ondelete="CASCADE"),
        nullable=False,
    )
    query_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("pubmed_query.id", ondelete="RESTRICT"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    pubmed_result_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    doi_resolved_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    doi_skipped_exists_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    doi_enqueued_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    doi_failed_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    ingest_job_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('RUNNING','SUCCESS','PARTIAL_SUCCESS','FAILED')",
            name="ck_query_execution_status",
        ),
        UniqueConstraint("run_id", "query_id", name="uq_query_execution_run_query"),
        Index("ix_query_execution_run_id", "run_id"),
        Index("ix_query_execution_query_id", "query_id"),
        Index("ix_query_execution_status_started_at", "status", "started_at"),
    )


class QueryExecutionDoi(Base):
    __tablename__ = "query_execution_doi"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        nullable=False,
        server_default=text("gen_random_uuid()"),
    )
    query_execution_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("query_execution.id", ondelete="CASCADE"),
        nullable=False,
    )
    run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("scheduler_run.id", ondelete="CASCADE"),
        nullable=False,
    )
    doi: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('SKIPPED_EXISTS','ENQUEUED','INGESTED','FAILED')",
            name="ck_query_execution_doi_status",
        ),
        UniqueConstraint(
            "query_execution_id",
            "doi",
            name="uq_query_execution_doi_query_execution_id_doi",
        ),
        Index("ix_query_execution_doi_run_id", "run_id"),
        Index("ix_query_execution_doi_query_execution_id", "query_execution_id"),
        Index("ix_query_execution_doi_status_created_at", "status", "created_at"),
    )
