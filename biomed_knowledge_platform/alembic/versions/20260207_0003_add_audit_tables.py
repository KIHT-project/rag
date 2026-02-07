"""add audit tables

Revision ID: 20260207_0003
Revises: 1a92c91b718c
Create Date: 2026-02-07

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260207_0003"
down_revision = "1a92c91b718c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "audit_request",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.BIGINT(), nullable=True),
        sa.Column("http_method", sa.String(length=16), nullable=False),
        sa.Column("path", sa.String(length=512), nullable=False),
        sa.Column("query_string_raw", sa.Text(), nullable=True),
        sa.Column("headers_raw", sa.Text(), nullable=False),
        sa.Column("request_body_raw", sa.Text(), nullable=True),
        sa.Column("response_status_code", sa.Integer(), nullable=True),
        sa.Column("response_headers_raw", sa.Text(), nullable=True),
        sa.Column("response_body_raw", sa.Text(), nullable=True),
        sa.Column("client_ip", sa.String(length=128), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("api_key_id", sa.String(length=256), nullable=True),
        sa.Column("session_id", sa.String(length=256), nullable=True),
        sa.Column("user_id", sa.String(length=256), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error_id", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_request_request_id", "audit_request", ["request_id"], unique=False)
    op.create_index(
        "ix_audit_request_received_at", "audit_request", ["received_at"], unique=False
    )
    op.create_index(
        "ix_audit_request_status_received_at",
        "audit_request",
        ["status", "received_at"],
        unique=False,
    )
    op.create_index(
        "ix_audit_request_path_received_at", "audit_request", ["path", "received_at"], unique=False
    )

    op.create_table(
        "audit_evaluation_run",
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column("run_type", sa.String(length=32), nullable=False),
        sa.Column("trigger_source", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.BIGINT(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("dataset_name", sa.String(length=256), nullable=True),
        sa.Column("dataset_version", sa.String(length=128), nullable=True),
        sa.Column("config_snapshot_raw", sa.Text(), nullable=False),
        sa.Column("model_provider", sa.String(length=128), nullable=False),
        sa.Column("model_name", sa.String(length=256), nullable=False),
        sa.Column("model_params_raw", sa.Text(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("error_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index(
        "ix_audit_eval_run_request_id", "audit_evaluation_run", ["request_id"], unique=False
    )
    op.create_index(
        "ix_audit_eval_run_status_started_at",
        "audit_evaluation_run",
        ["status", "started_at"],
        unique=False,
    )
    op.create_index(
        "ix_audit_eval_run_type_started_at",
        "audit_evaluation_run",
        ["run_type", "started_at"],
        unique=False,
    )
    op.create_index(
        "ix_audit_eval_run_model_started_at",
        "audit_evaluation_run",
        ["model_provider", "model_name", "started_at"],
        unique=False,
    )

    op.create_table(
        "audit_event",
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column("run_id", sa.String(length=128), nullable=True),
        sa.Column("parent_event_id", sa.String(length=64), nullable=True),
        sa.Column("event_type", sa.String(length=128), nullable=False),
        sa.Column("component", sa.String(length=64), nullable=False),
        sa.Column("phase", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("payload_raw", sa.Text(), nullable=True),
        sa.Column("stacktrace_raw", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["audit_evaluation_run.run_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index(
        "ix_audit_event_request_id_created_at",
        "audit_event",
        ["request_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_audit_event_run_id_created_at", "audit_event", ["run_id", "created_at"], unique=False
    )
    op.create_index(
        "ix_audit_event_event_type_created_at",
        "audit_event",
        ["event_type", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_audit_event_status_created_at", "audit_event", ["status", "created_at"], unique=False
    )

    op.create_table(
        "audit_evaluation_artifact",
        sa.Column("artifact_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column("artifact_type", sa.String(length=64), nullable=False),
        sa.Column("artifact_name", sa.String(length=256), nullable=False),
        sa.Column("mime_type", sa.String(length=128), nullable=True),
        sa.Column("content_raw", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.BIGINT(), nullable=False),
        sa.Column("metadata_raw", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["audit_evaluation_run.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("artifact_id"),
    )
    op.create_index(
        "ix_audit_eval_artifact_run_id_created_at",
        "audit_evaluation_artifact",
        ["run_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_audit_eval_artifact_request_id_created_at",
        "audit_evaluation_artifact",
        ["request_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_audit_eval_artifact_type_created_at",
        "audit_evaluation_artifact",
        ["artifact_type", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_audit_eval_artifact_sha256",
        "audit_evaluation_artifact",
        ["content_sha256"],
        unique=False,
    )

    op.create_table(
        "audit_metric_result",
        sa.Column("metric_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column("phase", sa.String(length=128), nullable=False),
        sa.Column("metric_name", sa.String(length=128), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("metric_unit", sa.String(length=64), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("aggregation", sa.String(length=32), nullable=True),
        sa.Column("metadata_raw", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["audit_evaluation_run.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("metric_id"),
    )
    op.create_index(
        "ix_audit_metric_run_id_phase", "audit_metric_result", ["run_id", "phase"], unique=False
    )
    op.create_index(
        "ix_audit_metric_name_created_at",
        "audit_metric_result",
        ["metric_name", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_audit_metric_seed_name", "audit_metric_result", ["seed", "metric_name"], unique=False
    )

    op.create_table(
        "audit_error",
        sa.Column("error_id", sa.String(length=64), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column("run_id", sa.String(length=128), nullable=True),
        sa.Column("event_id", sa.String(length=64), nullable=True),
        sa.Column("exception_class", sa.String(length=256), nullable=False),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("stacktrace_raw", sa.Text(), nullable=False),
        sa.Column("component", sa.String(length=64), nullable=False),
        sa.Column("phase", sa.String(length=128), nullable=True),
        sa.Column("is_fatal", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["audit_event.event_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["run_id"], ["audit_evaluation_run.run_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("error_id"),
    )
    op.create_index(
        "ix_audit_error_request_id_created_at",
        "audit_error",
        ["request_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_audit_error_run_id_created_at", "audit_error", ["run_id", "created_at"], unique=False
    )
    op.create_index(
        "ix_audit_error_exception_created_at",
        "audit_error",
        ["exception_class", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_audit_error_component_created_at",
        "audit_error",
        ["component", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_audit_error_component_created_at", table_name="audit_error")
    op.drop_index("ix_audit_error_exception_created_at", table_name="audit_error")
    op.drop_index("ix_audit_error_run_id_created_at", table_name="audit_error")
    op.drop_index("ix_audit_error_request_id_created_at", table_name="audit_error")
    op.drop_table("audit_error")

    op.drop_index("ix_audit_metric_seed_name", table_name="audit_metric_result")
    op.drop_index("ix_audit_metric_name_created_at", table_name="audit_metric_result")
    op.drop_index("ix_audit_metric_run_id_phase", table_name="audit_metric_result")
    op.drop_table("audit_metric_result")

    op.drop_index("ix_audit_eval_artifact_sha256", table_name="audit_evaluation_artifact")
    op.drop_index(
        "ix_audit_eval_artifact_type_created_at", table_name="audit_evaluation_artifact"
    )
    op.drop_index(
        "ix_audit_eval_artifact_request_id_created_at", table_name="audit_evaluation_artifact"
    )
    op.drop_index("ix_audit_eval_artifact_run_id_created_at", table_name="audit_evaluation_artifact")
    op.drop_table("audit_evaluation_artifact")

    op.drop_index("ix_audit_event_status_created_at", table_name="audit_event")
    op.drop_index("ix_audit_event_event_type_created_at", table_name="audit_event")
    op.drop_index("ix_audit_event_run_id_created_at", table_name="audit_event")
    op.drop_index("ix_audit_event_request_id_created_at", table_name="audit_event")
    op.drop_table("audit_event")

    op.drop_index("ix_audit_eval_run_model_started_at", table_name="audit_evaluation_run")
    op.drop_index("ix_audit_eval_run_type_started_at", table_name="audit_evaluation_run")
    op.drop_index("ix_audit_eval_run_status_started_at", table_name="audit_evaluation_run")
    op.drop_index("ix_audit_eval_run_request_id", table_name="audit_evaluation_run")
    op.drop_table("audit_evaluation_run")

    op.drop_index("ix_audit_request_path_received_at", table_name="audit_request")
    op.drop_index("ix_audit_request_status_received_at", table_name="audit_request")
    op.drop_index("ix_audit_request_received_at", table_name="audit_request")
    op.drop_index("ix_audit_request_request_id", table_name="audit_request")
    op.drop_table("audit_request")
