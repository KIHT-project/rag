"""init pubmed scheduler schema

Revision ID: 20260208_0001
Revises:
Create Date: 2026-02-08

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260208_0001"
down_revision = None
branch_labels = None
depends_on = None


RUN_STATUS_VALUES = "'RUNNING','SUCCESS','PARTIAL_SUCCESS','FAILED'"
DOI_STATUS_VALUES = "'SKIPPED_EXISTS','ENQUEUED','INGESTED','FAILED'"
TRIGGER_TYPE_VALUES = "'SCHEDULED','MANUAL'"


SCHEMA = "pubmed_scheduler"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};")

    op.create_table(
        "schema_version",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("key", name="uq_schema_version_key"),
        schema=SCHEMA,
    )

    op.create_table(
        "pubmed_query",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("pubmed_query", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("last_successful_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_pubmed_query"),
        schema=SCHEMA,
    )
    op.create_index("ix_pubmed_query_enabled", "pubmed_query", ["enabled"], unique=False, schema=SCHEMA)
    op.create_index(
        "ix_pubmed_query_last_successful_run_at",
        "pubmed_query",
        ["last_successful_run_at"],
        unique=False,
        schema=SCHEMA,
    )

    op.create_table(
        "scheduler_run",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("trigger_type", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            f"status IN ({RUN_STATUS_VALUES})",
            name="ck_scheduler_run_status",
        ),
        sa.CheckConstraint(
            f"trigger_type IN ({TRIGGER_TYPE_VALUES})",
            name="ck_scheduler_run_trigger_type",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_scheduler_run"),
        schema=SCHEMA,
    )
    op.create_index("ix_scheduler_run_started_at", "scheduler_run", ["started_at"], unique=False, schema=SCHEMA)
    op.create_index(
        "ix_scheduler_run_status_started_at",
        "scheduler_run",
        ["status", "started_at"],
        unique=False,
        schema=SCHEMA,
    )

    op.create_table(
        "query_execution",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("pubmed_result_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("doi_resolved_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "doi_skipped_exists_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("doi_enqueued_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("doi_failed_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("ingest_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            f"status IN ({RUN_STATUS_VALUES})",
            name="ck_query_execution_status",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            [f"{SCHEMA}.scheduler_run.id"],
            ondelete="CASCADE",
            name="fk_query_execution_run_id_scheduler_run",
        ),
        sa.ForeignKeyConstraint(
            ["query_id"],
            [f"{SCHEMA}.pubmed_query.id"],
            ondelete="RESTRICT",
            name="fk_query_execution_query_id_pubmed_query",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_query_execution"),
        sa.UniqueConstraint("run_id", "query_id", name="uq_query_execution_run_query"),
        schema=SCHEMA,
    )
    op.create_index("ix_query_execution_run_id", "query_execution", ["run_id"], unique=False, schema=SCHEMA)
    op.create_index("ix_query_execution_query_id", "query_execution", ["query_id"], unique=False, schema=SCHEMA)
    op.create_index(
        "ix_query_execution_status_started_at",
        "query_execution",
        ["status", "started_at"],
        unique=False,
        schema=SCHEMA,
    )

    op.create_table(
        "query_execution_doi",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("query_execution_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("doi", sa.String(length=512), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            f"status IN ({DOI_STATUS_VALUES})",
            name="ck_query_execution_doi_status",
        ),
        sa.ForeignKeyConstraint(
            ["query_execution_id"],
            [f"{SCHEMA}.query_execution.id"],
            ondelete="CASCADE",
            name="fk_query_execution_doi_query_execution_id",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            [f"{SCHEMA}.scheduler_run.id"],
            ondelete="CASCADE",
            name="fk_query_execution_doi_run_id",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_query_execution_doi"),
        sa.UniqueConstraint(
            "query_execution_id",
            "doi",
            name="uq_query_execution_doi_query_execution_id_doi",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_query_execution_doi_run_id",
        "query_execution_doi",
        ["run_id"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_query_execution_doi_query_execution_id",
        "query_execution_doi",
        ["query_execution_id"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_query_execution_doi_status_created_at",
        "query_execution_doi",
        ["status", "created_at"],
        unique=False,
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_index("ix_query_execution_doi_status_created_at", table_name="query_execution_doi", schema=SCHEMA)
    op.drop_index("ix_query_execution_doi_query_execution_id", table_name="query_execution_doi", schema=SCHEMA)
    op.drop_index("ix_query_execution_doi_run_id", table_name="query_execution_doi", schema=SCHEMA)
    op.drop_table("query_execution_doi", schema=SCHEMA)

    op.drop_index("ix_query_execution_status_started_at", table_name="query_execution", schema=SCHEMA)
    op.drop_index("ix_query_execution_query_id", table_name="query_execution", schema=SCHEMA)
    op.drop_index("ix_query_execution_run_id", table_name="query_execution", schema=SCHEMA)
    op.drop_table("query_execution", schema=SCHEMA)

    op.drop_index("ix_scheduler_run_status_started_at", table_name="scheduler_run", schema=SCHEMA)
    op.drop_index("ix_scheduler_run_started_at", table_name="scheduler_run", schema=SCHEMA)
    op.drop_table("scheduler_run", schema=SCHEMA)

    op.drop_index("ix_pubmed_query_last_successful_run_at", table_name="pubmed_query", schema=SCHEMA)
    op.drop_index("ix_pubmed_query_enabled", table_name="pubmed_query", schema=SCHEMA)
    op.drop_table("pubmed_query", schema=SCHEMA)

    op.drop_table("schema_version", schema=SCHEMA)
