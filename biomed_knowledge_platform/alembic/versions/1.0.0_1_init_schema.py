"""init schema

Revision ID: 20260120_0001
Revises: 
Create Date: 2026-01-20

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260120_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS core_db;")

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
        schema="core_db",
    )


def downgrade() -> None:
    op.drop_table("schema_version", schema="core_db")

