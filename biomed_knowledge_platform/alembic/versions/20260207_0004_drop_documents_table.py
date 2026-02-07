"""drop documents table

Revision ID: 20260207_0004
Revises: 20260207_0003
Create Date: 2026-02-07

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260207_0004"
down_revision = "20260207_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())
    if "documents" not in table_names:
        return

    indexes = {idx.get("name") for idx in inspector.get_indexes("documents")}
    if "ix_documents_doi" in indexes:
        op.drop_index("ix_documents_doi", table_name="documents")
    op.drop_table("documents")


def downgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("doi", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=1024), nullable=True),
        sa.Column("authors_json", sa.Text(), nullable=True),
        sa.Column("journal", sa.String(length=512), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_doi", "documents", ["doi"], unique=True)
