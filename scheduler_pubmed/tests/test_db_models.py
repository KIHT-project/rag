from __future__ import annotations

from db.models.scheduler import (
    PubMedQuery,
    QueryExecution,
    QueryExecutionDoi,
    SchedulerRun,
)
from db.models.schema_version import SchemaVersion


def test_model_tablenames() -> None:
    assert SchemaVersion.__tablename__ == "schema_version"
    assert PubMedQuery.__tablename__ == "pubmed_query"
    assert SchedulerRun.__tablename__ == "scheduler_run"
    assert QueryExecution.__tablename__ == "query_execution"
    assert QueryExecutionDoi.__tablename__ == "query_execution_doi"


def test_query_execution_has_expected_unique_constraint() -> None:
    names = {constraint.name for constraint in QueryExecution.__table__.constraints}
    assert "uq_query_execution_run_query" in names


def test_query_execution_doi_has_expected_unique_constraint() -> None:
    names = {constraint.name for constraint in QueryExecutionDoi.__table__.constraints}
    assert "uq_query_execution_doi_query_execution_id_doi" in names
