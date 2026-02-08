from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from scheduler_pubmed.src.api.endpoints import runs as endpoint_mod
from scheduler_pubmed.src.api.error_handlers import install_error_handlers
from scheduler_pubmed.src.core.domains.scheduler import (
    DoiExecutionStatus,
    QueryExecution,
    RunDoiResult,
    RunStatus,
    SchedulerRun,
)
from scheduler_pubmed.src.core.errors.errors import business_error


class _FakeRuntime:
    def __init__(self) -> None:
        self.list_calls: list[tuple[RunStatus | None, datetime | None, datetime | None]] = []
        self.get_calls: list[str] = []
        self.doi_calls: list[str] = []

        self.run = SchedulerRun(
            run_id=uuid4(),
            status=RunStatus.SUCCESS,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            queries=[
                QueryExecution(
                    query_execution_id=uuid4(),
                    query_id=uuid4(),
                    status=RunStatus.SUCCESS,
                    pubmed_result_count=10,
                    doi_resolved_count=5,
                    doi_skipped_exists_count=2,
                    doi_enqueued_count=3,
                    doi_failed_count=0,
                    ingest_job_id=None,
                )
            ],
        )
        self.run_dois = [
            RunDoiResult(
                doi="10.1000/a",
                status=DoiExecutionStatus.INGESTED,
                error_message=None,
            )
        ]

    async def list_runs(
        self,
        *,
        status: RunStatus | None,
        from_at: datetime | None,
        to_at: datetime | None,
    ) -> list[SchedulerRun]:
        self.list_calls.append((status, from_at, to_at))
        return [self.run]

    async def get_run(self, *, run_id):
        self.get_calls.append(str(run_id))
        return self.run

    async def list_run_dois(self, *, run_id):
        self.doi_calls.append(str(run_id))
        return self.run_dois


@pytest.fixture
def runs_client(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(endpoint_mod.router)

    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(endpoint_mod, "_get_runtime", lambda _request: fake_runtime)

    with TestClient(app) as client:
        yield client, fake_runtime


def test_list_scheduler_runs_returns_200(runs_client) -> None:
    client, runtime = runs_client

    response = client.get(
        "/v1/pubmed/runs",
        params={
            "status": "SUCCESS",
            "from": "2026-02-08T00:00:00Z",
            "to": "2026-02-08T23:59:00Z",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert body[0]["run_id"] == str(runtime.run.run_id)
    assert body[0]["queries"][0]["status"] == "SUCCESS"
    assert runtime.list_calls[0][0] == RunStatus.SUCCESS


def test_get_scheduler_run_returns_200(runs_client) -> None:
    client, runtime = runs_client

    response = client.get(f"/v1/pubmed/runs/{runtime.run.run_id}")

    assert response.status_code == 200
    assert response.json()["run_id"] == str(runtime.run.run_id)
    assert runtime.get_calls == [str(runtime.run.run_id)]


def test_list_run_dois_returns_200(runs_client) -> None:
    client, runtime = runs_client

    response = client.get(f"/v1/pubmed/runs/{runtime.run.run_id}/dois")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert body[0]["doi"] == "10.1000/a"
    assert body[0]["status"] == "INGESTED"
    assert runtime.doi_calls == [str(runtime.run.run_id)]


def test_get_scheduler_run_not_found_returns_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(endpoint_mod.router)

    class _NotFoundRuntime:
        async def get_run(self, *, run_id):  # noqa: ARG002
            raise business_error(
                code="not_found",
                message="Scheduler run not found",
                details=None,
            )

    monkeypatch.setattr(endpoint_mod, "_get_runtime", lambda _request: _NotFoundRuntime())

    with TestClient(app) as client:
        response = client.get(f"/v1/pubmed/runs/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_runs_endpoints_have_runs_tag() -> None:
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(endpoint_mod.router)

    schema = app.openapi()
    assert schema["paths"]["/v1/pubmed/runs"]["get"]["tags"] == ["Runs"]
    assert schema["paths"]["/v1/pubmed/runs/{runId}"]["get"]["tags"] == ["Runs"]
    assert schema["paths"]["/v1/pubmed/runs/{runId}/dois"]["get"]["tags"] == ["Runs"]
