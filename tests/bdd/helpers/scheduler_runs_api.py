from __future__ import annotations


def list_scheduler_runs(
    client,
    *,
    status: str | None = None,
    from_at: str | None = None,
    to_at: str | None = None,
):
    params: dict[str, str] = {}
    if status is not None:
        params["status"] = status
    if from_at is not None:
        params["from"] = from_at
    if to_at is not None:
        params["to"] = to_at
    return client.get("/v1/pubmed/runs", params=params)


def get_scheduler_run(client, *, run_id: str):
    return client.get(f"/v1/pubmed/runs/{run_id}")


def list_run_dois(client, *, run_id: str):
    return client.get(f"/v1/pubmed/runs/{run_id}/dois")
