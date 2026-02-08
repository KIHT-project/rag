from __future__ import annotations


def trigger_scheduler_run(client, payload: dict | None = None):
    return client.post("/v1/pubmed/scheduler/run", json=payload)


def get_scheduler_status(client):
    return client.get("/v1/pubmed/scheduler/status")
