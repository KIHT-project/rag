from __future__ import annotations

from typing import TypeAlias

from scheduler_pubmed.src.api.models.schemas import (
    PubMedQuery,
    PubMedQueryCreate,
    PubMedQueryUpdate,
    SchedulerRun,
    SchedulerRunCreated,
    SchedulerRunRequest,
    SchedulerStatus,
    RunDoiResult,
)

CreatePubMedQueryRequest: TypeAlias = PubMedQueryCreate
CreatePubMedQueryResponse: TypeAlias = PubMedQuery

UpdatePubMedQueryRequest: TypeAlias = PubMedQueryUpdate
UpdatePubMedQueryResponse: TypeAlias = PubMedQuery

GetPubMedQueryResponse: TypeAlias = PubMedQuery
ListPubMedQueriesResponse: TypeAlias = list[PubMedQuery]

DisablePubMedQueryResponse: TypeAlias = PubMedQuery
EnablePubMedQueryResponse: TypeAlias = PubMedQuery

GetSchedulerStatusResponse: TypeAlias = SchedulerStatus
RunSchedulerRequest: TypeAlias = SchedulerRunRequest
RunSchedulerResponse: TypeAlias = SchedulerRunCreated
GetSchedulerRunResponse: TypeAlias = SchedulerRun
ListSchedulerRunsResponse: TypeAlias = list[SchedulerRun]
ListRunDoisResponse: TypeAlias = list[RunDoiResult]
