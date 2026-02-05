from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from biomed_platform.api.models.generated import schemas
from biomed_platform.core.domains.ingestion import IngestBatchAccepted, JobState
from biomed_platform.core.domains.pubmed import PubMedDocument
from biomed_platform.core.errors.errors import BusinessError
from biomed_platform.core.use_cases.document_fetch import DocumentFetchUseCase


class _PubMedClient:
    def __init__(self, doc: PubMedDocument | None):
        self._doc = doc

    async def fetch_document(self, *, doi: str | None, pmid: str | None):
        return self._doc


@pytest.mark.anyio
async def test_fetch_one_full_text_ingests() -> None:
    doc = PubMedDocument(
        doi="10.1000/xyz123",
        pmid="123",
        pmcid="PMC1",
        title="Title",
        journal="Journal",
        year=2024,
        authors=["A Author"],
        mesh_terms=["Thrombosis"],
        abstract="Abstract text",
        full_text="Full text",
    )
    pubmed = _PubMedClient(doc)

    ingestion = AsyncMock()
    ingestion.ingest_batch = AsyncMock(
        return_value=IngestBatchAccepted(job_id="job", state=JobState.queued)
    )

    use_case = DocumentFetchUseCase(pubmed_client=pubmed, ingestion_service=ingestion)

    res = await use_case.fetch_one(
        request_id="rid",
        embedding_model_id="m",
        doi="10.1000/xyz123",
        pmid=None,
        ingest_enabled=True,
    )

    assert res.content_text == "Full text"
    assert res.content_text_source == schemas.ContentTextSource.pmc
    assert res.full_text_available is True
    assert res.ingest is not None
    cmd = ingestion.ingest_batch.call_args.args[0]
    assert cmd.items[0].disease == schemas.Disease.thrombosis.value


@pytest.mark.anyio
async def test_fetch_one_infers_cancer() -> None:
    doc = PubMedDocument(
        doi="10.1000/xyz999",
        pmid="999",
        pmcid=None,
        title="Breast cancer: biomarkers and treatment",
        journal="Journal",
        year=2024,
        authors=["A Author"],
        mesh_terms=["Breast Neoplasms"],
        abstract="Breast cancer remains a leading cause of mortality.",
        full_text=None,
    )
    pubmed = _PubMedClient(doc)

    ingestion = AsyncMock()
    ingestion.ingest_batch = AsyncMock(
        return_value=IngestBatchAccepted(job_id="job", state=JobState.queued)
    )

    use_case = DocumentFetchUseCase(pubmed_client=pubmed, ingestion_service=ingestion)

    res = await use_case.fetch_one(
        request_id="rid",
        embedding_model_id="m",
        doi="10.1000/xyz999",
        pmid=None,
        ingest_enabled=True,
    )

    assert res.content_text_source == schemas.ContentTextSource.abstract
    cmd = ingestion.ingest_batch.call_args.args[0]
    assert cmd.items[0].disease == schemas.Disease.cancer.value


@pytest.mark.anyio
async def test_fetch_one_abstract_no_ingest() -> None:
    doc = PubMedDocument(
        doi="10.1000/xyz123",
        pmid="123",
        pmcid=None,
        title="Title",
        journal="Journal",
        year=2024,
        authors=None,
        mesh_terms=None,
        abstract="Abstract text",
        full_text=None,
    )
    pubmed = _PubMedClient(doc)

    use_case = DocumentFetchUseCase(pubmed_client=pubmed, ingestion_service=None)

    res = await use_case.fetch_one(
        request_id="rid",
        embedding_model_id="m",
        doi="10.1000/xyz123",
        pmid=None,
        ingest_enabled=False,
    )

    assert res.content_text == "Abstract text"
    assert res.content_text_source == schemas.ContentTextSource.abstract
    assert res.full_text_available is False
    assert res.ingest is None


@pytest.mark.anyio
async def test_fetch_one_invalid_input_raises() -> None:
    pubmed = _PubMedClient(None)
    use_case = DocumentFetchUseCase(pubmed_client=pubmed, ingestion_service=None)

    with pytest.raises(BusinessError) as exc:
        await use_case.fetch_one(
            request_id="rid",
            embedding_model_id="m",
            doi=None,
            pmid=None,
            ingest_enabled=False,
        )

    assert exc.value.code == "validation_error"


@pytest.mark.anyio
async def test_fetch_one_not_found_raises() -> None:
    pubmed = _PubMedClient(None)
    use_case = DocumentFetchUseCase(pubmed_client=pubmed, ingestion_service=None)

    with pytest.raises(BusinessError) as exc:
        await use_case.fetch_one(
            request_id="rid",
            embedding_model_id="m",
            doi="10.1/x",
            pmid=None,
            ingest_enabled=False,
        )

    assert exc.value.code == "not_found"
