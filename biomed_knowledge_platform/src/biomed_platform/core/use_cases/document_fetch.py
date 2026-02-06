from __future__ import annotations

from dataclasses import dataclass, replace

from biomed_platform.common.utils import compute_body_hash_from_items, normalize_doi
from biomed_platform.core.domains.documents import ContentTextSource, DocumentFetchResponse
from biomed_platform.core.domains.ingestion import IngestBatchCommand, IngestItem
from biomed_platform.core.domains.pubmed import PubMedDocument
from biomed_platform.core.domains.retrieval import Disease, SourceType
from biomed_platform.core.errors.errors import business_error, SystemError
from biomed_platform.core.ports.ingestion import IngestionService
from biomed_platform.core.ports.pubmed import PubMedClient

_DEFAULT_DISEASE = Disease.unknown.value
_THROMBOSIS_TERMS = (
    "thrombosis",
    "thromboembolism",
    "venous thromboembolism",
    "vte",
)
_CANCER_TERMS = (
    "cancer",
    "carcinoma",
    "neoplasm",
    "neoplasms",
    "tumor",
    "tumour",
    "malignancy",
)


@dataclass(slots=True)
class DocumentFetchUseCase:
    pubmed_client: PubMedClient
    ingestion_service: IngestionService | None = None

    async def fetch_one(
        self,
        *,
        request_id: str,
        embedding_model_id: str,
        doi: str | None,
        pmid: str | None,
        ingest_enabled: bool,
    ) -> DocumentFetchResponse:
        if bool(doi) == bool(pmid):
            raise business_error(
                code="validation_error",
                message="Provide exactly one of doi or pmid",
                details={"doi": doi, "pmid": pmid},
            )

        doc = await self.pubmed_client.fetch_document(doi=doi, pmid=pmid)
        if doc is None:
            raise business_error(
                code="not_found",
                message="DOI or PMID not found",
                details={"doi": doi, "pmid": pmid},
            )

        response = self._to_response(
            request_id=request_id,
            doc=doc,
            requested_doi=doi,
            requested_pmid=pmid,
        )

        if ingest_enabled:
            if self.ingestion_service is None:
                raise SystemError(
                    code="service_not_configured",
                    message="Ingestion service not configured",
                    details=None,
                    retryable=False,
                )
            ingest_resp = await self._ingest(
                request_id=request_id,
                embedding_model_id=embedding_model_id,
                doc=doc,
                doi_value=response.doi,
                content_text=response.content_text,
                source_type=response.source_type,
            )
            response = replace(response, ingest=ingest_resp)

        return response

    def build_fetch_response(
        self,
        *,
        request_id: str,
        doc: PubMedDocument,
        requested_doi: str | None,
        requested_pmid: str | None,
    ) -> DocumentFetchResponse:
        return self._to_response(
            request_id=request_id,
            doc=doc,
            requested_doi=requested_doi,
            requested_pmid=requested_pmid,
        )

    def build_ingest_item(
        self,
        *,
        doc: PubMedDocument,
        doi_value: str | None,
        content_text: str,
        source_type: SourceType,
    ) -> IngestItem:
        doi_value = doi_value or doc.doi or ""
        doi_normalized = normalize_doi(doi_value)
        if not doi_normalized:
            pmid_val = doc.pmid or ""
            doi_value = doi_value or f"pmid:{pmid_val}"
            doi_normalized = doi_normalized or doi_value.lower()

        disease = self._resolve_disease(doc)

        return IngestItem(
            doi_original=doi_value,
            doi_normalized=doi_normalized,
            disease=disease,
            source_type=source_type.value,
            content_text=content_text,
            year=doc.year,
            title=doc.title,
            journal=doc.journal,
            authors=tuple(doc.authors or ()),
        )

    def _to_response(
        self,
        *,
        request_id: str,
        doc: PubMedDocument,
        requested_doi: str | None,
        requested_pmid: str | None,
    ) -> DocumentFetchResponse:
        content_text = doc.full_text or doc.abstract or ""
        if not content_text.strip():
            raise business_error(
                code="not_found",
                message="Document text not available",
                details={"doi": doc.doi, "pmid": doc.pmid},
            )

        full_text_available = bool(doc.full_text)
        content_source = (
            ContentTextSource.pmc if full_text_available else ContentTextSource.abstract
        )
        source_type = SourceType.full_text if full_text_available else SourceType.pubmed_abstract

        resolved_doi = doc.doi or requested_doi
        resolved_pmid = doc.pmid or requested_pmid

        return DocumentFetchResponse(
            request_id=request_id,
            doi=resolved_doi,
            pmid=resolved_pmid,
            title=doc.title,
            journal=doc.journal,
            year=doc.year,
            authors=doc.authors,
            source_type=source_type,
            content_text=content_text,
            content_text_source=content_source,
            full_text_available=full_text_available,
            ingest=None,
        )

    async def _ingest(
        self,
        *,
        request_id: str,
        embedding_model_id: str,
        doc: PubMedDocument,
        doi_value: str | None,
        content_text: str,
        source_type: SourceType | None,
    ):
        resolved_source_type = source_type or SourceType.pubmed_abstract
        doi_value = doi_value or doc.doi or ""
        doi_normalized = normalize_doi(doi_value)
        if not doi_normalized:
            pmid_val = doc.pmid or ""
            doi_value = doi_value or f"pmid:{pmid_val}"
            doi_normalized = doi_normalized or doi_value.lower()

        disease = self._resolve_disease(doc)

        item = IngestItem(
            doi_original=doi_value,
            doi_normalized=doi_normalized,
            disease=disease,
            source_type=resolved_source_type.value,
            content_text=content_text,
            year=doc.year,
            title=doc.title,
            journal=doc.journal,
            authors=tuple(doc.authors or ()),
        )

        body_hash = compute_body_hash_from_items(
            effective_embedding_model_id=embedding_model_id,
            items=[item],
        )

        cmd = IngestBatchCommand(
            effective_embedding_model_id=embedding_model_id,
            items=(item,),
            idempotency_key=None,
            body_hash=body_hash,
            correlation_id=request_id,
        )

        if self.ingestion_service is None:
            raise SystemError(
                code="service_not_configured",
                message="Ingestion service not configured",
                details=None,
                retryable=False,
            )

        accepted = await self.ingestion_service.ingest_batch(cmd)
        return accepted

    def _resolve_disease(self, doc: PubMedDocument) -> str:
        texts: list[str] = []
        for val in (doc.title, doc.abstract):
            if isinstance(val, str) and val:
                texts.append(val.lower())

        for term in doc.mesh_terms or []:
            if isinstance(term, str) and term:
                texts.append(term.lower())

        joined = " ".join(texts)
        if any(token in joined for token in _THROMBOSIS_TERMS):
            return Disease.thrombosis.value
        if any(token in joined for token in _CANCER_TERMS):
            return Disease.cancer.value

        return _DEFAULT_DISEASE
