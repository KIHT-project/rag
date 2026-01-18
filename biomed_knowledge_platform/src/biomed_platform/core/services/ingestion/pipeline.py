from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.ingestion import IngestItem, VectorPoint
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.ports.ingestion import (
    IngestionPipeline,
    Chunker,
    EmbeddingProvider,
    VectorWriter,
)

log = get_logger(__name__)

_POINT_NAMESPACE = uuid.UUID("b2c2df40-7a1b-4f9a-9e1c-0e7c8a7f6c1a")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _point_id(*, doc_id: str, chunk_index: int) -> str:
    name = f"{doc_id}:{chunk_index}"
    return str(uuid.uuid5(_POINT_NAMESPACE, name))


@dataclass(frozen=True, slots=True)
class DefaultIngestionPipeline(IngestionPipeline):
    chunker: Chunker
    embedder: EmbeddingProvider
    index: VectorWriter

    async def ingest_item(
        self,
        *,
        job_id: str,
        embedding_model_id: str,
        doc_id: str,
        item: IngestItem,
    ) -> None:
        log.info(
            "Ingestion started, job_id=%s, doc_id=%s, embedding_model_id=%s",
            job_id,
            doc_id,
            embedding_model_id,
        )

        text = (item.content_text or "").strip()
        if not text:
            log.error(
                "Empty content_text, job_id=%s, doc_id=%s, doi_original=%s",
                job_id,
                doc_id,
                item.doi_original,
            )
            raise SystemError(
                code="empty_document_text",
                message="Empty content_text in ingestion item",
                details={"doi_original": item.doi_original},
                retryable=False,
            )

        chunks = self.chunker.chunk(text=text)
        if not chunks:
            log.error(
                "Chunking produced no chunks, job_id=%s, doc_id=%s",
                job_id,
                doc_id,
            )
            raise SystemError(
                code="no_chunks",
                message="Chunking produced no chunks",
                details={"doi_original": item.doi_original},
                retryable=False,
            )

        log.debug(
            "Chunking completed, job_id=%s, doc_id=%s, chunk_count=%s",
            job_id,
            doc_id,
            len(chunks),
        )

        vectors = await self.embedder.embed_texts(
            model_id=embedding_model_id,
            texts=[c.text for c in chunks],
        )

        if len(vectors) != len(chunks):
            log.error(
                "Embedding count mismatch, job_id=%s, doc_id=%s, chunks=%s, vectors=%s",
                job_id,
                doc_id,
                len(chunks),
                len(vectors),
            )
            raise SystemError(
                code="embedding_count_mismatch",
                message="Embedding output count mismatch",
                details={"chunks": len(chunks), "vectors": len(vectors)},
                retryable=True,
            )

        vector_size = len(vectors[0]) if vectors else 0
        if vector_size <= 0:
            log.error(
                "Invalid vector size, job_id=%s, doc_id=%s, vector_size=%s",
                job_id,
                doc_id,
                vector_size,
            )
            raise SystemError(
                code="invalid_vector_size",
                message="Invalid vector size",
                details={"vector_size": vector_size},
                retryable=False,
            )

        log.debug(
            "Embeddings generated, job_id=%s, doc_id=%s, vector_size=%s",
            job_id,
            doc_id,
            vector_size,
        )

        await self.index.ensure_collection(
            embedding_model_id=embedding_model_id,
            vector_size=vector_size,
        )

        log.debug(
            "Vector collection ensured, embedding_model_id=%s, vector_size=%s",
            embedding_model_id,
            vector_size,
        )

        if await self.index.exists(embedding_model_id=embedding_model_id, doc_id=doc_id):
            raise SystemError(
                code="duplicate_doc",
                message="Document already indexed",
                details={"doc_id": doc_id, "doi_normalized": item.doi_normalized},
                retryable=False,
            )

        created_at = _utc_now_iso()
        points: list[VectorPoint] = []

        for c, v in zip(chunks, vectors, strict=True):
            points.append(
                VectorPoint(
                    point_id=_point_id(doc_id=doc_id, chunk_index=c.index),
                    vector=v,
                    payload={
                        "job_id": job_id,
                        "doc_id": doc_id,
                        "doi_original": item.doi_original,
                        "doi_normalized": item.doi_normalized,
                        "embedding_model_id": embedding_model_id,
                        "chunk_index": c.index,
                        "chunk_start": c.start,
                        "chunk_end": c.end,
                        "text": c.text,
                        "title": item.title,
                        "journal": item.journal,
                        "year": item.year,
                        "authors": list(item.authors),
                        "disease": item.disease,
                        "source_type": item.source_type,
                        "created_at": created_at,
                    },
                )
            )

        log.debug(
            "Vector points prepared, job_id=%s, doc_id=%s, point_count=%s",
            job_id,
            doc_id,
            len(points),
        )

        await self.index.upsert(
            embedding_model_id=embedding_model_id,
            points=points,
        )

        log.info(
            "Ingestion completed, job_id=%s, doc_id=%s, chunks=%s, embedding_model_id=%s",
            job_id,
            doc_id,
            len(points),
            embedding_model_id,
        )
