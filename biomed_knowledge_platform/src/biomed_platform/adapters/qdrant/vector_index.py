from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, TypeVar

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct,
    VectorParams,
    Range,
)

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.ingestion import VectorPoint
from biomed_platform.core.domains.retrieval import ChunkCandidate, SourceType, VectorSearchHit
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.ports.ingestion import VectorWriter

log = get_logger(__name__)

_T = TypeVar("_T")


def parse_distance(value: str) -> Distance:
    v = (value or "").strip().upper()
    log.debug("Parsing qdrant distance, raw_value=%s", value)

    if v == "COSINE":
        return Distance.COSINE
    if v == "DOT":
        return Distance.DOT
    if v == "EUCLID":
        return Distance.EUCLID

    log.error("Invalid qdrant distance value, value=%s", value)
    raise SystemError(
        code="invalid_qdrant_distance",
        message="Invalid qdrant distance in qdrant.yaml",
        details={"distance": value},
        retryable=False,
    )


@dataclass(slots=True)
class QdrantVectorIndex(VectorWriter):
    client: QdrantClient
    collection_name_prefix: str
    distance: Distance

    _lock_global: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _locks_by_collection: dict[str, asyncio.Lock] = field(
        default_factory=dict, init=False, repr=False
    )

    async def _call_blocking(self, fn: Callable[[], _T]) -> _T:
        try:
            return await asyncio.to_thread(fn)
        except Exception:
            log.exception("Qdrant blocking call failed, op=%s", getattr(fn, "__name__", "unknown"))
            raise

    def _collection_name(self, *, embedding_model_id: str) -> str:
        safe = (embedding_model_id or "").replace("/", "_").replace(":", "_")
        prefix = (self.collection_name_prefix or "docs").strip() or "docs"
        name = f"{prefix}_{safe}"

        log.debug(
            "Resolved collection name, embedding_model_id=%s, collection_name=%s",
            embedding_model_id,
            name,
        )
        return name

    async def _collection_lock(self, *, name: str) -> asyncio.Lock:
        async with self._lock_global:
            lock = self._locks_by_collection.get(name)
            if lock is None:
                lock = asyncio.Lock()
                self._locks_by_collection[name] = lock
            return lock

    def _validate_vector_size(self, *, embedding_model_id: str, vector_size: int) -> None:
        if vector_size > 0:
            return

        log.error(
            "Invalid vector size for collection ensure, embedding_model_id=%s, vector_size=%s",
            embedding_model_id,
            vector_size,
        )
        raise SystemError(
            code="invalid_vector_size",
            message="Vector size must be positive",
            details={"vector_size": vector_size},
            retryable=False,
        )

    async def _get_collections(self, *, name: str) -> list[object]:
        try:

            def _get():
                return self.client.get_collections().collections

            return await self._call_blocking(_get)

        except Exception as exc:
            log.exception("Qdrant get_collections failed, name=%s", name)
            raise SystemError(
                code="qdrant_unavailable",
                message="Qdrant is unavailable",
                details={"operation": "get_collections"},
                retryable=True,
            ) from exc

    def _collection_exists(self, *, existing: Sequence[object], name: str) -> bool:
        return any(getattr(c, "name", None) == name for c in existing)

    async def _create_collection(self, *, name: str, vector_size: int) -> None:
        log.info(
            "Creating qdrant collection, name=%s, vector_size=%s, distance=%s",
            name,
            vector_size,
            self.distance.name,
        )

        try:

            def _create() -> None:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=vector_size, distance=self.distance),
                )

            await self._call_blocking(_create)

        except UnexpectedResponse as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 409:
                log.info("Qdrant collection already exists after concurrent create, name=%s", name)
                return

            log.exception("Qdrant create_collection failed, name=%s", name)
            raise SystemError(
                code="qdrant_create_collection_failed",
                message="Failed to create qdrant collection",
                details={"collection": name, "status_code": status_code},
                retryable=True,
            ) from exc

        except Exception as exc:
            log.exception("Qdrant create_collection failed, name=%s", name)
            raise SystemError(
                code="qdrant_create_collection_failed",
                message="Failed to create qdrant collection",
                details={"collection": name},
                retryable=True,
            ) from exc

        log.info("Qdrant collection created successfully, name=%s", name)

    async def ensure_collection(self, *, embedding_model_id: str, vector_size: int) -> None:
        self._validate_vector_size(embedding_model_id=embedding_model_id, vector_size=vector_size)

        name = self._collection_name(embedding_model_id=embedding_model_id)
        lock = await self._collection_lock(name=name)

        async with lock:
            log.debug("Ensuring qdrant collection, name=%s, vector_size=%s", name, vector_size)

            existing = await self._get_collections(name=name)
            if self._collection_exists(existing=existing, name=name):
                log.debug("Qdrant collection already exists, name=%s", name)
                return

            await self._create_collection(name=name, vector_size=vector_size)

    async def upsert(self, *, embedding_model_id: str, points: Sequence[VectorPoint]) -> None:
        if not points:
            log.debug(
                "Upsert skipped, no points provided, embedding_model_id=%s",
                embedding_model_id,
            )
            return

        name = self._collection_name(embedding_model_id=embedding_model_id)

        log.debug("Preparing qdrant upsert, name=%s, point_count=%s", name, len(points))

        qpoints = [
            PointStruct(
                id=p.point_id,
                vector=list(p.vector),
                payload=dict(p.payload),
            )
            for p in points
        ]

        try:

            def _upsert() -> None:
                self.client.upsert(
                    collection_name=name,
                    points=qpoints,
                    wait=True,
                )

            await self._call_blocking(_upsert)

        except Exception as exc:
            log.exception("Qdrant upsert failed, name=%s, points=%s", name, len(qpoints))
            raise SystemError(
                code="qdrant_upsert_failed",
                message="Failed to upsert vectors to qdrant",
                details={"collection": name, "points": len(qpoints)},
                retryable=True,
            ) from exc

        log.info("Qdrant upsert completed, name=%s, points=%s", name, len(qpoints))

    async def exists(self, *, embedding_model_id: str, doc_id: str) -> bool:
        if not doc_id:
            return False

        name = self._collection_name(embedding_model_id=embedding_model_id)

        try:

            def _count() -> int:
                res = self.client.count(
                    collection_name=name,
                    count_filter=Filter(
                        must=[
                            FieldCondition(
                                key="doc_id",
                                match=MatchValue(value=doc_id),
                            )
                        ]
                    ),
                    exact=True,
                )
                return int(getattr(res, "count", 0))

            return (await self._call_blocking(_count)) > 0

        except UnexpectedResponse as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 404 or (status_code is None and "not found" in str(exc).lower()):
                log.debug(
                    "Qdrant collection missing during exists check, treating as not found, name=%s",
                    name,
                )
                return False

            raise SystemError(
                code="qdrant_exists_failed",
                message="Failed to check document existence in qdrant",
                details={"collection": name, "doc_id": doc_id, "status_code": status_code},
                retryable=True,
            ) from exc

        except Exception as exc:
            raise SystemError(
                code="qdrant_exists_failed",
                message="Failed to check document existence in qdrant",
                details={"collection": name, "doc_id": doc_id},
                retryable=True,
            ) from exc

    def _delete_any(self, *, name: str, doc_id: str) -> Any:
        selector = Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        )

        if hasattr(self.client, "delete"):
            return self.client.delete(
                collection_name=name,
                points_selector=selector,
                wait=True,
            )

        if hasattr(self.client, "delete_points"):
            return self.client.delete_points(
                collection_name=name,
                points_selector=selector,
                wait=True,
            )

        raise AttributeError("No compatible delete method found on QdrantClient")

    async def delete_by_doc_id(self, *, embedding_model_id: str, doc_id: str) -> None:
        if not doc_id:
            return

        name = self._collection_name(embedding_model_id=embedding_model_id)

        try:
            await self._call_blocking(lambda: self._delete_any(name=name, doc_id=doc_id))

        except AttributeError as exc:
            raise SystemError(
                code="qdrant_client_incompatible",
                message="Installed qdrant client is incompatible with this server code",
                details={"missing": "delete or delete_points"},
                retryable=False,
            ) from exc

        except UnexpectedResponse as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 404:
                log.debug(
                    "Qdrant collection missing during delete_by_doc_id, name=%s",
                    name,
                )
                return

            raise SystemError(
                code="qdrant_delete_failed",
                message="Failed to delete vectors from qdrant",
                details={"collection": name, "status_code": status_code},
                retryable=True,
            ) from exc

        except Exception as exc:
            raise SystemError(
                code="qdrant_delete_failed",
                message="Failed to delete vectors from qdrant",
                details={"collection": name},
                retryable=True,
            ) from exc

        log.info("Qdrant delete completed, name=%s, doc_id=%s", name, doc_id)

    def _resolve_filter(self, raw: object | None) -> Filter | None:
        if isinstance(raw, Filter):
            return raw
        if isinstance(raw, dict):
            return self._to_filter(raw)
        return None

    def _search_any(
        self,
        *,
        name: str,
        query_vector: Sequence[float],
        filt: Filter | None,
        top_k: int,
    ) -> Any:
        if hasattr(self.client, "query_points"):
            return self.client.query_points(
                collection_name=name,
                query=list(query_vector),
                query_filter=filt,
                limit=int(top_k),
                with_payload=True,
                with_vectors=False,
            )

        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=name,
                query_vector=list(query_vector),
                query_filter=filt,
                limit=int(top_k),
                with_payload=True,
                with_vectors=False,
            )

        raise AttributeError("No compatible search method found on QdrantClient")

    def _map_hits_with_score(self, points: Sequence[object]) -> list[VectorSearchHit]:
        hits: list[VectorSearchHit] = []
        for p in points:
            hits.append(
                VectorSearchHit(
                    point_id=str(getattr(p, "id", None)),
                    score=float(getattr(p, "score", None) or 0.0),
                    payload=dict(getattr(p, "payload", None) or {}),
                )
            )
        return hits

    async def search(
        self,
        *,
        embedding_model_id: str,
        query_vector: Sequence[float],
        top_k: int,
        qfilter: object | None,
    ) -> list[VectorSearchHit]:
        if top_k <= 0:
            return []

        name = self._collection_name(embedding_model_id=embedding_model_id)
        filt = self._resolve_filter(qfilter)

        try:
            res = await self._call_blocking(
                lambda: self._search_any(
                    name=name,
                    query_vector=query_vector,
                    filt=filt,
                    top_k=top_k,
                )
            )

        except AttributeError as exc:
            raise SystemError(
                code="qdrant_client_incompatible",
                message="Installed qdrant client is incompatible with this server code",
                details={"missing": "query_points or search"},
                retryable=False,
            ) from exc

        except UnexpectedResponse as exc:

            status_code = getattr(exc, "status_code", None)

            if status_code is None:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)

            if status_code == 404:
                log.debug(
                    "Qdrant collection missing during search, returning empty, name=%s",
                    name,
                )

                return []

            raise SystemError(
                code="qdrant_search_failed",
                message="Failed to search qdrant",
                details={"collection": name, "status_code": status_code},
                retryable=True,
            ) from exc

        except Exception as exc:
            raise SystemError(
                code="qdrant_search_failed",
                message="Failed to search qdrant",
                details={"collection": name},
                retryable=True,
            ) from exc

        points = getattr(res, "points", None) or res or []
        return self._map_hits_with_score(points)

    def _scroll_all(
        self,
        *,
        name: str,
        scroll_filter: Filter,
        limit: int,
    ) -> list[object]:
        out: list[object] = []
        offset = None
        remaining = int(limit) if limit and limit > 0 else 20000

        while remaining > 0:
            page_limit = min(256, remaining)

            page, next_offset = self.client.scroll(
                collection_name=name,
                scroll_filter=scroll_filter,
                limit=page_limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if page:
                out.extend(page)
                remaining -= len(page)

            if next_offset is None or not page:
                break

            offset = next_offset

        return out

    def _map_hits_without_score(self, points: Sequence[object]) -> list[VectorSearchHit]:
        hits: list[VectorSearchHit] = []
        for p in points:
            hits.append(
                VectorSearchHit(
                    point_id=str(getattr(p, "id", None)),
                    score=0.0,
                    payload=dict(getattr(p, "payload", None) or {}),
                )
            )
        return hits

    async def fetch_by_doc_ids(
        self,
        *,
        embedding_model_id: str,
        doc_ids: Sequence[str],
        base_filter: object | None,
        limit: int = 20000,
    ) -> list[VectorSearchHit]:
        doc_ids_clean = [d for d in doc_ids if d]
        if not doc_ids_clean:
            return []

        name = self._collection_name(embedding_model_id=embedding_model_id)
        base = self._resolve_filter(base_filter)

        should = [FieldCondition(key="doc_id", match=MatchValue(value=d)) for d in doc_ids_clean]

        scroll_filter = (
            Filter(should=should)
            if base is None
            else Filter(
                must=list(base.must or []),
                should=should,
                must_not=list(base.must_not or []),
            )
        )

        try:
            points = await self._call_blocking(
                lambda: self._scroll_all(
                    name=name,
                    scroll_filter=scroll_filter,
                    limit=limit,
                )
            )

        except UnexpectedResponse as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 404:
                log.debug(
                    "Qdrant collection missing during fetch_by_doc_ids, returning empty, name=%s",
                    name,
                )
                return []

            raise SystemError(
                code="qdrant_fetch_failed",
                message="Failed to fetch document chunks from qdrant",
                details={"collection": name, "status_code": status_code},
                retryable=True,
            ) from exc

        except Exception as exc:
            raise SystemError(
                code="qdrant_fetch_failed",
                message="Failed to fetch document chunks from qdrant",
                details={"collection": name},
                retryable=True,
            ) from exc

        return self._map_hits_without_score(points)

    async def fetch_all(
        self,
        *,
        embedding_model_id: str,
        limit: int = 20000,
    ) -> list[VectorSearchHit]:
        name = self._collection_name(embedding_model_id=embedding_model_id)

        try:
            points = await self._call_blocking(
                lambda: self._scroll_all(
                    name=name,
                    scroll_filter=Filter(must=[]),
                    limit=limit,
                )
            )

        except UnexpectedResponse as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 404:
                log.debug(
                    "Qdrant collection missing during fetch_all, returning empty, name=%s",
                    name,
                )
                return []

            raise SystemError(
                code="qdrant_fetch_failed",
                message="Failed to fetch document chunks from qdrant",
                details={"collection": name, "status_code": status_code},
                retryable=True,
            ) from exc

        except Exception as exc:
            raise SystemError(
                code="qdrant_fetch_failed",
                message="Failed to fetch document chunks from qdrant",
                details={"collection": name},
                retryable=True,
            ) from exc

        return self._map_hits_without_score(points)

    def _to_filter(self, raw: dict[str, object]) -> Filter | None:
        must: list[Any] = []

        doi = raw.get("doi")
        if isinstance(doi, str) and doi.strip():
            must.append(FieldCondition(key="doi_original", match=MatchValue(value=doi.strip())))

        doi_normalized = raw.get("doi_normalized")
        if isinstance(doi_normalized, str) and doi_normalized.strip():
            must.append(
                FieldCondition(key="doi_normalized", match=MatchValue(value=doi_normalized.strip()))
            )

        disease = raw.get("disease")
        if isinstance(disease, str) and disease:
            must.append(FieldCondition(key="disease", match=MatchValue(value=disease)))

        source_type = raw.get("source_type")
        if isinstance(source_type, str) and source_type:
            must.append(FieldCondition(key="source_type", match=MatchValue(value=source_type)))

        year = raw.get("year")
        if isinstance(year, int):
            must.append(FieldCondition(key="year", match=MatchValue(value=int(year))))

        year_min = raw.get("year_min")
        year_max = raw.get("year_max")
        if year_min is not None or year_max is not None:
            must.append(
                FieldCondition(
                    key="year",
                    range=Range(
                        gte=int(year_min) if isinstance(year_min, int) else None,
                        lte=int(year_max) if isinstance(year_max, int) else None,
                    ),
                )
            )

        return Filter(must=must) if must else None

    def _to_chunk_candidate(self, hit: VectorSearchHit) -> ChunkCandidate:
        payload = hit.payload or {}

        doc_id = str(payload.get("doc_id") or "")
        doi = str(payload.get("doi_original") or payload.get("doi") or "")

        title = payload.get("title")
        year = payload.get("year")
        section = payload.get("section")
        source_type = payload.get("source_type")
        text = payload.get("text")

        parsed_title = str(title) if isinstance(title, str) else None
        parsed_section = str(section) if isinstance(section, str) else None
        parsed_year = int(year) if isinstance(year, int) else None
        parsed_text = str(text) if isinstance(text, str) else None

        parsed_source_type = None
        if isinstance(source_type, str):
            try:
                parsed_source_type = SourceType(source_type)
            except Exception:
                parsed_source_type = None

        return ChunkCandidate(
            chunk_id=str(hit.point_id),
            doc_id=doc_id,
            doi=doi,
            title=parsed_title,
            year=parsed_year,
            section=parsed_section,
            source_type=parsed_source_type,
            score=float(hit.score),
            chunk_text=parsed_text,
        )

    async def search_chunks(
        self,
        *,
        embedding_model_id: str,
        query_vector: Sequence[float],
        top_k: int,
        qfilter: object | None,
    ) -> list[ChunkCandidate]:
        raw_hits = await self.search(
            embedding_model_id=embedding_model_id,
            query_vector=query_vector,
            top_k=top_k,
            qfilter=qfilter,
        )

        out: list[ChunkCandidate] = []
        for h in raw_hits:
            out.append(self._to_chunk_candidate(h))
        return out

    def _retrieve_any(self, *, name: str, chunk_ids: Sequence[str]) -> Any:
        ids = [c for c in chunk_ids if c]

        if hasattr(self.client, "retrieve"):
            return self.client.retrieve(
                collection_name=name,
                ids=ids,
                with_payload=True,
                with_vectors=False,
            )

        if hasattr(self.client, "get_points"):
            return self.client.get_points(
                collection_name=name,
                ids=ids,
                with_payload=True,
                with_vectors=False,
            )

        raise AttributeError("No compatible retrieve method found on QdrantClient")

    async def fetch_chunks_by_ids(
        self,
        *,
        embedding_model_id: str,
        chunk_ids: Sequence[str],
    ) -> list[ChunkCandidate]:
        ids = [c for c in chunk_ids if c]
        if not ids:
            return []

        name = self._collection_name(embedding_model_id=embedding_model_id)

        try:
            res = await self._call_blocking(lambda: self._retrieve_any(name=name, chunk_ids=ids))

        except AttributeError as exc:
            raise SystemError(
                code="qdrant_client_incompatible",
                message="Installed qdrant client is incompatible with this server code",
                details={"missing": "retrieve or get_points"},
                retryable=False,
            ) from exc

        except UnexpectedResponse as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 404:
                log.debug(
                    "Qdrant collection missing during fetch_chunks_by_ids, returning empty,"
                    " name=%s",
                    name,
                )
                return []

            raise SystemError(
                code="qdrant_fetch_failed",
                message="Failed to fetch chunks from qdrant",
                details={"collection": name, "status_code": status_code},
                retryable=True,
            ) from exc

        except Exception as exc:
            raise SystemError(
                code="qdrant_fetch_failed",
                message="Failed to fetch chunks from qdrant",
                details={"collection": name},
                retryable=True,
            ) from exc

        points = getattr(res, "points", None) or res or []

        out: list[ChunkCandidate] = []
        for p in points:
            hit = VectorSearchHit(
                point_id=str(getattr(p, "id", None)),
                score=float(getattr(p, "score", None) or 0.0),
                payload=dict(getattr(p, "payload", None) or {}),
            )
            out.append(self._to_chunk_candidate(hit))

        return out
