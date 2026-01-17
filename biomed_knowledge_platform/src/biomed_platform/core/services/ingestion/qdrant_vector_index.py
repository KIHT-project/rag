# src/biomed_platform/core/services/ingestion/qdrant_vector_index.py
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
)

from biomed_platform.common.logging import get_logger
from biomed_platform.core.domains.ingestion import VectorPoint
from biomed_platform.core.domains.retrieval import VectorSearchHit
from biomed_platform.core.errors.errors import SystemError
from biomed_platform.core.services.ingestion_ports import VectorIndex

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
class QdrantVectorIndex(VectorIndex):
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
        for c in existing:
            if getattr(c, "name", None) == name:
                return True
        return False

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
                "Upsert skipped, no points provided, embedding_model_id=%s", embedding_model_id
            )
            return

        name = self._collection_name(embedding_model_id=embedding_model_id)

        log.debug("Preparing qdrant upsert, name=%s, point_count=%s", name, len(points))

        qpoints: list[PointStruct] = [
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

            cnt = await self._call_blocking(_count)
            return cnt > 0

        except UnexpectedResponse as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 404:
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
        filt: Filter | None = qfilter if isinstance(qfilter, Filter) else None

        try:

            def _search_any() -> Any:
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

            res = await self._call_blocking(_search_any)

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
                log.debug("Qdrant collection missing during search, returning empty, name=%s", name)

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

        points = getattr(res, "points", None)
        if points is None:
            points = res or []

        hits: list[VectorSearchHit] = []
        for p in points:
            pid = getattr(p, "id", None)
            score = getattr(p, "score", None)
            payload = getattr(p, "payload", None) or {}
            hits.append(
                VectorSearchHit(
                    point_id=str(pid),
                    score=float(score or 0.0),
                    payload=dict(payload),
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

        base: Filter | None = base_filter if isinstance(base_filter, Filter) else None

        should = [
            FieldCondition(key="doc_id", match=MatchValue(value=did)) for did in doc_ids_clean
        ]

        if base is None:
            scroll_filter = Filter(should=should)
        else:
            scroll_filter = Filter(
                must=list(base.must or []),
                should=should,
                must_not=list(base.must_not or []),
            )

        try:

            def _scroll_all() -> list[object]:
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

            points = await self._call_blocking(_scroll_all)

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

        hits: list[VectorSearchHit] = []
        for p in points or []:
            pid = getattr(p, "id", None)
            payload = getattr(p, "payload", None) or {}
            hits.append(
                VectorSearchHit(
                    point_id=str(pid),
                    score=0.0,
                    payload=dict(payload),
                )
            )

        return hits
