from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from biomed_platform.common.logging import get_logger
from biomed_platform.core.ports.ingestion import IdempotencyStore

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _IdemRecord:
    body_hash: str
    job_id: str
    created_at: datetime


class InMemoryIdempotencyStore(IdempotencyStore):
    def __init__(self, *, ttl_seconds: int = 86400) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl = timedelta(seconds=ttl_seconds)
        self._by_key: dict[str, _IdemRecord] = {}

        log.debug(
            "Idempotency store initialized, ttl_seconds=%d",
            ttl_seconds,
        )

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _is_expired(self, rec: _IdemRecord, now: datetime) -> bool:
        created_at = rec.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return (now - created_at) > self._ttl

    def _cleanup(self, now: datetime) -> None:
        expired = [k for k, v in self._by_key.items() if self._is_expired(v, now)]
        if not expired:
            return

        for k in expired:
            self._by_key.pop(k, None)

        log.debug(
            "Idempotency cleanup removed expired records, count=%d",
            len(expired),
        )

    async def get_job_id(self, *, key: str, body_hash: str) -> str | None:
        now = self._now()
        self._cleanup(now)

        rec = self._by_key.get(key)
        if rec is None:
            log.debug(
                "Idempotency miss, key=%s",
                key,
            )
            return None

        if rec.body_hash != body_hash:
            log.warning(
                "Idempotency key reused with different body hash, key=%s",
                key,
            )
            return None

        log.debug(
            "Idempotency hit, key=%s, job_id=%s",
            key,
            rec.job_id,
        )
        return rec.job_id

    async def put(
        self,
        *,
        key: str,
        body_hash: str,
        job_id: str,
        created_at: datetime,
    ) -> None:
        now = self._now()
        self._cleanup(now)

        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        replaced = key in self._by_key
        self._by_key[key] = _IdemRecord(
            body_hash=body_hash,
            job_id=job_id,
            created_at=created_at,
        )

        log.debug(
            "Idempotency record stored, key=%s, job_id=%s, replaced_existing=%s",
            key,
            job_id,
            replaced,
        )

    async def peek_record(self, *, key: str) -> _IdemRecord | None:
        rec = self._by_key.get(key)
        log.debug(
            "Idempotency peek, key=%s, found=%s",
            key,
            rec is not None,
        )
        return rec
