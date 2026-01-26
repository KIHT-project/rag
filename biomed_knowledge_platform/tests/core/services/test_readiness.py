from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import httpx
import pytest

from biomed_platform.core.domains.readiness import (
    CheckStatus,
    ReadinessChecks,
    ReadinessResult,
    ReadinessStatus,
)
from biomed_platform.core.services import readiness as readiness_mod


@dataclass(frozen=True)
class _FakeResponse:
    status_code: int


class _FakeAsyncClient:
    def __init__(self, get_impl: Callable[[str], Any]) -> None:
        self._get_impl = get_impl
        self.closed = False
        self.requested_urls: list[str] = []

    async def get(self, url: str) -> _FakeResponse:
        self.requested_urls.append(url)
        result = self._get_impl(url)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, _FakeResponse):
            return result
        raise TypeError(
            f"Fake client get_impl must return _FakeResponse or Exception, got {type(result)}"
        )

    async def aclose(self) -> None:
        self.closed = True


class TestReadinessService:
    def test_normalize_ollama_base_url_strips_version_suffix(self) -> None:
        # Given
        url = "http://ollama:11434/version"

        # When
        normalized = readiness_mod.normalize_ollama_base_url(url)

        # Then
        assert normalized == "http://ollama:11434"

    def test_normalize_ollama_base_url_strips_version_suffix_with_trailing_slash(self) -> None:
        # Given
        url = "http://ollama:11434/version/"

        # When
        normalized = readiness_mod.normalize_ollama_base_url(url)

        # Then
        assert normalized == "http://ollama:11434"

    def test_normalize_ollama_base_url_returns_empty_for_empty_input(self) -> None:
        # Given
        url = ""

        # When
        normalized = readiness_mod.normalize_ollama_base_url(url)

        # Then
        assert normalized == ""

    def test_evaluate_readiness_status_returns_ready_when_all_ok(self) -> None:
        # Given
        checks = ReadinessChecks(qdrant=CheckStatus.ok, llm=CheckStatus.ok, rdbms=CheckStatus.ok)

        # When
        status = readiness_mod.evaluate_readiness_status(checks)

        # Then
        assert status == ReadinessStatus.ready

    @pytest.mark.parametrize(
        ("qdrant", "llm"),
        [
            (CheckStatus.error, CheckStatus.ok),
            (CheckStatus.ok, CheckStatus.error),
            (CheckStatus.unhealthy, CheckStatus.ok),
            (CheckStatus.ok, CheckStatus.unhealthy),
            (CheckStatus.degraded, CheckStatus.ok),
            (CheckStatus.ok, CheckStatus.degraded),
            (CheckStatus.missing_config, CheckStatus.ok),
            (CheckStatus.ok, CheckStatus.missing_config),
        ],
    )
    def test_evaluate_readiness_status_returns_not_ready_when_any_not_ok(
        self, qdrant: CheckStatus, llm: CheckStatus
    ) -> None:
        # Given
        checks = ReadinessChecks(qdrant=qdrant, llm=llm, rdbms=CheckStatus.ok)

        # When
        status = readiness_mod.evaluate_readiness_status(checks)

        # Then
        assert status == ReadinessStatus.not_ready

    @pytest.mark.asyncio
    async def test_check_qdrant_returns_missing_config_when_base_url_empty(self) -> None:
        # Given
        client = _FakeAsyncClient(lambda _: _FakeResponse(status_code=200))

        # When
        status, err = await readiness_mod.check_qdrant(client, "")

        # Then
        assert status == CheckStatus.missing_config
        assert err == {"reason": "Missing configuration"}
        assert client.requested_urls == []

    @pytest.mark.asyncio
    async def test_check_ollama_returns_missing_config_when_base_url_empty(self) -> None:
        # Given
        client = _FakeAsyncClient(lambda _: _FakeResponse(status_code=200))

        # When
        status, err = await readiness_mod.check_ollama(client, "")

        # Then
        assert status == CheckStatus.missing_config
        assert err == {"reason": "Missing configuration"}
        assert client.requested_urls == []

    @pytest.mark.asyncio
    async def test_check_qdrant_returns_ok_on_2xx(self) -> None:
        # Given
        def fake_get(url: str) -> _FakeResponse:
            assert url == "http://qdrant:6333/collections"
            return _FakeResponse(status_code=200)

        client = _FakeAsyncClient(fake_get)

        # When
        status, err = await readiness_mod.check_qdrant(client, "http://qdrant:6333")

        # Then
        assert status == CheckStatus.ok
        assert err is None

    @pytest.mark.asyncio
    async def test_check_ollama_returns_ok_on_2xx(self) -> None:
        # Given
        def fake_get(url: str) -> _FakeResponse:
            assert url == "http://ollama:11434/api/version"
            return _FakeResponse(status_code=204)

        client = _FakeAsyncClient(fake_get)

        # When
        status, err = await readiness_mod.check_ollama(client, "http://ollama:11434")

        # Then
        assert status == CheckStatus.ok
        assert err is None

    @pytest.mark.asyncio
    async def test_check_qdrant_returns_degraded_on_4xx(self) -> None:
        # Given
        client = _FakeAsyncClient(lambda _: _FakeResponse(status_code=404))

        # When
        status, err = await readiness_mod.check_qdrant(client, "http://qdrant:6333")

        # Then
        assert status == CheckStatus.degraded
        assert err == {"reason": "http_4xx", "status_code": 404}

    @pytest.mark.asyncio
    async def test_check_ollama_returns_degraded_on_4xx(self) -> None:
        # Given
        client = _FakeAsyncClient(lambda _: _FakeResponse(status_code=429))

        # When
        status, err = await readiness_mod.check_ollama(client, "http://ollama:11434")

        # Then
        assert status == CheckStatus.degraded
        assert err == {"reason": "http_4xx", "status_code": 429}

    @pytest.mark.asyncio
    async def test_check_qdrant_returns_unhealthy_on_5xx(self) -> None:
        # Given
        client = _FakeAsyncClient(lambda _: _FakeResponse(status_code=500))

        # When
        status, err = await readiness_mod.check_qdrant(client, "http://qdrant:6333")

        # Then
        assert status == CheckStatus.unhealthy
        assert err == {"reason": "http_5xx", "status_code": 500}

    @pytest.mark.asyncio
    async def test_check_ollama_returns_unhealthy_on_5xx(self) -> None:
        # Given
        client = _FakeAsyncClient(lambda _: _FakeResponse(status_code=503))

        # When
        status, err = await readiness_mod.check_ollama(client, "http://ollama:11434")

        # Then
        assert status == CheckStatus.unhealthy
        assert err == {"reason": "http_5xx", "status_code": 503}

    @pytest.mark.asyncio
    async def test_check_qdrant_returns_error_on_timeout(self) -> None:
        # Given
        client = _FakeAsyncClient(lambda _: httpx.TimeoutException("timeout"))

        # When
        status, err = await readiness_mod.check_qdrant(client, "http://qdrant:6333")

        # Then
        assert status == CheckStatus.error
        assert err == {"reason": "timeout", "base_url": "http://qdrant:6333"}

    @pytest.mark.asyncio
    async def test_check_ollama_returns_error_on_timeout(self) -> None:
        # Given
        client = _FakeAsyncClient(lambda _: httpx.TimeoutException("timeout"))

        # When
        status, err = await readiness_mod.check_ollama(client, "http://ollama:11434")

        # Then
        assert status == CheckStatus.error
        assert err == {"reason": "timeout", "base_url": "http://ollama:11434"}

    @pytest.mark.asyncio
    async def test_check_qdrant_returns_error_on_request_error(self) -> None:
        # Given
        client = _FakeAsyncClient(lambda _: httpx.RequestError("boom"))

        # When
        status, err = await readiness_mod.check_qdrant(client, "http://qdrant:6333")

        # Then
        assert status == CheckStatus.error
        assert err == {"reason": "RequestError", "base_url": "http://qdrant:6333"}

    @pytest.mark.asyncio
    async def test_check_ollama_returns_error_on_request_error(self) -> None:
        # Given
        client = _FakeAsyncClient(lambda _: httpx.RequestError("boom"))

        # When
        status, err = await readiness_mod.check_ollama(client, "http://ollama:11434")

        # Then
        assert status == CheckStatus.error
        assert err == {"reason": "RequestError", "base_url": "http://ollama:11434"}

    @pytest.mark.asyncio
    async def test_compute_readiness_returns_ready_when_both_dependencies_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given
        async def _fake_check_postgres(postgres_url: str):
            return CheckStatus.ok, None

        monkeypatch.setattr(readiness_mod, "check_postgres", _fake_check_postgres)
        def fake_get(url: str) -> Any:
            if url == "http://qdrant:6333/collections":
                return _FakeResponse(status_code=200)
            if url == "http://ollama:11434/api/version":
                return _FakeResponse(status_code=200)
            return _FakeResponse(status_code=500)

        client = _FakeAsyncClient(fake_get)
        timeout = httpx.Timeout(connect=0.1, read=0.1, write=0.1, pool=0.1)

        # When
        result = await readiness_mod.compute_readiness(
            qdrant_url="http://qdrant:6333",
            ollama_url="http://ollama:11434",
            postgres_url="postgresql://user:pass@localhost:5432/db",
            timeout=timeout,
            client=client,
        )

        # Then
        assert isinstance(result, ReadinessResult)
        assert result.status == ReadinessStatus.ready
        assert result.checks == ReadinessChecks(qdrant=CheckStatus.ok, llm=CheckStatus.ok, rdbms=CheckStatus.ok)
        assert result.errors is None
        assert client.closed is False

    @pytest.mark.asyncio
    async def test_compute_readiness_returns_not_ready_when_qdrant_not_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given
        async def _fake_check_postgres(postgres_url: str):
            return CheckStatus.ok, None

        monkeypatch.setattr(readiness_mod, "check_postgres", _fake_check_postgres)
        def fake_get(url: str) -> Any:
            if url == "http://qdrant:6333/collections":
                return _FakeResponse(status_code=500)
            if url == "http://ollama:11434/api/version":
                return _FakeResponse(status_code=200)
            return _FakeResponse(status_code=500)

        client = _FakeAsyncClient(fake_get)
        timeout = httpx.Timeout(connect=0.1, read=0.1, write=0.1, pool=0.1)

        # When
        result = await readiness_mod.compute_readiness(
            qdrant_url="http://qdrant:6333",
            ollama_url="http://ollama:11434",
            postgres_url="postgresql://user:pass@localhost:5432/db",
            timeout=timeout,
            client=client,
        )

        # Then
        assert result.status == ReadinessStatus.not_ready
        assert result.checks.qdrant == CheckStatus.unhealthy
        assert result.checks.llm == CheckStatus.ok
        assert result.errors == {"qdrant": {"reason": "http_5xx", "status_code": 500}}
        assert client.closed is False

    @pytest.mark.asyncio
    async def test_compute_readiness_returns_not_ready_when_ollama_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given
        async def _fake_check_postgres(postgres_url: str):
            return CheckStatus.ok, None

        monkeypatch.setattr(readiness_mod, "check_postgres", _fake_check_postgres)
        def fake_get(url: str) -> Any:
            if url == "http://qdrant:6333/collections":
                return _FakeResponse(status_code=200)
            if url == "http://ollama:11434/api/version":
                return httpx.TimeoutException("timeout")
            return _FakeResponse(status_code=500)

        client = _FakeAsyncClient(fake_get)
        timeout = httpx.Timeout(connect=0.1, read=0.1, write=0.1, pool=0.1)

        # When
        result = await readiness_mod.compute_readiness(
            qdrant_url="http://qdrant:6333",
            ollama_url="http://ollama:11434",
            postgres_url="postgresql://user:pass@localhost:5432/db",
            timeout=timeout,
            client=client,
        )

        # Then
        assert result.status == ReadinessStatus.not_ready
        assert result.checks.qdrant == CheckStatus.ok
        assert result.checks.llm == CheckStatus.error
        assert result.errors == {
            "ollama": {"reason": "timeout", "base_url": "http://ollama:11434"}
        }
        assert client.closed is False

    @pytest.mark.asyncio
    async def test_compute_readiness_closes_owned_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given
        async def _fake_check_postgres(postgres_url: str):
            return CheckStatus.ok, None

        monkeypatch.setattr(readiness_mod, "check_postgres", _fake_check_postgres)
        created: dict[str, Any] = {}

        def fake_async_client(*, timeout: httpx.Timeout) -> _FakeAsyncClient:
            def fake_get(url: str) -> Any:
                if url.endswith("/collections"):
                    return _FakeResponse(status_code=200)
                return _FakeResponse(status_code=200)

            client = _FakeAsyncClient(fake_get)
            created["client"] = client
            created["timeout"] = timeout
            return client

        monkeypatch.setattr(readiness_mod.httpx, "AsyncClient", fake_async_client)

        timeout = httpx.Timeout(connect=0.1, read=0.1, write=0.1, pool=0.1)

        # When
        result = await readiness_mod.compute_readiness(
            qdrant_url="http://qdrant:6333",
            ollama_url="http://ollama:11434",
            postgres_url="postgresql://user:pass@localhost:5432/db",
            timeout=timeout,
            client=None,
        )

        # Then
        assert result.status == ReadinessStatus.ready
        assert result.errors is None
        assert created["client"].closed is True
        assert created["timeout"] == timeout
