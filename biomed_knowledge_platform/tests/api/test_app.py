from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum

import pytest


def _install_qdrant_client_stub() -> None:
    if "qdrant_client" in sys.modules:
        return

    class Distance(Enum):
        COSINE = "COSINE"
        DOT = "DOT"
        EUCLID = "EUCLID"

    @dataclass(frozen=True)
    class _Dummy:
        value: object | None = None

    class QdrantClient:  # minimal stub
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class UnexpectedResponse(Exception):
        pass

    http_models = type(sys)("qdrant_client.http.models")
    http_models.Distance = Distance
    http_models.Filter = _Dummy
    http_models.FieldCondition = _Dummy
    http_models.MatchValue = _Dummy
    http_models.PointStruct = _Dummy
    http_models.VectorParams = _Dummy
    http_models.Range = _Dummy

    http_exceptions = type(sys)("qdrant_client.http.exceptions")
    http_exceptions.UnexpectedResponse = UnexpectedResponse

    http_pkg = type(sys)("qdrant_client.http")

    qc = type(sys)("qdrant_client")
    qc.QdrantClient = QdrantClient

    sys.modules["qdrant_client"] = qc
    sys.modules["qdrant_client.http"] = http_pkg
    sys.modules["qdrant_client.http.models"] = http_models
    sys.modules["qdrant_client.http.exceptions"] = http_exceptions


_install_qdrant_client_stub()


def _install_sqlalchemy_stub() -> None:
    if "sqlalchemy" in sys.modules:
        return

    # Minimal stubs to allow importing the FastAPI app in environments
    # where SQLAlchemy is not installed. Tests in this suite do not
    # exercise database integration.
    sqlalchemy = type(sys)("sqlalchemy")

    orm = type(sys)("sqlalchemy.orm")

    class DeclarativeBase:  # pragma: no cover
        pass

    orm.DeclarativeBase = DeclarativeBase

    ext = type(sys)("sqlalchemy.ext")
    ext_asyncio = type(sys)("sqlalchemy.ext.asyncio")

    class AsyncEngine:  # pragma: no cover
        async def dispose(self) -> None:
            return

    class AsyncSession:  # pragma: no cover
        pass

    def create_async_engine(*args, **kwargs):  # pragma: no cover
        return AsyncEngine()

    def async_sessionmaker(*args, **kwargs):  # pragma: no cover
        return object()

    ext_asyncio.AsyncEngine = AsyncEngine
    ext_asyncio.AsyncSession = AsyncSession
    ext_asyncio.create_async_engine = create_async_engine
    ext_asyncio.async_sessionmaker = async_sessionmaker

    sys.modules["sqlalchemy"] = sqlalchemy
    sys.modules["sqlalchemy.orm"] = orm
    sys.modules["sqlalchemy.ext"] = ext
    sys.modules["sqlalchemy.ext.asyncio"] = ext_asyncio


def _install_asyncpg_stub() -> None:
    if "asyncpg" in sys.modules:
        return

    asyncpg = type(sys)("asyncpg")

    async def connect(*args, **kwargs):  # pragma: no cover
        class _Conn:
            async def close(self) -> None:
                return

        return _Conn()

    asyncpg.connect = connect
    sys.modules["asyncpg"] = asyncpg


_install_sqlalchemy_stub()
_install_asyncpg_stub()

from fastapi import FastAPI

from biomed_platform.api import app as api_app
from biomed_platform.core.errors.errors import SystemError


def test_require_dict_accepts_dict() -> None:
    # Given a dict value
    val = {"k": "v"}

    # When validating it
    out = api_app._require_dict(val, code="c", message="m")

    # Then it is returned unchanged
    assert out is val


def test_require_dict_raises_system_error_for_non_dict() -> None:
    # Given a non dict value
    val = [1, 2]

    # When validating it
    # Then a SystemError is raised
    with pytest.raises(SystemError) as exc:
        api_app._require_dict(val, code="bad", message="m")

    assert exc.value.code == "bad"


def test_create_app_wires_state_and_routes() -> None:
    # Given application factory
    # When creating an app
    app = api_app.create_app()

    # Then a FastAPI instance is created
    assert isinstance(app, FastAPI)

    # Then core state objects are exposed for endpoints
    assert getattr(app.state, "settings") is not None
    assert getattr(app.state, "ingestion_service") is not None
    assert getattr(app.state, "search_use_case") is not None
    assert getattr(app.state, "embedding_provider") is not None
    assert getattr(app.state, "vector_index") is not None

    # Then system route exists and versioned routes exist
    paths = {getattr(r, "path", "") for r in app.router.routes}
    assert "/health" in paths
    assert any(p.startswith("/v1") for p in paths)
