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
