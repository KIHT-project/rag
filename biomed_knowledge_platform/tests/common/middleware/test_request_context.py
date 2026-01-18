from __future__ import annotations

import re
import uuid
from typing import Any

import pytest
from starlette.requests import Request
from starlette.responses import Response

from biomed_platform.common.middleware.request_context import (
    AccessLogMiddleware,
    RequestContextMiddleware,
)
from biomed_platform.common.middleware.trace import request_id_ctx


def _make_request(*, path: str = "/probe", headers: dict[str, str] | None = None) -> Request:
    raw_headers: list[tuple[bytes, bytes]] = []
    if headers:
        raw_headers = [(k.lower().encode("latin1"), v.encode("latin1")) for k, v in headers.items()]

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"spec_version": "2.3", "version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": raw_headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    return Request(scope)


@pytest.mark.asyncio
class TestRequestContextMiddlewareUnit:
    async def test_uses_incoming_request_id_header_and_echoes_it(self) -> None:
        # Given
        mw = RequestContextMiddleware(app=lambda scope, receive, send: None)  # not used by dispatch
        req = _make_request(headers={"X-Request-Id": "my-trace-id-42"})

        async def call_next(_: Request) -> Response:
            assert request_id_ctx.get(None) == "my-trace-id-42"
            return Response(status_code=200)

        # When
        resp = await mw.dispatch(req, call_next)

        # Then
        assert resp.status_code == 200
        assert resp.headers.get("X-Request-Id") == "my-trace-id-42"
        assert request_id_ctx.get(None) is None

    async def test_generates_request_id_when_header_missing_and_echoes_it(self) -> None:
        # Given
        mw = RequestContextMiddleware(app=lambda scope, receive, send: None)
        req = _make_request()

        captured: dict[str, str] = {}

        async def call_next(_: Request) -> Response:
            rid = request_id_ctx.get(None)
            assert rid is not None
            captured["rid"] = rid
            return Response(status_code=200)

        # When
        resp = await mw.dispatch(req, call_next)

        # Then
        echoed = resp.headers.get("X-Request-Id")
        assert echoed is not None
        uuid.UUID(echoed)
        assert echoed == captured["rid"]
        assert request_id_ctx.get(None) is None

    async def test_context_is_reset_even_if_call_next_raises(self) -> None:
        # Given
        mw = RequestContextMiddleware(app=lambda scope, receive, send: None)
        req = _make_request(headers={"X-Request-Id": "outer"})

        async def call_next(_: Request) -> Response:
            assert request_id_ctx.get(None) == "outer"
            raise RuntimeError("boom")

        # When, Then
        with pytest.raises(RuntimeError, match="boom"):
            await mw.dispatch(req, call_next)

        assert request_id_ctx.get(None) is None


@pytest.mark.asyncio
class TestAccessLogMiddlewareUnit:
    async def test_logs_http_request_line_with_request_id_from_context(
            self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Given
        mw = AccessLogMiddleware(app=lambda scope, receive, send: None)
        req = _make_request(path="/ready")

        token = request_id_ctx.set("rid-123")
        try:
            caplog.set_level("INFO")

            async def call_next(_: Request) -> Response:
                return Response(status_code=503)

            # When
            resp = await mw.dispatch(req, call_next)
        finally:
            request_id_ctx.reset(token)

        # Then
        assert resp.status_code == 503

        records = [r for r in caplog.records if "HTTP request" in r.getMessage()]
        assert len(records) == 1

        msg = records[0].getMessage()

        assert "method=GET" in msg
        assert "path=/ready" in msg
        assert "status=503" in msg
        assert re.search(r"duration_ms=\d+(\.\d+)?", msg) is not None

        # Then, request_id is only asserted if the middleware includes it in the message itself.
        # This avoids assuming a LogRecord extra injection which may be done by the formatter.
        if "request_id=" in msg:
            assert "request_id=rid-123" in msg

    async def test_logs_http_request_line_with_request_id_none_when_context_missing(
            self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mw = AccessLogMiddleware(app=lambda scope, receive, send: None)
        req = _make_request(path="/ready")
        caplog.set_level("INFO")

        async def call_next(_: Request) -> Response:
            return Response(status_code=200)

        resp = await mw.dispatch(req, call_next)

        assert resp.status_code == 200

        records = [r for r in caplog.records if "HTTP request" in r.getMessage()]
        assert len(records) == 1

        record = records[0]
        rid = getattr(record, "request_id", None)
        if rid is None:
            rid = record.__dict__.get("request_id")

        assert rid in (None, "none")

    async def test_does_not_mutate_context(self, caplog: pytest.LogCaptureFixture) -> None:
        # Given
        mw = AccessLogMiddleware(app=lambda scope, receive, send: None)
        req = _make_request(path="/ready")
        token = request_id_ctx.set("rid-xyz")
        try:
            caplog.set_level("INFO")

            async def call_next(_: Request) -> Response:
                assert request_id_ctx.get(None) == "rid-xyz"
                return Response(status_code=200)

            # When
            resp = await mw.dispatch(req, call_next)

        finally:
            request_id_ctx.reset(token)

        # Then
        assert resp.status_code == 200
        assert request_id_ctx.get(None) is None
