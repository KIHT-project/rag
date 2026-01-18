from __future__ import annotations

import pytest

from biomed_platform.common.middleware.trace import get_request_id, request_id_ctx



class TestRequestIdContextVar:
    def test_default_value_is_none(self) -> None:
        assert request_id_ctx.get() is None

    def test_get_request_id_returns_none_string_when_ctx_is_none(self) -> None:
        assert get_request_id() == "none"

    def test_set_and_get_request_id(self) -> None:
        token = request_id_ctx.set("rid-123")
        try:
            assert request_id_ctx.get() == "rid-123"
            assert get_request_id() == "rid-123"
        finally:
            request_id_ctx.reset(token)

        assert request_id_ctx.get() is None
        assert get_request_id() == "none"

    def test_reset_restores_previous_value(self) -> None:
        token1 = request_id_ctx.set("outer")
        try:
            token2 = request_id_ctx.set("inner")
            try:
                assert request_id_ctx.get() == "inner"
                assert get_request_id() == "inner"
            finally:
                request_id_ctx.reset(token2)

            assert request_id_ctx.get() == "outer"
            assert get_request_id() == "outer"
        finally:
            request_id_ctx.reset(token1)

        assert request_id_ctx.get() is None
        assert get_request_id() == "none"

    def test_reset_with_same_token_twice_raises(self) -> None:
        token = request_id_ctx.set("rid")
        request_id_ctx.reset(token)

        with pytest.raises(RuntimeError):
            request_id_ctx.reset(token)
