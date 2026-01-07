from __future__ import annotations

from biomed_platform.common.middleware.trace import request_id_ctx


class TestRequestIdContextVar:
    def test_default_value_is_none(self) -> None:
        # Given / When / Then
        assert request_id_ctx.get() is None

    def test_set_and_get_request_id(self) -> None:
        # Given
        token = request_id_ctx.set("rid-123")

        try:
            # When / Then
            assert request_id_ctx.get() == "rid-123"
        finally:
            request_id_ctx.reset(token)

        assert request_id_ctx.get() is None

    def test_reset_restores_previous_value(self) -> None:
        # Given
        token1 = request_id_ctx.set("outer")

        try:
            token2 = request_id_ctx.set("inner")

            try:
                # When / Then
                assert request_id_ctx.get() == "inner"
            finally:
                request_id_ctx.reset(token2)

            assert request_id_ctx.get() == "outer"
        finally:
            request_id_ctx.reset(token1)

        assert request_id_ctx.get() is None

    def test_nested_contexts_do_not_leak(self) -> None:
        # Given
        token_outer = request_id_ctx.set("outer")

        try:
            assert request_id_ctx.get() == "outer"

            token_inner = request_id_ctx.set("inner")
            try:
                # When / Then
                assert request_id_ctx.get() == "inner"
            finally:
                request_id_ctx.reset(token_inner)

            assert request_id_ctx.get() == "outer"
        finally:
            request_id_ctx.reset(token_outer)

        assert request_id_ctx.get() is None

    def test_reset_without_prior_set_raises(self) -> None:
        # Given
        token = request_id_ctx.set("rid")

        # When
        request_id_ctx.reset(token)

        # Then
        assert request_id_ctx.get() is None
