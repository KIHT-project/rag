from __future__ import annotations

from fastapi import APIRouter

from biomed_platform.api.router import router as v1_router


class TestV1Router:
    def test_router_is_apirouter(self) -> None:
        # Given

        # When
        r = v1_router

        # Then
        assert isinstance(r, APIRouter)

    def test_router_prefix_is_v1(self) -> None:
        # Given

        # When
        prefix = v1_router.prefix

        # Then
        assert prefix == "/v1"
