from __future__ import annotations

import asyncio

import pytest

from biomed_platform.core.services.ingestion import InMemoryDocumentRegistry

pytestmark = pytest.mark.asyncio


class TestInMemoryDocumentRegistry:
    async def test_reserve_then_commit_moves_doc_id_to_committed(self) -> None:
        reg = InMemoryDocumentRegistry()

        await reg.reserve(embedding_model_id="m1", doc_id="d1")

        space = reg._space("m1")
        assert "d1" in space.reserved
        assert "d1" not in space.committed

        await reg.commit(embedding_model_id="m1", doc_id="d1")

        space = reg._space("m1")
        assert "d1" not in space.reserved
        assert "d1" in space.committed

    async def test_release_removes_from_reserved_only(self) -> None:
        reg = InMemoryDocumentRegistry()

        await reg.reserve(embedding_model_id="m1", doc_id="d1")
        await reg.release(embedding_model_id="m1", doc_id="d1")

        space = reg._space("m1")
        assert "d1" not in space.reserved
        assert "d1" not in space.committed

    async def test_commit_without_reserve_is_idempotent_and_adds_to_committed(self) -> None:
        reg = InMemoryDocumentRegistry()

        await reg.commit(embedding_model_id="m1", doc_id="d1")

        space = reg._space("m1")
        assert "d1" not in space.reserved
        assert "d1" in space.committed

    async def test_release_without_reserve_is_noop(self) -> None:
        reg = InMemoryDocumentRegistry()

        await reg.release(embedding_model_id="m1", doc_id="d1")

        space = reg._space("m1")
        assert "d1" not in space.reserved
        assert "d1" not in space.committed

    async def test_reserve_raises_when_already_reserved(self) -> None:
        reg = InMemoryDocumentRegistry()

        await reg.reserve(embedding_model_id="m1", doc_id="d1")

        with pytest.raises(KeyError) as exc:
            await reg.reserve(embedding_model_id="m1", doc_id="d1")

        assert exc.value.args == ("d1",)

    async def test_reserve_raises_when_already_committed(self) -> None:
        reg = InMemoryDocumentRegistry()

        await reg.reserve(embedding_model_id="m1", doc_id="d1")
        await reg.commit(embedding_model_id="m1", doc_id="d1")

        with pytest.raises(KeyError) as exc:
            await reg.reserve(embedding_model_id="m1", doc_id="d1")

        assert exc.value.args == ("d1",)

    async def test_model_spaces_are_isolated(self) -> None:
        reg = InMemoryDocumentRegistry()

        await reg.reserve(embedding_model_id="m1", doc_id="d1")
        await reg.reserve(embedding_model_id="m2", doc_id="d1")

        space1 = reg._space("m1")
        space2 = reg._space("m2")

        assert "d1" in space1.reserved
        assert "d1" in space2.reserved
        assert space1 is not space2

    async def test_concurrent_reserve_same_doc_id_only_one_succeeds(self) -> None:
        reg = InMemoryDocumentRegistry()

        async def attempt() -> bool:
            try:
                await reg.reserve(embedding_model_id="m1", doc_id="d1")
                return True
            except KeyError:
                return False

        results = await asyncio.gather(*[attempt() for _ in range(50)])

        assert sum(results) == 1

        space = reg._space("m1")
        assert "d1" in space.reserved
        assert "d1" not in space.committed

    async def test_concurrent_commit_and_release_do_not_error(self) -> None:
        reg = InMemoryDocumentRegistry()

        await reg.reserve(embedding_model_id="m1", doc_id="d1")

        async def do_commit() -> None:
            await reg.commit(embedding_model_id="m1", doc_id="d1")

        async def do_release() -> None:
            await reg.release(embedding_model_id="m1", doc_id="d1")

        await asyncio.gather(*[do_commit() for _ in range(25)], *[do_release() for _ in range(25)])

        space = reg._space("m1")
        assert "d1" in space.committed
        assert "d1" not in space.reserved
