from __future__ import annotations

import asyncio

import pytest

from biomed_platform.core.services.ingestion.in_memory_queue import InMemoryIngestionQueue


pytestmark = pytest.mark.asyncio


class TestInMemoryIngestionQueue:
    def test_init_rejects_non_positive_max_size(self) -> None:
        with pytest.raises(ValueError):
            InMemoryIngestionQueue(max_size=0)

        with pytest.raises(ValueError):
            InMemoryIngestionQueue(max_size=-1)

    async def test_max_size_returns_configured_value(self) -> None:
        q = InMemoryIngestionQueue(max_size=3)
        assert q.max_size() == 3

    async def test_size_starts_at_zero(self) -> None:
        q = InMemoryIngestionQueue(max_size=3)
        assert q.size() == 0

    async def test_enqueue_increases_size(self) -> None:
        q = InMemoryIngestionQueue(max_size=3)

        await q.enqueue("j1")
        await q.enqueue("j2")

        assert q.size() == 2

    async def test_dequeue_returns_fifo_order_and_decreases_size(self) -> None:
        q = InMemoryIngestionQueue(max_size=3)

        await q.enqueue("j1")
        await q.enqueue("j2")

        got1 = await q.dequeue()
        got2 = await q.dequeue()

        assert got1 == "j1"
        assert got2 == "j2"
        assert q.size() == 0

    async def test_enqueue_raises_queue_full_when_at_capacity(self) -> None:
        q = InMemoryIngestionQueue(max_size=2)

        await q.enqueue("j1")
        await q.enqueue("j2")

        with pytest.raises(asyncio.QueueFull):
            await q.enqueue("j3")

        assert q.size() == 2

    async def test_dequeue_blocks_until_item_available(self) -> None:
        q = InMemoryIngestionQueue(max_size=2)

        async def consumer() -> str:
            return await q.dequeue()

        task = asyncio.create_task(consumer())

        await asyncio.sleep(0)  # allow task to start and block on dequeue
        assert not task.done()

        await q.enqueue("j1")

        got = await asyncio.wait_for(task, timeout=1.0)
        assert got == "j1"
        assert q.size() == 0
