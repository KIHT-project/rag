from __future__ import annotations

import pytest

from biomed_platform.core.services.ingestion.in_memory_queue import InMemoryIngestionQueue


@pytest.mark.asyncio
async def test_queue_enqueue_then_dequeue_preserves_order() -> None:
    # Given a queue with capacity
    q = InMemoryIngestionQueue(max_size=2)

    # When enqueueing two jobs and dequeuing them
    await q.enqueue("j1")
    await q.enqueue("j2")

    out1 = await q.dequeue()
    out2 = await q.dequeue()

    # Then order is FIFO and queue size updates
    assert out1 == "j1"
    assert out2 == "j2"
    assert q.size() == 0


@pytest.mark.asyncio
async def test_queue_enqueue_raises_when_full() -> None:
    # Given a queue with capacity 1
    q = InMemoryIngestionQueue(max_size=1)

    # When enqueueing beyond capacity
    await q.enqueue("j1")

    # Then QueueFull is raised by the underlying asyncio queue
    with pytest.raises(Exception):
        await q.enqueue("j2")
