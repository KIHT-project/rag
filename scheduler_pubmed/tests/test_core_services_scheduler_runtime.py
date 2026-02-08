from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

import scheduler_pubmed.src.core.services.scheduler_runtime as runtime_mod
from scheduler_pubmed.src.core.domains.scheduler import (
    RunStatus,
    SchedulerRunCreated,
    SchedulerStatus,
    TriggerType,
)
from scheduler_pubmed.src.core.services.scheduler_runtime import SchedulerRuntimeService


class _FakeUseCase:
    def __init__(self) -> None:
        self.trigger_types: list[TriggerType] = []
        self.executions: list[tuple[object, int | None]] = []

    async def trigger_run(self, *, trigger_type: TriggerType) -> SchedulerRunCreated:
        self.trigger_types.append(trigger_type)
        return SchedulerRunCreated(
            run_id=uuid4(), status=RunStatus.RUNNING, started_at=datetime.now(UTC)
        )

    async def execute_run(self, *, run_id, reldate_days: int | None = None):
        self.executions.append((run_id, reldate_days))

    async def get_status(
        self, *, enabled: bool, utc_schedule: list[str]
    ) -> SchedulerStatus:
        return SchedulerStatus(
            enabled=enabled,
            utc_schedule=utc_schedule,
            next_run_at=datetime.now(UTC),
            last_run_at=None,
            last_run_status=None,
        )


@pytest.mark.asyncio
async def test_trigger_manual_run_creates_background_execution() -> None:
    use_case = _FakeUseCase()
    runtime = SchedulerRuntimeService(
        use_case=use_case,  # type: ignore[arg-type]
        enabled=True,
        utc_schedule=["02:00"],
        automatic_schedule_enabled=False,
    )

    try:
        run = await runtime.trigger_manual_run()
        await asyncio.sleep(0)
        assert run.status == RunStatus.RUNNING
        assert use_case.trigger_types == [TriggerType.MANUAL]
        assert use_case.executions == [(run.run_id, None)]
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_get_status_uses_runtime_config() -> None:
    use_case = _FakeUseCase()
    runtime = SchedulerRuntimeService(
        use_case=use_case,  # type: ignore[arg-type]
        enabled=True,
        utc_schedule=["02:00", "14:00"],
        automatic_schedule_enabled=False,
    )

    status = await runtime.get_status()

    assert status.enabled is True
    assert status.utc_schedule == ["02:00", "14:00"]


@pytest.mark.asyncio
async def test_start_stop_with_auto_loop_disabled() -> None:
    use_case = _FakeUseCase()
    runtime = SchedulerRuntimeService(
        use_case=use_case,  # type: ignore[arg-type]
        enabled=True,
        utc_schedule=["02:00"],
        automatic_schedule_enabled=False,
    )

    await runtime.start()
    await runtime.stop()

    assert use_case.trigger_types == []


@pytest.mark.asyncio
async def test_start_is_idempotent_and_stop_cancels_schedule_loop() -> None:
    use_case = _FakeUseCase()
    runtime = SchedulerRuntimeService(
        use_case=use_case,  # type: ignore[arg-type]
        enabled=True,
        utc_schedule=["02:00"],
        automatic_schedule_enabled=True,
        loop_interval_seconds=60.0,
    )

    await runtime.start()
    first_task = runtime._schedule_loop_task  # noqa: SLF001
    assert first_task is not None

    await runtime.start()
    assert runtime._schedule_loop_task is first_task  # noqa: SLF001

    await runtime.stop()
    assert runtime._schedule_loop_task is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_stop_cancels_background_run_tasks() -> None:
    use_case = _FakeUseCase()
    runtime = SchedulerRuntimeService(
        use_case=use_case,  # type: ignore[arg-type]
        enabled=True,
        utc_schedule=["02:00"],
        automatic_schedule_enabled=False,
    )

    pending = asyncio.create_task(asyncio.sleep(60.0))
    runtime._run_tasks.add(pending)  # noqa: SLF001
    await runtime.stop()

    assert runtime._run_tasks == set()  # noqa: SLF001
    assert pending.cancelled()


@pytest.mark.asyncio
async def test_trigger_scheduled_run_creates_background_execution() -> None:
    use_case = _FakeUseCase()
    runtime = SchedulerRuntimeService(
        use_case=use_case,  # type: ignore[arg-type]
        enabled=True,
        utc_schedule=["02:00"],
        automatic_schedule_enabled=False,
    )

    try:
        run = await runtime.trigger_scheduled_run()
        await asyncio.sleep(0)
        assert run.status == RunStatus.RUNNING
        assert use_case.trigger_types == [TriggerType.SCHEDULED]
        assert use_case.executions == [(run.run_id, None)]
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_trigger_manual_run_passes_reldate_override() -> None:
    use_case = _FakeUseCase()
    runtime = SchedulerRuntimeService(
        use_case=use_case,  # type: ignore[arg-type]
        enabled=True,
        utc_schedule=["02:00"],
        automatic_schedule_enabled=False,
    )

    try:
        run = await runtime.trigger_manual_run(reldate_days=365)
        await asyncio.sleep(0)
        assert use_case.executions == [(run.run_id, 365)]
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_schedule_loop_triggers_slot_once_per_minute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = _FakeUseCase()
    runtime = SchedulerRuntimeService(
        use_case=use_case,  # type: ignore[arg-type]
        enabled=True,
        utc_schedule=["02:00"],
        automatic_schedule_enabled=True,
        loop_interval_seconds=1.0,
    )

    fixed_now = datetime(2026, 2, 8, 2, 0, tzinfo=UTC)

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return fixed_now

    monkeypatch.setattr(runtime_mod, "datetime", _FixedDateTime)

    trigger_count = 0

    async def _fake_trigger() -> SchedulerRunCreated:
        nonlocal trigger_count
        trigger_count += 1
        return SchedulerRunCreated(
            run_id=uuid4(), status=RunStatus.RUNNING, started_at=fixed_now
        )

    sleep_calls = 0

    async def _fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(runtime, "trigger_scheduled_run", _fake_trigger)
    monkeypatch.setattr(runtime_mod.asyncio, "sleep", _fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await runtime._schedule_loop()  # noqa: SLF001

    assert trigger_count == 1


@pytest.mark.asyncio
async def test_schedule_loop_logs_iteration_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = _FakeUseCase()
    runtime = SchedulerRuntimeService(
        use_case=use_case,  # type: ignore[arg-type]
        enabled=True,
        utc_schedule=["02:00"],
        automatic_schedule_enabled=True,
    )

    fixed_now = datetime(2026, 2, 8, 2, 0, tzinfo=UTC)

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return fixed_now

    async def _raise_in_trigger() -> SchedulerRunCreated:
        raise RuntimeError("boom")

    async def _cancel_sleep(_seconds: float) -> None:
        raise asyncio.CancelledError

    logged_messages: list[str] = []

    def _fake_log_exception(message: str) -> None:
        logged_messages.append(message)

    monkeypatch.setattr(runtime_mod, "datetime", _FixedDateTime)
    monkeypatch.setattr(runtime, "trigger_scheduled_run", _raise_in_trigger)
    monkeypatch.setattr(runtime_mod.asyncio, "sleep", _cancel_sleep)
    monkeypatch.setattr(runtime_mod.log, "exception", _fake_log_exception)

    with pytest.raises(asyncio.CancelledError):
        await runtime._schedule_loop()  # noqa: SLF001

    assert logged_messages == ["Scheduler loop iteration failed"]


@pytest.mark.asyncio
async def test_schedule_loop_re_raises_cancelled_error_from_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = _FakeUseCase()
    runtime = SchedulerRuntimeService(
        use_case=use_case,  # type: ignore[arg-type]
        enabled=True,
        utc_schedule=["02:00"],
        automatic_schedule_enabled=True,
    )

    def _cancelled_cleanup() -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(runtime, "_cleanup_fired_slots", _cancelled_cleanup)

    with pytest.raises(asyncio.CancelledError):
        await runtime._schedule_loop()  # noqa: SLF001


def test_cleanup_fired_slots_keeps_only_today_when_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = _FakeUseCase()
    runtime = SchedulerRuntimeService(
        use_case=use_case,  # type: ignore[arg-type]
        enabled=True,
        utc_schedule=["02:00"],
        automatic_schedule_enabled=True,
    )

    fixed_now = datetime(2026, 2, 8, 2, 0, tzinfo=UTC)

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return fixed_now

    runtime._fired_slots = {f"2026-02-08 02:{i:02d}" for i in range(8)}  # noqa: SLF001
    runtime._fired_slots.update(
        {f"2026-02-07 02:{i:02d}" for i in range(2)}
    )  # noqa: SLF001

    monkeypatch.setattr(runtime_mod, "datetime", _FixedDateTime)

    runtime._cleanup_fired_slots()  # noqa: SLF001

    assert runtime._fired_slots == {
        f"2026-02-08 02:{i:02d}" for i in range(8)
    }  # noqa: SLF001
