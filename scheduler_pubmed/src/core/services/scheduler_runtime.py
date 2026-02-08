from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID

from scheduler_pubmed.src.common.logging import get_logger
from scheduler_pubmed.src.core.domains.scheduler import (
    SchedulerRunCreated,
    SchedulerStatus,
    TriggerType,
)
from scheduler_pubmed.src.core.use_cases.scheduler import SchedulerOrchestrationUseCase

log = get_logger(__name__)


class SchedulerRuntimeService:
    def __init__(
        self,
        *,
        use_case: SchedulerOrchestrationUseCase,
        enabled: bool,
        utc_schedule: list[str],
        automatic_schedule_enabled: bool,
        loop_interval_seconds: float = 20.0,
    ) -> None:
        self._use_case = use_case
        self._enabled = bool(enabled)
        self._utc_schedule = list(utc_schedule)
        self._automatic_schedule_enabled = bool(automatic_schedule_enabled)
        self._loop_interval_seconds = max(1.0, float(loop_interval_seconds))

        self._run_tasks: set[asyncio.Task[None]] = set()
        self._schedule_loop_task: asyncio.Task[None] | None = None
        self._fired_slots: set[str] = set()

    async def start(self) -> None:
        if not self._enabled or not self._automatic_schedule_enabled:
            return
        if self._schedule_loop_task is not None:
            return
        self._schedule_loop_task = asyncio.create_task(self._schedule_loop())

    async def stop(self) -> None:
        if self._schedule_loop_task is not None:
            self._schedule_loop_task.cancel()
            await asyncio.gather(self._schedule_loop_task, return_exceptions=True)
            self._schedule_loop_task = None

        if self._run_tasks:
            for task in list(self._run_tasks):
                task.cancel()
            await asyncio.gather(*self._run_tasks, return_exceptions=True)
            self._run_tasks.clear()

    async def trigger_manual_run(self, *, reldate_days: int | None = None) -> SchedulerRunCreated:
        run = await self._use_case.trigger_run(trigger_type=TriggerType.MANUAL)
        self._start_run_task(run_id=run.run_id, reldate_days=reldate_days)
        return run

    async def trigger_scheduled_run(self) -> SchedulerRunCreated:
        run = await self._use_case.trigger_run(trigger_type=TriggerType.SCHEDULED)
        self._start_run_task(run_id=run.run_id, reldate_days=None)
        return run

    async def get_status(self) -> SchedulerStatus:
        return await self._use_case.get_status(
            enabled=self._enabled, utc_schedule=self._utc_schedule
        )

    def _start_run_task(self, *, run_id: UUID, reldate_days: int | None) -> None:
        task = asyncio.create_task(
            self._use_case.execute_run(
                run_id=run_id,
                reldate_days=reldate_days,
            )
        )
        self._run_tasks.add(task)
        task.add_done_callback(self._run_tasks.discard)

    async def _schedule_loop(self) -> None:
        while True:
            try:
                self._cleanup_fired_slots()
                now = datetime.now(UTC)
                now_slot = now.strftime("%H:%M")
                if now_slot in self._utc_schedule:
                    slot_key = now.strftime("%Y-%m-%d %H:%M")
                    if slot_key not in self._fired_slots:
                        self._fired_slots.add(slot_key)
                        await self.trigger_scheduled_run()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Scheduler loop iteration failed")

            await asyncio.sleep(self._loop_interval_seconds)

    def _cleanup_fired_slots(self) -> None:
        if len(self._fired_slots) < 10:
            return
        today_prefix = datetime.now(UTC).strftime("%Y-%m-%d")
        self._fired_slots = {key for key in self._fired_slots if key.startswith(today_prefix)}
