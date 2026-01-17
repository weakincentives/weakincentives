# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyright: reportImportCycles=false
# pyright: reportPrivateUsage=false
# pyright: reportUnknownVariableType=false

"""Scheduler for deterministic thread execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ._types import CheckpointInfo, Deadlock, ScheduleResult, StepResult
from ._worker import WorkerThread


@dataclass(slots=True)
class Scheduler:
    """Controls deterministic execution of worker threads.

    Only one thread runs at a time. The scheduler decides which
    thread to resume next based on the provided schedule or
    round-robin by default.
    """

    threads: dict[str, WorkerThread] = field(default_factory=dict)
    schedule: Sequence[str] | None = field(default=None)
    _position: int = field(default=0)
    _trace: list[str] = field(default_factory=list)
    _checkpoints: list[str | None] = field(default_factory=list)
    _current_thread_name: str | None = field(default=None)

    def register(self, name: str, target: Callable[[], None]) -> WorkerThread:
        """Register a worker thread."""
        if name in self.threads:
            msg = f"Thread {name!r} already registered"
            raise ValueError(msg)

        worker = WorkerThread(name=name, _target=target, _scheduler=self)
        self.threads[name] = worker
        return worker

    def checkpoint(self, name: str | None) -> None:
        """Called by checkpoint() when a thread yields.

        This method is invoked from within a worker thread.
        """
        thread_name = self._current_thread_name
        if thread_name is None:
            return

        worker = self.threads.get(thread_name)
        if worker is not None:  # pragma: no branch
            self._checkpoints.append(name)
            worker.pause_at_checkpoint(name)

    def step(self) -> StepResult:
        """Execute one scheduling step.

        Returns:
            StepResult indicating what happened.

        Raises:
            Exception: If a worker thread raised an exception.
        """
        # Check for errors from previous step (defensive - usually caught after resume)
        for worker in self.threads.values():
            if worker.error is not None:  # pragma: no cover - caught after resume
                raise worker.error

        runnable = [n for n, t in self.threads.items() if t.can_run]

        if not runnable:
            if any(
                not t.done for t in self.threads.values()
            ):  # pragma: no cover - requires blocked threads
                return StepResult.DEADLOCK
            return StepResult.ALL_DONE

        name = self._pick_next(runnable)
        self._trace.append(name)
        self._current_thread_name = name
        self.threads[name].resume_until_checkpoint()

        # Check for error after resuming
        worker = self.threads[name]
        if worker.error is not None:
            raise worker.error

        return StepResult.CONTINUE

    def _pick_next(self, runnable: list[str]) -> str:
        """Pick the next thread to run."""
        if self.schedule is not None:
            # Scripted schedule
            while self._position < len(self.schedule):
                name = self.schedule[self._position]
                self._position += 1
                if name == "*":
                    # Wildcard: pick any runnable
                    return runnable[0]
                if name in runnable:
                    return name
                # Thread not runnable, skip to next in schedule
            # Schedule exhausted, fall back to round-robin
            return runnable[0]

        # Round-robin: pick first runnable in registration order
        for name in self.threads:
            if name in runnable:
                return name

        return runnable[0]  # pragma: no cover - defensive

    def run(
        self,
        on_deadlock: Literal["raise", "return"] = "raise",
    ) -> ScheduleResult:
        """Run all threads to completion.

        Args:
            on_deadlock: How to handle deadlock. "raise" raises Deadlock,
                "return" returns a ScheduleResult with deadlocked=True.

        Returns:
            ScheduleResult with the execution trace.

        Raises:
            Deadlock: If on_deadlock="raise" and threads deadlock.
        """
        # Start all threads
        for worker in self.threads.values():
            if not worker._started:  # pragma: no branch
                worker.start()

        # Run until done or deadlock
        while True:
            result = self.step()
            if result == StepResult.CONTINUE:
                continue
            if result == StepResult.ALL_DONE:
                return ScheduleResult(
                    schedule=tuple(self._trace),
                    deadlocked=False,
                    checkpoints=tuple(self._checkpoints),
                )
            if (
                result == StepResult.DEADLOCK
            ):  # pragma: no cover - requires blocked threads
                if on_deadlock == "raise":
                    blocked = {
                        n: "waiting" for n, t in self.threads.items() if not t.done
                    }
                    raise Deadlock(
                        blocked=blocked,
                        schedule_so_far=tuple(self._trace),
                    )
                return ScheduleResult(
                    schedule=tuple(self._trace),
                    deadlocked=True,
                    checkpoints=tuple(self._checkpoints),
                )

    @property
    def trace(self) -> tuple[str, ...]:
        """Execution trace so far."""
        return tuple(self._trace)

    @property
    def checkpoints(self) -> tuple[CheckpointInfo, ...]:
        """Checkpoint info collected during execution."""
        return tuple(
            CheckpointInfo(
                thread_name=self._trace[i] if i < len(self._trace) else "",
                checkpoint_name=cp,
                sequence=i,
            )
            for i, cp in enumerate(self._checkpoints)
        )


__all__ = ["Scheduler"]
