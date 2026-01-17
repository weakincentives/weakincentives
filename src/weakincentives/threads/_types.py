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

"""Core types for thread testing framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from collections.abc import Mapping


class StepResult(Enum):
    """Result of a single scheduler step."""

    CONTINUE = auto()
    ALL_DONE = auto()
    DEADLOCK = auto()


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    """Result of running a schedule to completion."""

    schedule: tuple[str, ...]
    deadlocked: bool
    checkpoints: tuple[str | None, ...] = ()

    @override
    def __str__(self) -> str:
        status = "DEADLOCK" if self.deadlocked else "OK"
        return f"ScheduleResult({status}, {list(self.schedule)})"


@dataclass(frozen=True, slots=True)
class Deadlock(Exception):
    """Raised when all threads are blocked and none can proceed."""

    blocked: Mapping[str, str]
    schedule_so_far: tuple[str, ...]

    @override
    def __str__(self) -> str:
        blocked_info = ", ".join(f"{k}: {v}" for k, v in self.blocked.items())
        return (
            f"Deadlock detected. Blocked threads: {{{blocked_info}}}. "
            f"Schedule so far: {list(self.schedule_so_far)}"
        )


@dataclass(slots=True)
class CheckpointInfo:
    """Information about a checkpoint hit."""

    thread_name: str
    checkpoint_name: str | None
    sequence: int = field(default=0)


__all__ = [
    "CheckpointInfo",
    "Deadlock",
    "ScheduleResult",
    "StepResult",
]
