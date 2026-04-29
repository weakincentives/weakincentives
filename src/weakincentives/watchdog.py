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

"""Liveness tracking for long-running workers.

A :class:`Heartbeat` records the last time a worker reported activity.
A :class:`Watchdog` polls a set of heartbeats and fires an ``on_stuck``
callback for any whose silence exceeds a configured deadline. Together
they let an ``AgentLoop``, a mailbox worker, or any other long-lived
component prove it's still alive.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import final

from weakincentives.clock import SYSTEM_CLOCK, MonotonicClock, WallClock
from weakincentives.logging import get_logger

__all__ = [
    "Heartbeat",
    "HeartbeatStatus",
    "Watchdog",
]

_LOG = get_logger("weakincentives.watchdog")


@final
@dataclass(frozen=True, slots=True)
class HeartbeatStatus:
    """Snapshot of a heartbeat's last-known activity."""

    name: str
    last_beat_at: datetime
    last_beat_monotonic: float
    detail: str = ""


@final
class Heartbeat:
    """A single named heartbeat. Thread-safe."""

    def __init__(
        self,
        *,
        name: str,
        clock_wall: WallClock = SYSTEM_CLOCK,
        clock_mono: MonotonicClock = SYSTEM_CLOCK,
    ) -> None:
        super().__init__()
        self._name = name
        self._wall = clock_wall
        self._mono = clock_mono
        self._lock = threading.Lock()
        self._last_wall = clock_wall.utcnow()
        self._last_mono = clock_mono.monotonic()
        self._detail = ""

    @property
    def name(self) -> str:
        return self._name

    def beat(self, *, detail: str = "") -> None:
        with self._lock:
            self._last_wall = self._wall.utcnow()
            self._last_mono = self._mono.monotonic()
            self._detail = detail

    def status(self) -> HeartbeatStatus:
        with self._lock:
            return HeartbeatStatus(
                name=self._name,
                last_beat_at=self._last_wall,
                last_beat_monotonic=self._last_mono,
                detail=self._detail,
            )

    def silent_for(self) -> timedelta:
        with self._lock:
            return timedelta(seconds=self._mono.monotonic() - self._last_mono)


@dataclass(slots=True)
class Watchdog:
    """Periodically check a set of heartbeats and fire on stuck ones.

    The watchdog never blocks; it simply offers a :meth:`tick` method
    that the caller invokes from whichever scheduling primitive they
    already have (a thread, an asyncio task, a periodic timer, …).
    """

    deadline: timedelta
    on_stuck: Callable[[HeartbeatStatus], None]
    _heartbeats: dict[str, Heartbeat] = field(default_factory=lambda: {}, repr=False)
    _stuck: set[str] = field(default_factory=lambda: set(), repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def register(self, heartbeat: Heartbeat) -> None:
        with self._lock:
            self._heartbeats[heartbeat.name] = heartbeat

    def deregister(self, name: str) -> None:
        with self._lock:
            _ = self._heartbeats.pop(name, None)
            self._stuck.discard(name)

    def tick(self) -> tuple[HeartbeatStatus, ...]:
        """Inspect every registered heartbeat. Returns the stuck ones."""
        with self._lock:
            heartbeats = list(self._heartbeats.values())
        stuck: list[HeartbeatStatus] = []
        for heartbeat in heartbeats:
            if heartbeat.silent_for() < self.deadline:
                with self._lock:
                    self._stuck.discard(heartbeat.name)
                continue
            with self._lock:
                already_stuck = heartbeat.name in self._stuck
                self._stuck.add(heartbeat.name)
            if already_stuck:
                continue
            status = heartbeat.status()
            stuck.append(status)
            try:
                self.on_stuck(status)
            except Exception as error:
                _LOG.exception(
                    "watchdog.on_stuck_failed",
                    heartbeat=heartbeat.name,
                    error=str(error),
                )
        return tuple(stuck)

    def is_stuck(self, name: str) -> bool:
        with self._lock:
            return name in self._stuck

    def heartbeats(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._heartbeats.keys())
