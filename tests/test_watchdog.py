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

"""Tests for ``weakincentives.watchdog``."""

from __future__ import annotations

from datetime import timedelta

from weakincentives.clock import FakeClock
from weakincentives.watchdog import Heartbeat, HeartbeatStatus, Watchdog


def _heartbeat(clock: FakeClock, name: str = "worker") -> Heartbeat:
    return Heartbeat(name=name, clock_wall=clock, clock_mono=clock)


def test_heartbeat_records_beat_metadata() -> None:
    clock = FakeClock()
    heartbeat = _heartbeat(clock)
    heartbeat.beat(detail="step=1")
    status = heartbeat.status()
    assert status.name == "worker"
    assert status.detail == "step=1"
    assert status.last_beat_at == clock.utcnow()


def test_heartbeat_silent_for_grows_with_clock() -> None:
    clock = FakeClock()
    heartbeat = _heartbeat(clock)
    heartbeat.beat()
    clock.advance(7)
    assert heartbeat.silent_for() == timedelta(seconds=7)


def test_watchdog_fires_on_stuck_heartbeat() -> None:
    clock = FakeClock()
    fired: list[HeartbeatStatus] = []
    heartbeat = _heartbeat(clock, name="hot")
    heartbeat.beat()
    watchdog = Watchdog(deadline=timedelta(seconds=2), on_stuck=fired.append)
    watchdog.register(heartbeat)
    assert watchdog.tick() == ()
    clock.advance(5)
    stuck = watchdog.tick()
    assert len(stuck) == 1
    assert stuck[0].name == "hot"
    assert fired[0].name == "hot"
    assert watchdog.is_stuck("hot")


def test_watchdog_clears_stuck_state_when_heartbeat_resumes() -> None:
    clock = FakeClock()
    fired: list[HeartbeatStatus] = []
    heartbeat = _heartbeat(clock)
    heartbeat.beat()
    watchdog = Watchdog(deadline=timedelta(seconds=2), on_stuck=fired.append)
    watchdog.register(heartbeat)
    clock.advance(5)
    watchdog.tick()
    assert watchdog.is_stuck("worker")
    heartbeat.beat()
    watchdog.tick()
    assert not watchdog.is_stuck("worker")


def test_watchdog_does_not_double_fire_for_same_silence() -> None:
    clock = FakeClock()
    fired: list[HeartbeatStatus] = []
    heartbeat = _heartbeat(clock)
    heartbeat.beat()
    watchdog = Watchdog(deadline=timedelta(seconds=2), on_stuck=fired.append)
    watchdog.register(heartbeat)
    clock.advance(5)
    watchdog.tick()
    watchdog.tick()
    assert len(fired) == 1


def test_watchdog_continues_after_callback_error() -> None:
    clock = FakeClock()

    def failing(_status: HeartbeatStatus) -> None:
        raise RuntimeError("nope")

    heartbeat = _heartbeat(clock)
    heartbeat.beat()
    watchdog = Watchdog(deadline=timedelta(seconds=1), on_stuck=failing)
    watchdog.register(heartbeat)
    clock.advance(5)
    watchdog.tick()
    assert watchdog.is_stuck("worker")


def test_watchdog_deregister_clears_state() -> None:
    clock = FakeClock()
    heartbeat = _heartbeat(clock)
    heartbeat.beat()
    watchdog = Watchdog(deadline=timedelta(seconds=1), on_stuck=lambda _s: None)
    watchdog.register(heartbeat)
    clock.advance(5)
    watchdog.tick()
    assert watchdog.is_stuck("worker")
    watchdog.deregister("worker")
    assert not watchdog.is_stuck("worker")
    assert watchdog.heartbeats() == ()
