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

"""Tests for ``weakincentives.lifecycle``."""

from __future__ import annotations

import threading

import pytest

from weakincentives.lifecycle import (
    LoopGroup,
    LoopGroupError,
    ShutdownCoordinator,
)


def test_shutdown_runs_callbacks_in_reverse_order() -> None:
    seen: list[str] = []
    coord = ShutdownCoordinator()
    coord.register(lambda: seen.append("a"))
    coord.register(lambda: seen.append("b"))
    coord.register(lambda: seen.append("c"))
    coord.shutdown()
    assert seen == ["c", "b", "a"]


def test_shutdown_is_idempotent() -> None:
    seen: list[str] = []
    coord = ShutdownCoordinator()
    coord.register(lambda: seen.append("x"))
    coord.shutdown()
    coord.shutdown()
    assert seen == ["x"]


def test_register_after_shutdown_runs_immediately() -> None:
    coord = ShutdownCoordinator()
    coord.shutdown()
    seen: list[str] = []
    coord.register(lambda: seen.append("late"))
    assert seen == ["late"]


def test_shutdown_continues_through_callback_errors() -> None:
    seen: list[str] = []

    def boom() -> None:
        raise RuntimeError("nope")

    coord = ShutdownCoordinator()
    coord.register(lambda: seen.append("a"))
    coord.register(boom)
    coord.register(lambda: seen.append("c"))
    coord.shutdown()
    assert seen == ["c", "a"]
    assert coord.is_complete


def test_event_signals_shutting_down() -> None:
    coord = ShutdownCoordinator()
    assert not coord.is_shutting_down
    assert not coord.event.is_set()
    coord.shutdown()
    assert coord.is_shutting_down
    assert coord.event.is_set()


def test_loop_group_runs_workers_until_shutdown() -> None:
    coord = ShutdownCoordinator()
    group = LoopGroup(coordinator=coord, name="test")
    counters: list[int] = [0, 0]

    def worker(index: int) -> threading.Thread:
        def run(event: threading.Event) -> None:
            while not event.is_set():
                counters[index] += 1
                event.wait(timeout=0.01)

        return threading.Thread(target=run, args=(coord.event,))

    def adapter(index: int) -> None:
        def run(event: threading.Event) -> None:
            while not event.is_set():
                counters[index] += 1
                event.wait(timeout=0.01)

        group.spawn(run, name=f"worker-{index}")

    adapter(0)
    adapter(1)
    coord.shutdown()
    group.join(timeout=1.0)
    assert all(c >= 1 for c in counters)


def test_loop_group_records_worker_failures() -> None:
    coord = ShutdownCoordinator()
    group = LoopGroup(coordinator=coord)

    def fail(event: threading.Event) -> None:
        del event
        raise RuntimeError("worker boom")

    group.spawn(fail)
    with pytest.raises(LoopGroupError, match="failed"):
        group.join(timeout=1.0)


def test_loop_group_alive_workers_lists_running() -> None:
    coord = ShutdownCoordinator()
    group = LoopGroup(coordinator=coord)

    def run(event: threading.Event) -> None:
        event.wait()

    group.spawn(run, name="hot")
    assert "hot" in list(group.alive_workers())
    coord.shutdown()
    group.join(timeout=1.0)
