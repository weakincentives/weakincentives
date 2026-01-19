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

"""Threading test helpers and fixtures.

This module provides test fixtures for the threading primitives defined in
:mod:`weakincentives.threading`. Use these fixtures for deterministic
testing of concurrent code without real threads.

Example::

    from tests.helpers.threading import fake_executor, fake_gate

    def test_batch_processor(fake_executor: FakeExecutor) -> None:
        processor = BatchProcessor(executor=fake_executor)
        results = processor.process([1, 2, 3])
        assert results == [2, 4, 6]
        assert len(fake_executor.submitted) == 3
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

# Re-export fake implementations for convenience
from weakincentives.threading import (
    FakeBackgroundWorker,
    FakeCheckpoint,
    FakeExecutor,
    FakeGate,
    FakeLatch,
    FakeScheduler,
)


@pytest.fixture
def fake_executor() -> FakeExecutor:
    """Provide a fresh FakeExecutor for deterministic testing.

    The fake executor runs tasks synchronously in the calling thread,
    eliminating race conditions and enabling deterministic assertions.

    Example::

        def test_parallel_processing(fake_executor: FakeExecutor) -> None:
            processor = Processor(executor=fake_executor)
            results = processor.process([1, 2, 3])

            # All tasks completed synchronously
            assert len(fake_executor.submitted) == 3
            assert results == [2, 4, 6]
    """
    return FakeExecutor()


@pytest.fixture
def fake_gate() -> FakeGate:
    """Provide a fresh FakeGate for deterministic testing.

    The fake gate does not block on wait(). If a clock is provided,
    wait() advances the clock by the timeout duration.

    Example::

        def test_shutdown_signal(fake_gate: FakeGate) -> None:
            worker = Worker(stop_signal=fake_gate)
            worker.start()

            fake_gate.set()
            assert fake_gate.is_set()
    """
    return FakeGate()


@pytest.fixture
def fake_checkpoint() -> FakeCheckpoint:
    """Provide a fresh FakeCheckpoint for testing cooperative yielding.

    The fake checkpoint tracks yield and check counts for assertions.

    Example::

        def test_cancellation(fake_checkpoint: FakeCheckpoint) -> None:
            def task():
                for _ in range(10):
                    fake_checkpoint.check()
                    fake_checkpoint.yield_()

            task()
            assert fake_checkpoint.check_count == 10
            assert fake_checkpoint.yield_count == 10
    """
    return FakeCheckpoint()


@pytest.fixture
def fake_scheduler() -> FakeScheduler:
    """Provide a fresh FakeScheduler for testing task interleaving.

    The fake scheduler enables step-by-step execution of tasks.

    Example::

        def test_interleaving(fake_scheduler: FakeScheduler) -> None:
            order = []

            def task_a():
                order.append("a1")
                fake_scheduler.yield_()
                order.append("a2")

            fake_scheduler.schedule(task_a)
            fake_scheduler.run_until_complete()
            assert order == ["a1", "a2"]
    """
    return FakeScheduler()


@pytest.fixture
def fake_latch_factory() -> Callable[[int], FakeLatch]:
    """Provide a factory for creating FakeLatch instances.

    Example::

        def test_coordination(fake_latch_factory) -> None:
            latch = fake_latch_factory(3)
            assert latch.count == 3

            latch.count_down()
            latch.count_down()
            latch.count_down()
            assert latch.await_(timeout=1.0) is True
    """

    def factory(count: int) -> FakeLatch:
        return FakeLatch(count)

    return factory


__all__ = [
    "FakeBackgroundWorker",
    "FakeCheckpoint",
    "FakeExecutor",
    "FakeGate",
    "FakeLatch",
    "FakeScheduler",
    "fake_checkpoint",
    "fake_executor",
    "fake_gate",
    "fake_latch_factory",
    "fake_scheduler",
]
