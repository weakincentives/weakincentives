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

"""Injectable threading primitives for testable concurrent code.

This module provides protocols and implementations for threading operations,
enabling deterministic testing without real threads or blocking.

There are three categories of primitives:

- **Execution**: Submit and manage work (Executor, BackgroundWorker)
- **Coordination**: Synchronize between threads (Gate, Latch, CallbackRegistry)
- **Cooperation**: Yield control voluntarily (Checkpoint, CancellationToken, Scheduler)

Example (production)::

    from weakincentives.threading import SYSTEM_EXECUTOR, SystemGate

    # Execute work in thread pool
    future = SYSTEM_EXECUTOR.submit(lambda: expensive_computation())

    # Signal between threads
    gate = SystemGate()
    gate.wait(timeout=5.0)

Example (testing)::

    from weakincentives.threading import FakeExecutor, FakeGate

    executor = FakeExecutor()
    future = executor.submit(lambda: 42)
    assert future.done()  # Completed synchronously

    gate = FakeGate()
    gate.set()
    assert gate.is_set()
"""

from __future__ import annotations

from typing import Final

from weakincentives.threading._callback_registry import CallbackRegistry
from weakincentives.threading._cancellation import (
    FakeCheckpoint,
    SimpleCancellationToken,
    SystemCheckpoint,
)
from weakincentives.threading._executor import (
    CompletedFuture,
    FakeExecutor,
    SystemExecutor,
)
from weakincentives.threading._gate import FakeGate, SystemGate
from weakincentives.threading._latch import FakeLatch, Latch
from weakincentives.threading._scheduler import FakeScheduler, FifoScheduler
from weakincentives.threading._types import (
    CancellationToken,
    CancelledException,
    Checkpoint,
    Executor,
    Future,
    Gate,
    Scheduler,
)
from weakincentives.threading._worker import BackgroundWorker, FakeBackgroundWorker

# Module-level singleton for production use
SYSTEM_EXECUTOR: Final[Executor] = SystemExecutor()
"""Default system executor instance.

Uses a thread pool that creates workers on demand.
Tests can inject FakeExecutor instead for deterministic behavior.
"""

__all__ = [
    "SYSTEM_EXECUTOR",
    "BackgroundWorker",
    "CallbackRegistry",
    "CancellationToken",
    "CancelledException",
    "Checkpoint",
    "CompletedFuture",
    "Executor",
    "FakeBackgroundWorker",
    "FakeCheckpoint",
    "FakeExecutor",
    "FakeGate",
    "FakeLatch",
    "FakeScheduler",
    "FifoScheduler",
    "Future",
    "Gate",
    "Latch",
    "Scheduler",
    "SimpleCancellationToken",
    "SystemCheckpoint",
    "SystemExecutor",
    "SystemGate",
]
