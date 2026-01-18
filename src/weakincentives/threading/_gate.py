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

"""Gate implementations for thread signaling."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from weakincentives.clock import Clock


@dataclass
class SystemGate:
    """Production gate using threading.Event.

    Example::

        gate = SystemGate()

        # In worker thread
        if gate.wait(timeout=5.0):
            print("Gate opened!")

        # In control thread
        gate.set()  # Release waiters
    """

    _event: threading.Event = field(default_factory=threading.Event, repr=False)

    def set(self) -> None:
        """Open the gate, releasing all waiters."""
        self._event.set()

    def clear(self) -> None:
        """Close the gate."""
        self._event.clear()

    def is_set(self) -> bool:
        """Return True if the gate is open."""
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the gate opens or timeout expires."""
        return self._event.wait(timeout=timeout)


@dataclass
class FakeGate:
    """Test gate with optional clock integration.

    When a clock is provided, wait() advances the clock by the timeout
    duration instead of blocking. This enables testing time-dependent
    code without real delays.

    Example::

        from weakincentives.clock import FakeClock

        clock = FakeClock()
        gate = FakeGate(clock=clock)

        # wait() returns immediately, advances clock
        result = gate.wait(timeout=5.0)
        assert result is False  # Gate not set
        assert clock.monotonic() == 5.0  # Time advanced

        gate.set()
        assert gate.wait(timeout=1.0) is True
    """

    clock: Clock | None = None
    _is_set: bool = field(default=False, repr=False)
    _wait_count: int = field(default=0, repr=False)

    def set(self) -> None:
        """Open the gate."""
        self._is_set = True

    def clear(self) -> None:
        """Close the gate."""
        self._is_set = False

    def is_set(self) -> bool:
        """Return True if the gate is open."""
        return self._is_set

    def wait(self, timeout: float | None = None) -> bool:
        """Check gate state, advancing clock if timeout provided.

        Does not block. If the gate is not set and a timeout is provided,
        advances the clock by the timeout duration.
        """
        self._wait_count += 1

        if self._is_set:
            return True

        # Advance clock by timeout if gate is not set
        if timeout is not None and self.clock is not None:
            self.clock.sleep(timeout)

        return self._is_set

    @property
    def wait_count(self) -> int:
        """Number of times wait() was called."""
        return self._wait_count

    def reset(self) -> None:
        """Reset state for reuse in tests."""
        self._is_set = False
        self._wait_count = 0


__all__ = [
    "FakeGate",
    "SystemGate",
]
