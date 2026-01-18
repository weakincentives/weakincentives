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

"""Latch implementations for one-shot synchronization."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from weakincentives.clock import Clock


@dataclass
class Latch:
    """One-shot barrier that releases when count reaches zero.

    Unlike a barrier which can be reused, a latch is a one-time
    synchronization point. Once the count reaches zero, all
    waiters are released and subsequent waits return immediately.

    Example::

        latch = Latch(3)

        def worker():
            do_work()
            latch.count_down()

        # Start 3 workers...

        # Block until all workers complete
        if latch.await_(timeout=30.0):
            print("All workers done")
    """

    initial_count: int
    _count: int = field(init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _condition: threading.Condition = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.initial_count < 0:
            msg = "Latch count must be non-negative"
            raise ValueError(msg)
        self._count = self.initial_count
        self._condition = threading.Condition(self._lock)

    def count_down(self) -> None:
        """Decrement the count, releasing waiters if zero.

        Does nothing if count is already zero.
        """
        with self._condition:
            if self._count > 0:
                self._count -= 1
                if self._count == 0:
                    self._condition.notify_all()

    def await_(self, timeout: float | None = None) -> bool:
        """Wait until the count reaches zero.

        Args:
            timeout: Maximum seconds to wait, or None for no limit.

        Returns:
            True if count reached zero, False if timeout expired.
        """
        with self._condition:
            if self._count == 0:
                return True
            return self._condition.wait_for(
                lambda: self._count == 0,
                timeout=timeout,
            )

    @property
    def count(self) -> int:
        """Current count value."""
        with self._lock:
            return self._count


@dataclass
class FakeLatch:
    """Test latch that doesn't block.

    Useful for testing coordination logic without real blocking.

    Example::

        latch = FakeLatch(3)
        assert latch.count == 3

        latch.count_down()
        assert latch.count == 2
        assert latch.await_(timeout=1.0) is False

        latch.count_down()
        latch.count_down()
        assert latch.await_(timeout=1.0) is True
    """

    initial_count: int
    clock: Clock | None = None
    _count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.initial_count < 0:
            msg = "Latch count must be non-negative"
            raise ValueError(msg)
        self._count = self.initial_count

    def count_down(self) -> None:
        """Decrement the count."""
        if self._count > 0:
            self._count -= 1

    def await_(self, timeout: float | None = None) -> bool:
        """Check if count is zero without blocking.

        If count is not zero and a clock is provided, advances the clock.
        """
        if self._count == 0:
            return True

        if timeout is not None and self.clock is not None:
            self.clock.sleep(timeout)

        return self._count == 0

    @property
    def count(self) -> int:
        """Current count value."""
        return self._count

    def reset(self) -> None:
        """Reset count to initial value."""
        self._count = self.initial_count


__all__ = [
    "FakeLatch",
    "Latch",
]
