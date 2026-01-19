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

"""Cancellation token and checkpoint implementations."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from weakincentives.threading._types import CancelledException

if TYPE_CHECKING:
    from weakincentives.threading._types import (
        CancellationToken as CancellationTokenProtocol,
    )


@dataclass
class SimpleCancellationToken:
    """Thread-safe cancellation token.

    Example::

        token = SimpleCancellationToken()

        def worker():
            while not token.is_cancelled():
                do_work()
                token.check()  # Raises CancelledException if cancelled

        # In control thread
        token.cancel()  # Request cancellation
    """

    _cancelled: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _parent: SimpleCancellationToken | None = field(default=None, repr=False)

    def cancel(self) -> None:
        """Request cancellation."""
        with self._lock:
            self._cancelled = True

    def is_cancelled(self) -> bool:
        """Return True if cancellation was requested.

        Also returns True if a parent token was cancelled.
        """
        with self._lock:
            if self._cancelled:
                return True
        if self._parent is not None:
            return self._parent.is_cancelled()
        return False

    def check(self) -> None:
        """Raise CancelledException if cancelled."""
        if self.is_cancelled():
            raise CancelledException("Task was cancelled")

    def child(self) -> SimpleCancellationToken:
        """Create a child token that cancels when parent cancels."""
        return SimpleCancellationToken(_parent=self)


@dataclass
class SystemCheckpoint:
    """Production checkpoint for cooperative yielding.

    Uses time.sleep(0) to yield control, which releases the GIL
    and allows other Python threads to run.

    Example::

        checkpoint = SystemCheckpoint()

        for item in large_dataset:
            checkpoint.check()  # Check cancellation
            process(item)
            checkpoint.yield_()  # Yield control
    """

    _token: SimpleCancellationToken = field(
        default_factory=SimpleCancellationToken,
        repr=False,
    )

    def __init__(self, token: CancellationTokenProtocol | None = None) -> None:
        """Initialize checkpoint with optional cancellation token."""
        object.__init__(self)
        if token is None:
            self._token = SimpleCancellationToken()
        elif isinstance(token, SimpleCancellationToken):
            self._token = token
        else:
            # Wrap external token
            self._token = _WrappedToken(token)  # type: ignore[assignment]

    def yield_(self) -> None:
        """Yield control by releasing the GIL briefly."""
        time.sleep(0)

    def check(self) -> None:
        """Check cancellation and raise if cancelled."""
        self._token.check()

    def is_cancelled(self) -> bool:
        """Return True if cancellation was requested."""
        return self._token.is_cancelled()

    @property
    def token(self) -> SimpleCancellationToken:
        """The cancellation token for this checkpoint."""
        return self._token


@dataclass
class FakeCheckpoint:
    """Test checkpoint with yield/check counting.

    Does not block on yield. Tracks yield and check counts for assertions.

    Example::

        checkpoint = FakeCheckpoint()

        def task():
            checkpoint.yield_()
            checkpoint.check()
            checkpoint.yield_()

        task()

        assert checkpoint.yield_count == 2
        assert checkpoint.check_count == 1

        # Test cancellation
        checkpoint.token.cancel()
        with pytest.raises(CancelledException):
            checkpoint.check()
    """

    _token: SimpleCancellationToken = field(
        default_factory=SimpleCancellationToken,
        repr=False,
    )
    _yield_count: int = field(default=0, repr=False)
    _check_count: int = field(default=0, repr=False)

    def yield_(self) -> None:
        """Record yield without blocking."""
        self._yield_count += 1

    def check(self) -> None:
        """Check cancellation and record the check."""
        self._check_count += 1
        self._token.check()

    def is_cancelled(self) -> bool:
        """Return True if cancellation was requested."""
        return self._token.is_cancelled()

    @property
    def token(self) -> SimpleCancellationToken:
        """The cancellation token for this checkpoint."""
        return self._token

    @property
    def yield_count(self) -> int:
        """Number of times yield_() was called."""
        return self._yield_count

    @property
    def check_count(self) -> int:
        """Number of times check() was called."""
        return self._check_count

    def reset(self) -> None:
        """Reset counters and cancellation state."""
        self._yield_count = 0
        self._check_count = 0
        self._token = SimpleCancellationToken()


class _WrappedToken:
    """Wraps an external CancellationToken protocol implementation."""

    def __init__(self, token: CancellationTokenProtocol) -> None:
        super().__init__()
        self._token = token

    def cancel(self) -> None:
        self._token.cancel()

    def is_cancelled(self) -> bool:
        return self._token.is_cancelled()

    def check(self) -> None:
        self._token.check()

    def child(self) -> SimpleCancellationToken:
        return SimpleCancellationToken(_parent=self)  # type: ignore[arg-type]


__all__ = [
    "FakeCheckpoint",
    "SimpleCancellationToken",
    "SystemCheckpoint",
]
