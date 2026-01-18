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

"""Thread-safe callback registry."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class CallbackRegistry(Generic[T]):
    """Thread-safe registry for callbacks.

    Callbacks are registered and unregistered under a lock, but invoked
    outside the lock to prevent deadlocks. This pattern is used by
    Heartbeat for beat callbacks.

    Example::

        registry = CallbackRegistry[str]()

        def on_message(msg: str) -> None:
            print(f"Received: {msg}")

        registry.register(on_message)

        # Invoke all callbacks (exceptions don't stop other callbacks)
        errors = registry.invoke_all("hello")
        if errors:
            for error in errors:
                logger.error("Callback failed", exc_info=error)

    Thread Safety:
        - register() and unregister() are atomic
        - Callbacks are copied before invocation (snapshot)
        - Callbacks execute without holding the lock
        - One callback's exception doesn't affect others
    """

    _callbacks: list[Callable[[T], None]] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def register(self, callback: Callable[[T], None]) -> None:
        """Register a callback.

        Args:
            callback: Function to call with the value on invoke.
        """
        with self._lock:
            self._callbacks.append(callback)

    def unregister(self, callback: Callable[[T], None]) -> None:
        """Unregister a callback.

        Does nothing if the callback is not registered.

        Args:
            callback: The callback to remove.
        """
        with self._lock:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

    def invoke(self, value: T) -> int:
        """Invoke all callbacks, stopping on first exception.

        Args:
            value: The value to pass to each callback.

        Returns:
            Number of callbacks successfully invoked.

        Raises:
            Exception: The first exception raised by a callback.
        """
        # Snapshot callbacks under lock
        with self._lock:
            callbacks = list(self._callbacks)

        # Invoke outside lock
        for i, callback in enumerate(callbacks):
            callback(value)
        return len(callbacks)

    def invoke_all(self, value: T) -> list[Exception]:
        """Invoke all callbacks, collecting exceptions.

        Continues invoking remaining callbacks even if one raises.

        Args:
            value: The value to pass to each callback.

        Returns:
            List of exceptions raised by callbacks (empty if all succeeded).
        """
        # Snapshot callbacks under lock
        with self._lock:
            callbacks = list(self._callbacks)

        # Invoke outside lock, collecting errors
        errors: list[Exception] = []
        for callback in callbacks:
            try:
                callback(value)
            except Exception as e:
                errors.append(e)
        return errors

    def clear(self) -> None:
        """Remove all registered callbacks."""
        with self._lock:
            self._callbacks.clear()

    @property
    def count(self) -> int:
        """Number of registered callbacks."""
        with self._lock:
            return len(self._callbacks)


__all__ = [
    "CallbackRegistry",
]
