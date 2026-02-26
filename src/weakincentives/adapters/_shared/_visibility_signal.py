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

"""Thread-safe signal for VisibilityExpansionRequired propagation.

This module provides a mechanism to propagate VisibilityExpansionRequired
exceptions from MCP tool handlers back to the adapter. Since the MCP bridge
runs in a different execution context than the adapter, we cannot rely on
normal exception propagation. Instead, we use a shared signal that stores
the exception for the adapter to check after SDK query completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from ...prompt import VisibilityExpansionRequired


@dataclass(slots=True)
class VisibilityExpansionSignal:
    """Thread-safe container for VisibilityExpansionRequired signal.

    When a tool handler raises VisibilityExpansionRequired, the MCP bridge
    stores the exception in this signal and returns an error response to the
    SDK. After the SDK query completes, the adapter checks this signal and
    re-raises the stored exception if present.

    This enables progressive disclosure to work with the Claude Agent SDK
    adapter despite the architectural differences in how exceptions propagate.

    Thread Safety:
        All operations are protected by a lock. Multiple tool handlers could
        theoretically raise VisibilityExpansionRequired, but only the first
        exception is stored (first-one-wins semantics).

    Example:
        signal = VisibilityExpansionSignal()

        # In MCP tool handler:
        try:
            result = handler(params, context=context)
        except VisibilityExpansionRequired as exc:
            signal.set(exc)
            return error_response

        # In adapter after SDK query:
        stored_exc = signal.get_and_clear()
        if stored_exc is not None:
            raise stored_exc
    """

    _exception: VisibilityExpansionRequired | None = field(default=None)
    _lock: Lock = field(default_factory=Lock)

    def set(self, exc: VisibilityExpansionRequired) -> None:
        """Store a VisibilityExpansionRequired exception.

        Only the first exception is stored; subsequent calls are ignored.
        This implements first-one-wins semantics.

        Args:
            exc: The VisibilityExpansionRequired exception to store.
        """
        with self._lock:
            if self._exception is None:
                self._exception = exc

    def get_and_clear(self) -> VisibilityExpansionRequired | None:
        """Get and clear the stored exception.

        Returns:
            The stored VisibilityExpansionRequired exception, or None if
            no exception was stored. The signal is cleared after this call.
        """
        with self._lock:
            exc = self._exception
            self._exception = None
            return exc

    def is_set(self) -> bool:
        """Check if an exception is stored without clearing it.

        Returns:
            True if an exception is stored, False otherwise.
        """
        with self._lock:
            return self._exception is not None
