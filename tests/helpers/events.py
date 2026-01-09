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

"""Test utilities for working with dispatchers."""

from __future__ import annotations

from weakincentives.runtime.events import DispatchResult
from weakincentives.runtime.events.types import EventHandler


class NullDispatcher:
    """Dispatcher implementation that discards all events."""

    @staticmethod
    def subscribe(event_type: type[object], handler: EventHandler) -> None:
        """No-op subscription hook."""
        del event_type, handler

    @staticmethod
    def unsubscribe(event_type: type[object], handler: EventHandler) -> bool:
        """No-op unsubscription hook.

        Always returns ``False`` since no handlers are stored.
        """
        del event_type, handler
        return False

    @staticmethod
    def dispatch(event: object) -> DispatchResult:
        """Drop the provided event instance."""

        return DispatchResult(
            event=event,
            handlers_invoked=(),
            errors=(),
        )


__all__ = ["NullDispatcher"]
