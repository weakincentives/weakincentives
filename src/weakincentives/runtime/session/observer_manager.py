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

"""Observer management for session state change notifications."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, cast
from uuid import uuid4

from ...types.dataclass import SupportsDataclass
from ..logging import StructuredLogger, get_logger
from ._observer_types import SliceObserver, Subscription
from ._slice_types import SessionSlice, SessionSliceType

logger: StructuredLogger = get_logger(
    __name__, context={"component": "observer_manager"}
)


@dataclass(slots=True)
class ObserverRegistration:
    """Registration entry linking an observer to its subscription handle."""

    observer: SliceObserver[Any]
    subscription: Subscription


class ObserverManager:
    """Thread-safe manager for slice observers.

    Manages:
    - Observer registrations (dict mapping slice types to observer lists)
    - Observer notification after state changes
    """

    __slots__ = ("_lock", "_observers")

    def __init__(self, lock: RLock) -> None:
        """Initialize the observer manager.

        Args:
            lock: Shared RLock for thread-safe access.
        """
        super().__init__()
        self._lock = lock
        self._observers: dict[SessionSliceType, list[ObserverRegistration]] = {}

    def observe[S: SupportsDataclass](
        self,
        slice_type: type[S],
        observer: SliceObserver[S],
    ) -> Subscription:
        """Register an observer called when the slice changes.

        The observer receives ``(old_values, new_values)`` after each state
        update. Returns a :class:`Subscription` handle that can be used to
        unsubscribe.

        Thread-safe: Acquires lock during registration and unsubscribe.
        """
        subscription_id = uuid4()

        def unsubscribe() -> None:
            with self._lock:
                registrations = self._observers.get(slice_type, [])
                self._observers[slice_type] = [
                    reg
                    for reg in registrations
                    if reg.subscription.subscription_id != subscription_id
                ]

        subscription = Subscription(
            unsubscribe_fn=unsubscribe, subscription_id=subscription_id
        )

        registration = ObserverRegistration(
            observer=cast(SliceObserver[Any], observer),
            subscription=subscription,
        )
        with self._lock:
            bucket = self._observers.setdefault(slice_type, [])
            bucket.append(registration)

        return subscription

    def notify_observers(
        self, state_changes: dict[SessionSliceType, tuple[SessionSlice, SessionSlice]]
    ) -> None:
        """Call registered observers for slices that changed.

        Exceptions in observers are logged and isolated (non-blocking).

        Thread-safe: Acquires lock when fetching observer list.
        """
        for slice_type, (old_values, new_values) in state_changes.items():
            with self._lock:
                observer_registrations = list(self._observers.get(slice_type, ()))

            for registration in observer_registrations:
                try:
                    registration.observer(old_values, new_values)
                except Exception:
                    observer_name = getattr(
                        registration.observer,
                        "__qualname__",
                        repr(registration.observer),
                    )
                    logger.exception(
                        "Observer invocation failed.",
                        event="session_observer_failed",
                        context={
                            "observer": observer_name,
                            "slice_type": slice_type.__qualname__,
                        },
                    )


__all__ = ["ObserverManager", "ObserverRegistration"]
