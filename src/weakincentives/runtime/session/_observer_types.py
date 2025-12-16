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

"""Observer types for session state change notifications."""

from __future__ import annotations

from collections.abc import Callable
from typing import override
from uuid import UUID, uuid4

from ...types.dataclass import SupportsDataclass

type SliceObserver[T: SupportsDataclass] = Callable[
    [tuple[T, ...], tuple[T, ...]],
    None,
]
"""Observer callback invoked when a slice changes.

Receives ``(old_values, new_values)`` after each state update.
"""


class Subscription:
    """Handle for unsubscribing an observer."""

    __slots__ = ("_unsubscribe_fn", "subscription_id")

    def __init__(
        self,
        unsubscribe_fn: Callable[[], None] | None = None,
        subscription_id: UUID | None = None,
    ) -> None:
        super().__init__()
        self.subscription_id: UUID = subscription_id if subscription_id else uuid4()
        self._unsubscribe_fn: Callable[[], None] | None = unsubscribe_fn

    def unsubscribe(self) -> bool:
        """Remove the observer. Returns True if successfully unsubscribed."""
        if self._unsubscribe_fn is not None:
            self._unsubscribe_fn()
            self._unsubscribe_fn = None
            return True
        return False

    @override
    def __repr__(self) -> str:
        return f"Subscription(subscription_id={self.subscription_id!r})"


__all__ = [
    "SliceObserver",
    "Subscription",
]
