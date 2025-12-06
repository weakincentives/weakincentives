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

"""Type definitions for mutation builders.

This module defines the :class:`MutationProvider` protocol separately to avoid
import cycles with protocols.py and _types.py.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Protocol

from ...prompt._types import SupportsDataclass

if TYPE_CHECKING:
    from .snapshots import Snapshot


class MutationProvider(Protocol):
    """Protocol for objects that can provide mutation operations.

    This protocol is implemented by :class:`Session` and defines the
    internal interface used by :class:`MutationBuilder`.
    """

    def mutation_seed_slice[S: SupportsDataclass](
        self, slice_type: type[S], values: Iterable[S]
    ) -> None: ...

    def mutation_clear_slice[S: SupportsDataclass](
        self,
        slice_type: type[S],
        predicate: Callable[[S], bool] | None = None,
    ) -> None: ...

    def mutation_reset(self) -> None: ...

    def mutation_rollback(self, snapshot: Snapshot) -> None: ...

    def mutation_register_reducer[S: SupportsDataclass](
        self,
        data_type: type[SupportsDataclass],
        reducer: Any,  # TypedReducer[S] - avoiding import cycle  # noqa: ANN401
        *,
        slice_type: type[S] | None = None,
    ) -> None: ...

    def mutation_dispatch_event(
        self, slice_type: type[SupportsDataclass], event: SupportsDataclass
    ) -> None: ...


__all__ = ["MutationProvider"]
