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

# pyright: reportImportCycles=false

"""Shared typing helpers for session reducers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

from ..prompt._types import SupportsDataclass

if TYPE_CHECKING:
    from .reducer_context import ReducerContext


class ReducerEvent(Protocol):
    """Structural type satisfied by session data events."""

    @property
    def value(self) -> SupportsDataclass | None: ...


S = TypeVar("S")


class TypedReducer(Protocol[S]):
    """Callable signature expected for session reducers."""

    def __call__(
        self,
        slice_values: tuple[S, ...],
        event: ReducerEvent,
        *,
        context: ReducerContext,
    ) -> tuple[S, ...]: ...


__all__ = ["ReducerEvent", "TypedReducer"]
