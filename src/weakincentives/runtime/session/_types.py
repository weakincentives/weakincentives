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

"""Shared typing helpers for session reducers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

from ...prompt._types import SupportsDataclass

if TYPE_CHECKING:
    from .protocols import SessionProtocol


ReducerEvent = SupportsDataclass


S = TypeVar("S", bound=SupportsDataclass)


class ReducerContextProtocol(Protocol):
    """Protocol implemented by reducer context objects."""

    session: SessionProtocol


class TypedReducer(Protocol[S]):
    """Protocol for reducer callables maintained by :class:`Session`."""

    def __call__(
        self,
        slice_values: tuple[S, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[S, ...]: ...


__all__ = [
    "ReducerContextProtocol",
    "ReducerEvent",
    "TypedReducer",
]
