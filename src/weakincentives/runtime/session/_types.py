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

from typing import TYPE_CHECKING, Protocol

from ...types.dataclass import SupportsDataclass
from .slices import SliceOp, SliceView

if TYPE_CHECKING:
    from .protocols import SessionViewProtocol


type ReducerEvent = SupportsDataclass


class ReducerContextProtocol(Protocol):
    """Protocol implemented by reducer context objects."""

    session: SessionViewProtocol


class TypedReducer[S: SupportsDataclass](Protocol):
    """Protocol for reducer callables maintained by :class:`Session`.

    Reducers receive a lazy SliceView and return a SliceOp describing
    the mutation to apply.
    """

    def __call__(
        self,
        view: SliceView[S],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> SliceOp[S]: ...


__all__ = [
    "ReducerContextProtocol",
    "ReducerEvent",
    "TypedReducer",
]
