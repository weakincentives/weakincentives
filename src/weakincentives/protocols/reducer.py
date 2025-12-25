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

"""Reducer protocol definitions.

This module defines the reducer contract used throughout weakincentives.
It has minimal dependencies to prevent circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..types.dataclass import SupportsDataclass

if TYPE_CHECKING:
    from .session import SessionViewProtocol

ReducerEvent = SupportsDataclass
"""Type alias for reducer event parameters."""


class ReducerContextProtocol(Protocol):
    """Protocol implemented by reducer context objects."""

    session: SessionViewProtocol


class TypedReducer[S: SupportsDataclass](Protocol):
    """Protocol for reducer callables maintained by Session.

    Reducers receive a lazy SliceView and return a SliceOp describing
    the mutation to apply. The view and return types use object since
    the actual SliceView and SliceOp types are defined in runtime.session.slices.
    """

    def __call__(
        self,
        view: object,  # SliceView[S] at runtime
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> object:  # SliceOp[S] at runtime
        ...


__all__ = [
    "ReducerContextProtocol",
    "ReducerEvent",
    "TypedReducer",
]
