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

"""Error types for design-by-contract enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, override

from ..errors import WinkError


class StateError(WinkError, RuntimeError):
    """Base class for state machine enforcement errors."""


@dataclass(frozen=True, slots=True)
class InvalidStateError(StateError):
    """Method called in invalid state.

    Raised when a method decorated with @transition or @in_state is called
    while the object is not in a valid source state.

    Attributes:
        cls: The class containing the method.
        method: Name of the method that was called.
        current_state: The actual state when the method was called.
        valid_states: The states that would have been valid.
    """

    cls: type[Any]
    method: str
    current_state: Enum
    valid_states: tuple[Enum, ...]

    @override
    def __str__(self) -> str:
        """Format error message with state details."""
        valid = ", ".join(s.name for s in self.valid_states)
        return (
            f"{self.cls.__name__}.{self.method}() requires state in "
            f"[{valid}], but current state is {self.current_state.name}"
        )


__all__ = [
    "InvalidStateError",
    "StateError",
]
