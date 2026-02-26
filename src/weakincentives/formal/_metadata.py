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

"""Metadata dataclasses for TLA+ formal specifications.

Pure data types with no dependencies beyond the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..dataclasses import FrozenDataclassMixin


@dataclass(slots=True, frozen=True)
class StateVar(FrozenDataclassMixin):
    """TLA+ state variable metadata.

    Attributes:
        name: Variable name (e.g., "pending", "invisible")
        type: TLA+ type annotation (e.g., "Seq", "Function", "Set")
        description: Human-readable description
        initial_value: Optional custom TLA+ expression for initial value
    """

    name: str
    type: str
    description: str = ""
    initial_value: str | None = None


@dataclass(slots=True, frozen=True)
class ActionParameter(FrozenDataclassMixin):
    """TLA+ action parameter with domain.

    Attributes:
        name: Parameter name (e.g., "consumer", "timeout")
        domain: TLA+ domain expression (e.g., "1..NumConsumers", "{\"a\", \"b\", \"c\"}")
    """

    name: str
    domain: str


@dataclass(slots=True, frozen=True)
class Action(FrozenDataclassMixin):
    """TLA+ action metadata.

    Attributes:
        name: Action name (e.g., "Receive", "Send")
        parameters: Action parameters with domains (e.g., [ActionParameter("consumer", "1..NumConsumers")])
        preconditions: List of TLA+ precondition expressions
        updates: Mapping from state variable to TLA+ update expression
        description: Human-readable description
    """

    name: str
    parameters: tuple[ActionParameter, ...] = field(default_factory=tuple)
    preconditions: tuple[str, ...] = field(default_factory=tuple)
    updates: dict[str, str] = field(default_factory=lambda: {})
    description: str = ""


@dataclass(slots=True, frozen=True)
class Invariant(FrozenDataclassMixin):
    """TLA+ invariant metadata.

    Attributes:
        id: Unique identifier (e.g., "INV-1", "INV-MessageStateExclusive")
        name: Invariant name (e.g., "MessageStateExclusive")
        predicate: TLA+ predicate expression
        description: Human-readable description
    """

    id: str
    name: str
    predicate: str
    description: str = ""
