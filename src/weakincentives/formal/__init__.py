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

"""Formal specification support for weakincentives.

This module provides decorators and metadata classes for embedding TLA+
formal specifications directly in Python code. Specs are extracted and
validated by a pytest plugin.

Example:
    >>> from weakincentives.formal import formal_spec, Action, Invariant, StateVar
    >>>
    >>> @formal_spec(
    ...     module="Counter",
    ...     state_vars=[StateVar("count", "Nat", "Current count")],
    ...     actions=[Action("Increment", updates={"count": "count + 1"})],
    ...     invariants=[Invariant("INV-1", "NonNegative", "count >= 0")],
    ... )
    ... class Counter:
    ...     def __init__(self):
    ...         self.count = 0
    ...
    ...     def increment(self):
    ...         self.count += 1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

__all__ = [
    "StateVar",
    "Action",
    "Invariant",
    "FormalSpec",
    "formal_spec",
]

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class StateVar:
    """TLA+ state variable metadata.

    Attributes:
        name: Variable name (e.g., "pending", "invisible")
        type: TLA+ type annotation (e.g., "Seq", "Function", "Set")
        description: Human-readable description
    """

    name: str
    type: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class Action:
    """TLA+ action metadata.

    Attributes:
        name: Action name (e.g., "Receive", "Send")
        parameters: Action parameters (e.g., ["consumer", "timeout"])
        preconditions: List of TLA+ precondition expressions
        updates: Mapping from state variable to TLA+ update expression
        description: Human-readable description
    """

    name: str
    parameters: tuple[str, ...] = field(default_factory=tuple)
    preconditions: tuple[str, ...] = field(default_factory=tuple)
    updates: dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True, slots=True)
class Invariant:
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


@dataclass(frozen=True, slots=True)
class FormalSpec:
    """Complete formal specification metadata.

    This represents a TLA+ module that can be extracted and validated.

    Attributes:
        module: TLA+ module name
        extends: Modules to extend (e.g., ["Integers", "Sequences"])
        constants: Constant definitions with default values
        state_vars: State variable declarations
        actions: Action definitions
        invariants: Invariant definitions
        helpers: Helper operator definitions (raw TLA+)
    """

    module: str
    extends: tuple[str, ...] = ("Integers", "Sequences", "FiniteSets")
    constants: dict[str, Any] = field(default_factory=dict)
    state_vars: tuple[StateVar, ...] = field(default_factory=tuple)
    actions: tuple[Action, ...] = field(default_factory=tuple)
    invariants: tuple[Invariant, ...] = field(default_factory=tuple)
    helpers: dict[str, str] = field(default_factory=dict)

    def to_tla(self) -> str:
        """Generate TLA+ module from metadata.

        Returns:
            Complete TLA+ module as a string
        """
        # Generate header line with consistent width (77 chars total)
        # Format: "---- MODULE ModuleName ----" with padding to 77 chars
        module_line = f"MODULE {self.module}"
        dashes_needed = 77 - len(module_line) - 2  # -2 for spaces
        left_dashes = dashes_needed // 2
        right_dashes = dashes_needed - left_dashes
        header = f"{'-' * left_dashes} {module_line} {'-' * right_dashes}"

        lines = [
            header,
            "(* Generated from Python formal specification metadata *)",
            "",
        ]

        # EXTENDS clause
        if self.extends:
            lines.append(f"EXTENDS {', '.join(self.extends)}")
            lines.append("")

        # Constants
        if self.constants:
            const_names = list(self.constants.keys())
            lines.append("CONSTANTS")
            for i, name in enumerate(const_names):
                comma = "," if i < len(const_names) - 1 else ""
                lines.append(f"    {name}{comma}")
            lines.append("")

        # State variables
        if self.state_vars:
            lines.append("VARIABLES")
            for i, var in enumerate(self.state_vars):
                comma = "," if i < len(self.state_vars) - 1 else ""
                comment = f"  \\* {var.description}" if var.description else ""
                lines.append(f"    {var.name}{comma}{comment}")
            lines.append("")

            # vars tuple
            var_names = [var.name for var in self.state_vars]
            lines.append(f"vars == <<{', '.join(var_names)}>>")
            lines.append("")

        # Helper operators
        if self.helpers:
            lines.append("-----------------------------------------------------------------------------")
            lines.append("(* Helper Operators *)")
            lines.append("")
            for name, definition in self.helpers.items():
                lines.append(f"{name} ==")
                for line in definition.split("\n"):
                    lines.append(f"    {line}" if line.strip() else "")
                lines.append("")

        # Init formula (default: all variables at default values)
        if self.state_vars:
            lines.append("-----------------------------------------------------------------------------")
            lines.append("(* Initial State *)")
            lines.append("")
            lines.append("Init ==")
            for i, var in enumerate(self.state_vars):
                prefix = "    /\\" if i > 0 else "   "
                # Provide sensible defaults based on type
                if "Seq" in var.type:
                    default = "<<>>"
                elif "Set" in var.type:
                    default = "{}"
                elif "Function" in var.type:
                    default = "[x \\in {} |-> 0]"
                elif "Nat" in var.type or "Int" in var.type:
                    default = "0"
                else:
                    default = "NULL"  # fallback
                lines.append(f"{prefix} {var.name} = {default}")
            lines.append("")

        # Actions
        if self.actions:
            lines.append("-----------------------------------------------------------------------------")
            lines.append("(* Actions *)")
            lines.append("")
            for action in self.actions:
                if action.description:
                    lines.append(f"(* {action.description} *)")

                # Action signature
                if action.parameters:
                    sig = f"{action.name}({', '.join(action.parameters)})"
                else:
                    sig = action.name
                lines.append(f"{sig} ==")

                # Preconditions
                if action.preconditions:
                    for precond in action.preconditions:
                        lines.append(f"    /\\ {precond}")

                # Updates
                if action.updates:
                    for var, expr in action.updates.items():
                        lines.append(f"    /\\ {var}' = {expr}")

                # UNCHANGED
                unchanged_vars = [
                    var.name
                    for var in self.state_vars
                    if var.name not in action.updates
                ]
                if unchanged_vars:
                    lines.append(f"    /\\ UNCHANGED <<{', '.join(unchanged_vars)}>>")

                lines.append("")

        # Next formula (disjunction of all actions)
        if self.actions:
            lines.append("-----------------------------------------------------------------------------")
            lines.append("(* Next State *)")
            lines.append("")
            lines.append("Next ==")
            for i, action in enumerate(self.actions):
                # For parameterized actions, existentially quantify over parameters
                if action.parameters:
                    # Assume parameters are typed (e.g., "consumer \\in 1..NumConsumers")
                    action_call = f"\\E {', '.join(action.parameters)} : {action.name}({', '.join(action.parameters)})"
                else:
                    action_call = action.name
                prefix = "    \\/" if i > 0 else "   "
                lines.append(f"{prefix} {action_call}")
            lines.append("")

        # Spec formula (Init /\ [][Next]_vars)
        if self.state_vars and self.actions:
            lines.append("-----------------------------------------------------------------------------")
            lines.append("(* Specification *)")
            lines.append("")
            lines.append("Spec == Init /\\ [][Next]_vars")
            lines.append("")

        # Invariants
        if self.invariants:
            lines.append("-----------------------------------------------------------------------------")
            lines.append("(* Invariants *)")
            lines.append("")
            for inv in self.invariants:
                if inv.description:
                    lines.append(f"(* {inv.id}: {inv.description} *)")
                else:
                    lines.append(f"(* {inv.id} *)")
                lines.append(f"{inv.name} ==")

                # Handle multi-line predicates
                pred_lines = inv.predicate.split("\n")
                if len(pred_lines) == 1:
                    lines.append(f"    {inv.predicate}")
                else:
                    for pline in pred_lines:
                        lines.append(f"    {pline}" if pline.strip() else "")

                lines.append("")

        # Footer line must match header width
        lines.append("=" * len(header))
        return "\n".join(lines)

    def to_tla_config(
        self,
        *,
        init: str | None = None,
        next: str | None = None,
        check_deadlock: bool = False,
        state_constraint: str | None = None,
    ) -> str:
        """Generate TLC model checker configuration.

        Args:
            init: Initial state formula (optional, for simulation mode)
            next: Next-state formula (optional, for simulation mode)
            check_deadlock: Whether to check for deadlocks
            state_constraint: Optional state constraint expression

        Returns:
            TLC configuration file content
        """
        lines = []

        # Use SPECIFICATION Spec if available
        if init is None and next is None:
            lines.append("SPECIFICATION Spec")
            lines.append("")
        elif init and next:
            lines.append(f"INIT {init}")
            lines.append(f"NEXT {next}")
            lines.append("")

        # Constants
        if self.constants:
            lines.append("CONSTANTS")
            for name, value in self.constants.items():
                lines.append(f"    {name} = {value}")
            lines.append("")

        # Invariants
        if self.invariants:
            lines.append("INVARIANTS")
            for inv in self.invariants:
                lines.append(f"    {inv.name}")
            lines.append("")

        # State constraint
        if state_constraint:
            lines.append(f"CONSTRAINT {state_constraint}")
            lines.append("")

        # Deadlock checking
        if not check_deadlock:
            lines.append("CHECK_DEADLOCK FALSE")
            lines.append("")

        return "\n".join(lines)


def formal_spec(
    module: str,
    *,
    extends: tuple[str, ...] | None = None,
    constants: dict[str, Any] | None = None,
    state_vars: list[StateVar] | None = None,
    actions: list[Action] | None = None,
    invariants: list[Invariant] | None = None,
    helpers: dict[str, str] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Attach formal specification metadata to a class.

    This decorator associates TLA+ formal specification metadata with a Python
    class. The metadata can be extracted by pytest plugins to generate and
    validate TLA+ modules.

    Args:
        module: TLA+ module name
        extends: Modules to extend (default: Integers, Sequences, FiniteSets)
        constants: Constant definitions with default values
        state_vars: State variable declarations
        actions: Action definitions
        invariants: Invariant definitions
        helpers: Helper operator definitions (raw TLA+)

    Returns:
        Decorator that attaches __formal_spec__ attribute

    Example:
        >>> @formal_spec(
        ...     module="Counter",
        ...     state_vars=[StateVar("count", "Nat", "Current count")],
        ...     invariants=[Invariant("INV-1", "NonNegative", "count >= 0")],
        ... )
        ... class Counter:
        ...     pass
    """

    def decorator(cls: type[T]) -> type[T]:
        spec = FormalSpec(
            module=module,
            extends=tuple(extends) if extends else ("Integers", "Sequences", "FiniteSets"),
            constants=constants or {},
            state_vars=tuple(state_vars) if state_vars else (),
            actions=tuple(actions) if actions else (),
            invariants=tuple(invariants) if invariants else (),
            helpers=helpers or {},
        )
        setattr(cls, "__formal_spec__", spec)
        return cls

    return decorator
