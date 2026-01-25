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

"""Formal specification support for weakincentives using TLA+.

This module provides decorators and metadata classes for embedding TLA+
formal specifications directly in Python code. Specifications are attached
to Python classes via the ``@formal_spec`` decorator and can be extracted,
validated, and model-checked using the companion ``testing`` module.

Overview
--------

TLA+ (Temporal Logic of Actions) is a formal specification language for
describing concurrent and distributed systems. This package enables a
"specification-as-code" approach where TLA+ specifications live alongside
the Python implementation they describe, ensuring they stay synchronized.

The workflow is:

1. Annotate Python classes with ``@formal_spec`` containing the TLA+ model
2. Extract specifications to ``.tla`` and ``.cfg`` files
3. Run TLC model checker to verify invariants hold
4. Use pytest integration for continuous verification

Exports
-------

**Metadata Classes:**

- ``StateVar``: Declares a TLA+ state variable with name, type, and optional
  initial value. Types follow TLA+ conventions (``Nat``, ``Seq``, ``Set``,
  ``Function``, etc.).

- ``ActionParameter``: Defines a parameter for parameterized actions, with
  a name and domain expression (e.g., ``1..NumConsumers``).

- ``Action``: Describes a TLA+ action with preconditions, state updates,
  and optional parameters. Actions define how state transitions occur.

- ``Invariant``: Defines a safety property that must hold in all reachable
  states. Each invariant has an ID, name, and TLA+ predicate expression.

- ``FormalSpec``: The complete specification container holding all metadata.
  Provides ``to_tla()`` and ``to_tla_config()`` methods for generating
  TLA+ module and TLC configuration files.

**Decorator:**

- ``formal_spec``: Class decorator that attaches a ``FormalSpec`` to a Python
  class via the ``__formal_spec__`` attribute.

Basic Usage
-----------

Define a simple counter with a non-negativity invariant::

    from weakincentives.formal import formal_spec, Action, Invariant, StateVar

    @formal_spec(
        module="Counter",
        state_vars=[StateVar("count", "Nat", "Current count value")],
        actions=[
            Action(
                "Increment",
                preconditions=("count < 100",),  # Bounded
                updates={"count": "count + 1"},
            ),
            Action(
                "Reset",
                updates={"count": "0"},
            ),
        ],
        invariants=[
            Invariant("INV-1", "NonNegative", "count >= 0"),
            Invariant("INV-2", "Bounded", "count <= 100"),
        ],
    )
    class Counter:
        def __init__(self) -> None:
            self.count = 0

        def increment(self) -> None:
            if self.count < 100:
                self.count += 1

        def reset(self) -> None:
            self.count = 0

Parameterized Actions
---------------------

For systems with multiple actors or resources, use ``ActionParameter``::

    from weakincentives.formal import (
        formal_spec, Action, ActionParameter, Invariant, StateVar
    )

    @formal_spec(
        module="MessageQueue",
        constants={"NumConsumers": 3, "MaxMessages": 10},
        state_vars=[
            StateVar("messages", "Seq", "Queue of pending messages"),
            StateVar("claimed", "Function", "Messages claimed by consumers"),
        ],
        actions=[
            Action(
                "Send",
                preconditions=("Len(messages) < MaxMessages",),
                updates={"messages": "Append(messages, [id |-> Len(messages)+1])"},
            ),
            Action(
                "Receive",
                parameters=(ActionParameter("consumer", "1..NumConsumers"),),
                preconditions=(
                    "Len(messages) > 0",
                    "claimed[consumer] = NULL",
                ),
                updates={
                    "messages": "Tail(messages)",
                    "claimed": "[claimed EXCEPT ![consumer] = Head(messages)]",
                },
            ),
        ],
        invariants=[
            Invariant("INV-1", "NoDoubleClaim",
                     "\\A c1, c2 \\in 1..NumConsumers : "
                     "c1 /= c2 => claimed[c1] /= claimed[c2]"),
        ],
    )
    class MessageQueue:
        ...

Helper Operators
----------------

Complex specifications often need helper operators for readability::

    @formal_spec(
        module="VisibilityTimeout",
        helpers={
            "VisibleMessages": "{ m \\in messages : m.visible }",
            "InvisibleMessages": "{ m \\in messages : ~m.visible }",
            "ExpiredMessages": "{ m \\in InvisibleMessages : m.timeout <= now }",
        },
        state_vars=[
            StateVar("messages", "Set", "All messages in the system"),
            StateVar("now", "Nat", "Current logical time"),
        ],
        ...
    )
    class VisibilityTimeoutQueue:
        ...

State Constraints
-----------------

Use ``constraint`` to bound the state space for tractable model checking::

    @formal_spec(
        module="TimeBounded",
        constraint="now <= 5",  # Only explore up to time 5
        state_vars=[StateVar("now", "Nat", "Logical clock")],
        ...
    )
    class TimeBoundedSystem:
        ...

Extracting and Verifying
------------------------

Use the ``testing`` module to extract specs and run TLC::

    from pathlib import Path
    from weakincentives.formal.testing import extract_and_verify, extract_spec

    # Extract specification metadata
    spec = extract_spec(Counter)
    print(spec.to_tla())  # Print generated TLA+ module

    # Full extraction and optional model checking
    spec, tla_file, cfg_file, result = extract_and_verify(
        Counter,
        output_dir=Path("specs/tla/extracted"),
        model_check_enabled=True,  # Requires TLC installation
        tlc_config={"workers": "auto"},
    )

    if result:
        print(f"States explored: {result.states_generated}")
        assert result.passed, "Invariant violation detected!"

TLC Installation
----------------

Model checking requires the TLC model checker:

- **macOS**: ``brew install tlaplus``
- **Linux/Other**: Download from https://github.com/tlaplus/tlaplus/releases

Without TLC, specifications can still be extracted and written to files
for manual verification or use with the TLA+ Toolbox IDE.

Pytest Integration
------------------

Create a test that extracts and verifies your specification::

    import pytest
    from pathlib import Path
    from weakincentives.formal.testing import extract_and_verify, ModelCheckError

    def test_counter_formal_spec(tmp_path: Path) -> None:
        spec, tla_file, cfg_file, result = extract_and_verify(
            Counter,
            output_dir=tmp_path,
            model_check_enabled=True,
        )

        assert tla_file.exists()
        assert cfg_file.exists()
        assert result is not None
        assert result.passed

Generated TLA+ Structure
------------------------

The ``FormalSpec.to_tla()`` method generates a complete TLA+ module with:

1. **Header**: Module declaration and EXTENDS clause
2. **Constants**: Declared constants for parameterization
3. **Variables**: State variable declarations with comments
4. **Helpers**: User-defined helper operators
5. **Init**: Initial state formula (auto-generated from types)
6. **Actions**: Individual action definitions with preconditions/updates
7. **Next**: Next-state relation (disjunction of all actions)
8. **Spec**: Complete specification (Init /\\ [][Next]_vars)
9. **Invariants**: Safety properties to verify
10. **Constraint**: Optional state constraint for bounded checking

Best Practices
--------------

1. **Start simple**: Begin with basic state and invariants, add complexity
   incrementally.

2. **Use constraints**: Bound your state space with ``constraint`` to make
   model checking tractable.

3. **Meaningful IDs**: Give invariants clear IDs (e.g., "INV-MessageExclusive")
   for traceability.

4. **Document predicates**: Use the ``description`` field on invariants to
   explain what property is being verified.

5. **Keep specs synchronized**: Run formal verification in CI to catch
   specification drift.

See Also
--------

- ``weakincentives.formal.testing``: Extraction and model checking utilities
- TLA+ documentation: https://lamport.azurewebsites.net/tla/tla.html
- TLC User Guide: https://lamport.azurewebsites.net/tla/tlc.html
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

__all__ = [
    "Action",
    "ActionParameter",
    "FormalSpec",
    "Invariant",
    "StateVar",
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
        initial_value: Optional custom TLA+ expression for initial value
    """

    name: str
    type: str
    description: str = ""
    initial_value: str | None = None


@dataclass(frozen=True, slots=True)
class ActionParameter:
    """TLA+ action parameter with domain.

    Attributes:
        name: Parameter name (e.g., "consumer", "timeout")
        domain: TLA+ domain expression (e.g., "1..NumConsumers", "{\"a\", \"b\", \"c\"}")
    """

    name: str
    domain: str


@dataclass(frozen=True, slots=True)
class Action:
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
        constraint: Optional state constraint to limit TLC exploration (e.g., "now <= 3")
    """

    module: str
    extends: tuple[str, ...] = ("Integers", "Sequences", "FiniteSets")
    constants: dict[str, Any] = field(default_factory=lambda: {})
    state_vars: tuple[StateVar, ...] = field(default_factory=tuple)
    actions: tuple[Action, ...] = field(default_factory=tuple)
    invariants: tuple[Invariant, ...] = field(default_factory=tuple)
    helpers: dict[str, str] = field(default_factory=lambda: {})
    constraint: str | None = None

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
            lines.append(
                "-----------------------------------------------------------------------------"
            )
            lines.append("(* Helper Operators *)")
            lines.append("")
            for name, definition in self.helpers.items():
                lines.append(f"{name} ==")
                for line in definition.split("\n"):
                    lines.append(f"    {line}" if line.strip() else "")
                lines.append("")

        # Init formula (default: all variables at default values)
        if self.state_vars:
            lines.append(
                "-----------------------------------------------------------------------------"
            )
            lines.append("(* Initial State *)")
            lines.append("")
            lines.append("Init ==")
            for i, var in enumerate(self.state_vars):
                prefix = "    /\\" if i > 0 else "   "
                # Use custom initial value if provided, otherwise infer from type
                if var.initial_value is not None:
                    default = var.initial_value
                elif "Seq" in var.type:
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
            lines.append(
                "-----------------------------------------------------------------------------"
            )
            lines.append("(* Actions *)")
            lines.append("")
            for action in self.actions:
                if action.description:
                    lines.append(f"(* {action.description} *)")

                # Action signature
                if action.parameters:
                    param_names = ", ".join(p.name for p in action.parameters)
                    sig = f"{action.name}({param_names})"
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
            lines.append(
                "-----------------------------------------------------------------------------"
            )
            lines.append("(* Next State *)")
            lines.append("")
            lines.append("Next ==")
            for _i, action in enumerate(self.actions):
                # For parameterized actions, existentially quantify over parameters with domains
                if action.parameters:
                    # Build bounded quantifier: \E param1 \in domain1, param2 \in domain2 : Action(param1, param2)
                    param_bindings = ", ".join(
                        f"{p.name} \\in {p.domain}" for p in action.parameters
                    )
                    param_names = ", ".join(p.name for p in action.parameters)
                    action_call = f"\\E {param_bindings} : {action.name}({param_names})"
                else:
                    action_call = action.name
                # All disjuncts start with \/ to avoid precedence issues with \E quantifiers
                prefix = "    \\/"
                lines.append(f"{prefix} {action_call}")
            lines.append("")

        # Spec formula (Init /\ [][Next]_vars)
        if self.state_vars and self.actions:
            lines.append(
                "-----------------------------------------------------------------------------"
            )
            lines.append("(* Specification *)")
            lines.append("")
            lines.append("Spec == Init /\\ [][Next]_vars")
            lines.append("")

        # Invariants
        if self.invariants:
            lines.append(
                "-----------------------------------------------------------------------------"
            )
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

        # State constraint (for limiting TLC exploration)
        if self.constraint:
            lines.append(
                "-----------------------------------------------------------------------------"
            )
            lines.append("(* State Constraint *)")
            lines.append("")
            lines.append(f"StateConstraint == {self.constraint}")
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
        lines: list[str] = []

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

        # State constraint (use parameter if provided, otherwise use instance field)
        if state_constraint:
            # Custom constraint provided as parameter
            lines.append(f"CONSTRAINT {state_constraint}")
            lines.append("")
        elif self.constraint:
            # Use the StateConstraint operator defined in the spec
            lines.append("CONSTRAINT StateConstraint")
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
    constraint: str | None = None,
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
            extends=tuple(extends)
            if extends
            else ("Integers", "Sequences", "FiniteSets"),
            constants=constants or {},
            state_vars=tuple(state_vars) if state_vars else (),
            actions=tuple(actions) if actions else (),
            invariants=tuple(invariants) if invariants else (),
            helpers=helpers or {},
            constraint=constraint,
        )
        setattr(cls, "__formal_spec__", spec)
        return cls

    return decorator
