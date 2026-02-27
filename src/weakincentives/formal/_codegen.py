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

"""TLA+ code generation from formal specification metadata.

Depends only on the metadata dataclasses from ``_metadata``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from weakincentives.dataclasses import FrozenDataclassMixin
from weakincentives.formal._metadata import Action, Invariant, StateVar

_TLA_SEPARATOR = (
    "-----------------------------------------------------------------------------"
)


def _tla_section_header(lines: list[str], title: str) -> None:
    """Append a TLA+ section separator and title comment."""
    lines.append(_TLA_SEPARATOR)
    lines.append(f"(* {title} *)")
    lines.append("")


def _infer_default(var: StateVar) -> str:
    """Infer a default initial value for a TLA+ state variable."""
    if var.initial_value is not None:
        return var.initial_value
    if "Seq" in var.type:
        return "<<>>"
    if "Set" in var.type:
        return "{}"
    if "Function" in var.type:
        return "[x \\in {} |-> 0]"
    if "Nat" in var.type or "Int" in var.type:
        return "0"
    return "NULL"


@dataclass(slots=True, frozen=True)
class FormalSpec(FrozenDataclassMixin):
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
        header = self._tla_header()
        lines = [
            header,
            "(* Generated from Python formal specification metadata *)",
            "",
        ]

        self._emit_extends(lines)
        self._emit_constants(lines)
        self._emit_state_vars(lines)
        self._emit_helpers(lines)
        self._emit_init(lines)
        self._emit_actions(lines)
        self._emit_next(lines)
        self._emit_spec(lines)
        self._emit_invariants(lines)
        self._emit_constraint(lines)

        lines.append("=" * len(header))
        return "\n".join(lines)

    def _tla_header(self) -> str:
        module_line = f"MODULE {self.module}"
        dashes_needed = 77 - len(module_line) - 2
        left_dashes = dashes_needed // 2
        right_dashes = dashes_needed - left_dashes
        return f"{'-' * left_dashes} {module_line} {'-' * right_dashes}"

    def _emit_extends(self, lines: list[str]) -> None:
        if self.extends:
            lines.append(f"EXTENDS {', '.join(self.extends)}")
            lines.append("")

    def _emit_constants(self, lines: list[str]) -> None:
        if not self.constants:
            return
        const_names = list(self.constants.keys())
        lines.append("CONSTANTS")
        for i, name in enumerate(const_names):
            comma = "," if i < len(const_names) - 1 else ""
            lines.append(f"    {name}{comma}")
        lines.append("")

    def _emit_state_vars(self, lines: list[str]) -> None:
        if not self.state_vars:
            return
        lines.append("VARIABLES")
        for i, var in enumerate(self.state_vars):
            comma = "," if i < len(self.state_vars) - 1 else ""
            comment = f"  \\* {var.description}" if var.description else ""
            lines.append(f"    {var.name}{comma}{comment}")
        lines.append("")
        var_names = [var.name for var in self.state_vars]
        lines.append(f"vars == <<{', '.join(var_names)}>>")
        lines.append("")

    def _emit_helpers(self, lines: list[str]) -> None:
        if not self.helpers:
            return
        _tla_section_header(lines, "Helper Operators")
        for name, definition in self.helpers.items():
            lines.append(f"{name} ==")
            for line in definition.split("\n"):
                lines.append(f"    {line}" if line.strip() else "")
            lines.append("")

    def _emit_init(self, lines: list[str]) -> None:
        if not self.state_vars:
            return
        _tla_section_header(lines, "Initial State")
        lines.append("Init ==")
        for i, var in enumerate(self.state_vars):
            prefix = "    /\\" if i > 0 else "   "
            default = _infer_default(var)
            lines.append(f"{prefix} {var.name} = {default}")
        lines.append("")

    def _emit_actions(self, lines: list[str]) -> None:
        if not self.actions:
            return
        _tla_section_header(lines, "Actions")
        for action in self.actions:
            self._emit_single_action(lines, action)

    def _emit_single_action(self, lines: list[str], action: Action) -> None:
        if action.description:
            lines.append(f"(* {action.description} *)")
        if action.parameters:
            param_names = ", ".join(p.name for p in action.parameters)
            sig = f"{action.name}({param_names})"
        else:
            sig = action.name
        lines.append(f"{sig} ==")
        for precond in action.preconditions:
            lines.append(f"    /\\ {precond}")
        for var, expr in action.updates.items():
            lines.append(f"    /\\ {var}' = {expr}")
        unchanged_vars = [
            v.name for v in self.state_vars if v.name not in action.updates
        ]
        if unchanged_vars:
            lines.append(f"    /\\ UNCHANGED <<{', '.join(unchanged_vars)}>>")
        lines.append("")

    def _emit_next(self, lines: list[str]) -> None:
        if not self.actions:
            return
        _tla_section_header(lines, "Next State")
        lines.append("Next ==")
        for action in self.actions:
            if action.parameters:
                param_bindings = ", ".join(
                    f"{p.name} \\in {p.domain}" for p in action.parameters
                )
                param_names = ", ".join(p.name for p in action.parameters)
                action_call = f"\\E {param_bindings} : {action.name}({param_names})"
            else:
                action_call = action.name
            lines.append(f"    \\/ {action_call}")
        lines.append("")

    def _emit_spec(self, lines: list[str]) -> None:
        if self.state_vars and self.actions:
            _tla_section_header(lines, "Specification")
            lines.append("Spec == Init /\\ [][Next]_vars")
            lines.append("")

    def _emit_invariants(self, lines: list[str]) -> None:
        if not self.invariants:
            return
        _tla_section_header(lines, "Invariants")
        for inv in self.invariants:
            if inv.description:
                lines.append(f"(* {inv.id}: {inv.description} *)")
            else:
                lines.append(f"(* {inv.id} *)")
            lines.append(f"{inv.name} ==")
            pred_lines = inv.predicate.split("\n")
            if len(pred_lines) == 1:
                lines.append(f"    {inv.predicate}")
            else:
                for pline in pred_lines:
                    lines.append(f"    {pline}" if pline.strip() else "")
            lines.append("")

    def _emit_constraint(self, lines: list[str]) -> None:
        if not self.constraint:
            return
        _tla_section_header(lines, "State Constraint")
        lines.append(f"StateConstraint == {self.constraint}")
        lines.append("")

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
