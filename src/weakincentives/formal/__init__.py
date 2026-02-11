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
from typing import Any, TypeVar

from weakincentives.formal._codegen import FormalSpec
from weakincentives.formal._metadata import (
    Action,
    ActionParameter,
    Invariant,
    StateVar,
)

__all__ = [
    "Action",
    "ActionParameter",
    "FormalSpec",
    "Invariant",
    "StateVar",
    "formal_spec",
]

T = TypeVar("T")


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
