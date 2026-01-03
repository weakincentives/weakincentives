#!/usr/bin/env python3
"""Standalone demo of TLA+ embedding without full weakincentives imports."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weakincentives.formal import Action, FormalSpec, Invariant, StateVar, formal_spec


# Simple counter example
@formal_spec(
    module="Counter",
    state_vars=[StateVar("count", "Nat", "Current count value")],
    actions=[
        Action(name="Increment", updates={"count": "count + 1"}),
        Action(
            name="Decrement",
            preconditions=("count > 0",),
            updates={"count": "count - 1"},
        ),
    ],
    invariants=[Invariant("INV-1", "NonNegative", "count >= 0")],
)
class Counter:
    """Simple counter with TLA+ spec."""

    def __init__(self):
        self.count = 0


def main():
    """Show the extracted TLA+ spec."""
    print("=" * 77)
    print("Extracted TLA+ Specification for Counter")
    print("=" * 77)
    print()

    spec = Counter.__formal_spec__
    print(spec.to_tla())
    print()

    print("=" * 77)
    print("TLC Configuration for Counter")
    print("=" * 77)
    print()
    print(spec.to_tla_config())


if __name__ == "__main__":
    main()
