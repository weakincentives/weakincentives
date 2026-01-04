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

"""Tests for TLA+ spec extraction and validation."""

from __future__ import annotations

from weakincentives.formal import Action, FormalSpec, Invariant, StateVar


def test_formal_spec_to_tla():
    """Test FormalSpec.to_tla() generates valid TLA+ module."""
    spec = FormalSpec(
        module="TestCounter",
        constants={"MaxCount": 10},
        state_vars=(StateVar("count", "Nat", "Current count"),),
        actions=(
            Action(
                name="Increment",
                updates={"count": "count + 1"},
            ),
        ),
        invariants=(Invariant("INV-1", "NonNegative", "count >= 0"),),
    )

    tla = spec.to_tla()

    # Check module header
    assert "MODULE TestCounter" in tla

    # Check extends
    assert "EXTENDS Integers, Sequences, FiniteSets" in tla

    # Check constants
    assert "CONSTANTS" in tla
    assert "MaxCount" in tla

    # Check variables
    assert "VARIABLES" in tla
    assert "count" in tla
    assert "vars == <<count>>" in tla

    # Check action
    assert "Increment ==" in tla
    assert "count' = count + 1" in tla

    # Check invariant
    assert "INV-1" in tla
    assert "NonNegative ==" in tla
    assert "count >= 0" in tla


def test_formal_spec_to_tla_config():
    """Test FormalSpec.to_tla_config() generates valid TLC config."""
    spec = FormalSpec(
        module="TestCounter",
        constants={"MaxCount": 10},
        invariants=(Invariant("INV-1", "NonNegative", "count >= 0"),),
    )

    cfg = spec.to_tla_config()

    # Check specification
    assert "SPECIFICATION Spec" in cfg

    # Check constants
    assert "CONSTANTS" in cfg
    assert "MaxCount = 10" in cfg

    # Check invariants
    assert "INVARIANTS" in cfg
    assert "NonNegative" in cfg


def test_formal_spec_decorator_attachment():
    """Test @formal_spec decorator attaches metadata."""
    from weakincentives.formal import formal_spec

    @formal_spec(
        module="TestClass",
        state_vars=[StateVar("x", "Int")],
    )
    class TestClass:
        pass

    assert hasattr(TestClass, "__formal_spec__")
    spec = TestClass.__formal_spec__
    assert isinstance(spec, FormalSpec)
    assert spec.module == "TestClass"
    assert len(spec.state_vars) == 1
    assert spec.state_vars[0].name == "x"


def test_state_var_type_defaults():
    """Test StateVar type defaults in Init block."""
    # Test all type-based defaults
    spec = FormalSpec(
        module="TestTypes",
        state_vars=(
            StateVar("seq_var", "Seq(Int)", "Sequence variable"),
            StateVar("set_var", "Set(Int)", "Set variable"),
            StateVar("func_var", "Function", "Function variable"),
            StateVar("custom_var", "CustomType", "Custom type with NULL default"),
            StateVar("explicit_var", "Int", "Explicit initial", initial_value="42"),
        ),
    )

    tla = spec.to_tla()

    # Check Seq default
    assert "seq_var = <<>>" in tla

    # Check Set default
    assert "set_var = {}" in tla

    # Check Function default
    assert "func_var = [x \\in {} |-> 0]" in tla

    # Check NULL fallback for unknown type
    assert "custom_var = NULL" in tla

    # Check explicit initial value overrides type default
    assert "explicit_var = 42" in tla


def test_action_with_updates():
    """Test action with state updates."""
    spec = FormalSpec(
        module="TestUpdates",
        state_vars=(
            StateVar("x", "Int"),
            StateVar("y", "Int"),
            StateVar("z", "Int"),
        ),
        actions=(
            Action(
                name="Update",
                preconditions=("x > 0",),
                updates={"x": "x - 1", "y": "y + 1"},
                # z is unchanged
            ),
        ),
    )

    tla = spec.to_tla()

    # Check updates are generated
    assert "x' = x - 1" in tla
    assert "y' = y + 1" in tla

    # Check UNCHANGED for non-updated vars
    assert "UNCHANGED <<z>>" in tla


def test_spec_with_constraint():
    """Test spec with state constraint."""
    spec = FormalSpec(
        module="TestConstraint",
        state_vars=(StateVar("depth", "Nat"),),
        constraint="depth <= 100",
    )

    tla = spec.to_tla()

    # Check StateConstraint operator is generated
    assert "StateConstraint == depth <= 100" in tla
    assert "(* State Constraint *)" in tla

    # Check .cfg uses StateConstraint
    cfg = spec.to_tla_config()
    assert "CONSTRAINT StateConstraint" in cfg


def test_config_with_init_next():
    """Test config with custom INIT and NEXT."""
    spec = FormalSpec(
        module="TestCustom",
        constants={"N": 5},
        invariants=(Invariant("INV-1", "Safety", "TRUE"),),
    )

    cfg = spec.to_tla_config(init="CustomInit", next="CustomNext")

    # Check INIT/NEXT instead of SPECIFICATION
    assert "INIT CustomInit" in cfg
    assert "NEXT CustomNext" in cfg
    assert "SPECIFICATION Spec" not in cfg


def test_config_with_state_constraint_param():
    """Test config with state constraint parameter."""
    spec = FormalSpec(
        module="TestConstraintParam",
        constraint="depth <= 50",  # Instance constraint
    )

    # Override with parameter constraint
    cfg = spec.to_tla_config(state_constraint="depth <= 25")

    # Parameter constraint takes precedence
    assert "CONSTRAINT depth <= 25" in cfg
    assert "CONSTRAINT StateConstraint" not in cfg


def test_config_check_deadlock_true():
    """Test config with deadlock checking enabled (default)."""
    spec = FormalSpec(module="TestDeadlock")

    # Default: check_deadlock=True (no CHECK_DEADLOCK line)
    cfg = spec.to_tla_config(check_deadlock=True)
    assert "CHECK_DEADLOCK" not in cfg

    # Explicit False adds the line
    cfg = spec.to_tla_config(check_deadlock=False)
    assert "CHECK_DEADLOCK FALSE" in cfg


def test_spec_without_constants():
    """Test spec without constants section."""
    spec = FormalSpec(
        module="TestNoConstants",
        state_vars=(StateVar("x", "Int"),),
    )

    cfg = spec.to_tla_config()

    # Should not have CONSTANTS section
    assert "CONSTANTS" not in cfg


def test_spec_without_invariants():
    """Test spec without invariants section."""
    spec = FormalSpec(
        module="TestNoInvariants",
        state_vars=(StateVar("x", "Int"),),
    )

    cfg = spec.to_tla_config()

    # Should not have INVARIANTS section
    assert "INVARIANTS" not in cfg


def test_spec_without_extends():
    """Test spec without extends clause."""
    spec = FormalSpec(
        module="TestNoExtends",
        extends=(),  # Empty extends
        state_vars=(StateVar("x", "Int"),),
    )

    tla = spec.to_tla()

    # Should not have EXTENDS line
    assert "EXTENDS" not in tla


def test_action_without_updates():
    """Test action without state updates (only preconditions)."""
    spec = FormalSpec(
        module="TestNoUpdates",
        state_vars=(
            StateVar("x", "Int"),
            StateVar("y", "Int"),
        ),
        actions=(
            Action(
                name="CheckOnly",
                preconditions=("x > 0", "y > 0"),
                # No updates - all vars unchanged
            ),
        ),
    )

    tla = spec.to_tla()

    # Check preconditions are generated
    assert "x > 0" in tla
    assert "y > 0" in tla

    # Check all vars are UNCHANGED
    assert "UNCHANGED <<x, y>>" in tla


def test_config_with_only_init():
    """Test config with only init parameter (edge case)."""
    spec = FormalSpec(module="TestOnlyInit")

    # Only init, no next (neither branch executes - no SPEC line generated)
    cfg = spec.to_tla_config(init="CustomInit", next=None)

    # Neither default nor custom INIT/NEXT is generated
    assert "SPECIFICATION" not in cfg
    assert "INIT" not in cfg
    assert "NEXT" not in cfg


def test_action_with_parameters():
    """Test Action with parameters."""
    from weakincentives.formal import ActionParameter

    action = Action(
        name="Receive",
        parameters=(
            ActionParameter("consumer", "1..NumConsumers"),
            ActionParameter("timeout", "0..MaxTimeout"),
        ),
        preconditions=("timeout > 0", "Len(pending) > 0"),
        updates={"pending": "Tail(pending)"},
        description="Receive a message",
    )

    spec = FormalSpec(
        module="Test",
        state_vars=(StateVar("pending", "Seq"),),
        actions=(action,),
    )

    tla = spec.to_tla()

    # Check action signature
    assert "Receive(consumer, timeout) ==" in tla

    # Check preconditions
    assert "/\\ timeout > 0" in tla
    assert "/\\ Len(pending) > 0" in tla

    # Check updates
    assert "pending' = Tail(pending)" in tla


def test_invariant_multiline_predicate():
    """Test Invariant with multi-line predicate."""
    inv = Invariant(
        id="INV-COMPLEX",
        name="ComplexInvariant",
        predicate="\\/ x > 0\n\\/ y > 0",
    )

    spec = FormalSpec(
        module="Test",
        state_vars=(StateVar("x", "Int"), StateVar("y", "Int")),
        invariants=(inv,),
    )

    tla = spec.to_tla()

    # Check invariant is present
    assert "ComplexInvariant ==" in tla
    assert "\\/ x > 0" in tla
    assert "\\/ y > 0" in tla


def test_helpers_in_spec():
    """Test helper operator definitions."""
    spec = FormalSpec(
        module="Test",
        helpers={
            "NULL": "0",
            "IsEmpty(seq)": "Len(seq) = 0",
        },
    )

    tla = spec.to_tla()

    # Check helpers section
    assert "Helper Operators" in tla
    assert "NULL ==" in tla
    assert "0" in tla
    assert "IsEmpty(seq) ==" in tla
    assert "Len(seq) = 0" in tla


def test_unchanged_variables():
    """Test UNCHANGED clause for variables not in updates."""
    spec = FormalSpec(
        module="Test",
        state_vars=(
            StateVar("x", "Int"),
            StateVar("y", "Int"),
            StateVar("z", "Int"),
        ),
        actions=(
            Action(
                name="UpdateX",
                updates={"x": "x + 1"},
            ),
        ),
    )

    tla = spec.to_tla()

    # Check UNCHANGED includes y and z but not x
    assert "UNCHANGED <<y, z>>" in tla or "UNCHANGED <<z, y>>" in tla
