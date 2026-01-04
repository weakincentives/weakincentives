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
