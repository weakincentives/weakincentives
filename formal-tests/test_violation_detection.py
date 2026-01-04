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

"""Test that TLC correctly detects invariant violations.

This test uses a simple counter spec with an intentional violation to
validate that the formal verification infrastructure properly detects
and reports violations.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from weakincentives.formal import Action, Invariant, StateVar, formal_spec
from weakincentives.formal.testing import ModelCheckError, extract_and_verify


@formal_spec(
    module="BrokenCounter",
    constants={"MaxValue": 3},
    state_vars=[StateVar("count", "Int", "Counter value")],
    actions=[
        Action(
            name="Increment",
            preconditions=(),  # No preconditions - will violate MaxValue
            updates={"count": "count + 1"},
        )
    ],
    invariants=[
        Invariant(
            id="INV-1",
            name="CountInRange",
            predicate="count >= 0 /\\ count <= MaxValue",
            description="Counter should stay within valid range",
        )
    ],
    constraint="count <= 5",  # Small state space for fast testing
)
class BrokenCounter:
    """Counter with intentional invariant violation for testing."""

    pass


def test_violation_detected(
    extracted_specs_dir: Path,
    enable_model_checking: bool,
    tlc_config: dict[str, str | bool],
) -> None:
    """Verify that TLC detects the invariant violation in BrokenCounter.

    This test ensures that:
    1. TLC runs successfully
    2. The violation is detected
    3. The error message indicates which invariant was violated
    """
    # Check if TLC is available if model checking is requested
    if enable_model_checking:
        try:
            subprocess.run(["which", "tlc"], capture_output=True, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("TLC not installed (brew install tlaplus)")

    # Extract and verify - should raise ModelCheckError due to violation
    if enable_model_checking:
        with pytest.raises(ModelCheckError) as exc_info:
            extract_and_verify(
                BrokenCounter,
                output_dir=extracted_specs_dir,
                model_check_enabled=True,
                tlc_config=tlc_config,
            )

        # Verify the error message indicates the violation
        error_msg = str(exc_info.value)
        assert "violated" in error_msg.lower() or "failed" in error_msg.lower()
    else:
        # Just test extraction when model checking is disabled
        spec, tla_file, cfg_file, _result = extract_and_verify(
            BrokenCounter,
            output_dir=extracted_specs_dir,
            model_check_enabled=False,
            tlc_config=None,
        )

        # Verify spec structure
        assert spec.module == "BrokenCounter"
        assert len(spec.state_vars) == 1
        assert len(spec.actions) == 1
        assert len(spec.invariants) == 1

        # Verify files were created
        assert tla_file.exists()
        assert cfg_file.exists()
