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

"""Formal verification of RedisMailbox using TLA+ model checking.

This test extracts the TLA+ specification embedded in RedisMailbox via
the @formal_spec decorator and validates it using the TLC model checker.

Run with:
    pytest formal-tests/test_redis_mailbox.py              # Extract only
    pytest formal-tests/test_redis_mailbox.py -k model     # Model check
    pytest formal-tests/test_redis_mailbox.py -v           # Verbose
"""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.contrib.mailbox._redis import RedisMailbox
from weakincentives.formal.testing import extract_and_verify


def test_extract_redis_mailbox_spec(extracted_specs_dir: Path) -> None:
    """Extract RedisMailbox TLA+ specification to specs/tla/extracted/.

    This test extracts the embedded TLA+ spec and writes it to .tla and .cfg files.
    It verifies the spec structure but does not run model checking (fast).
    """
    spec, tla_file, cfg_file, _ = extract_and_verify(
        RedisMailbox,
        output_dir=extracted_specs_dir,
        model_check_enabled=False,
    )

    # Verify spec structure
    assert spec.module == "RedisMailbox"
    assert len(spec.state_vars) == 11, "Should have 11 state variables"
    assert len(spec.actions) == 10, "Should have 10 actions"
    assert len(spec.invariants) == 6, "Should have 6 invariants"

    # Verify files were created
    assert tla_file.exists(), f"TLA+ file should exist at {tla_file}"
    assert cfg_file.exists(), f"Config file should exist at {cfg_file}"

    # Verify file content basics
    tla_content = tla_file.read_text()
    assert "MODULE RedisMailbox" in tla_content
    assert "EXTENDS Integers, Sequences, FiniteSets, TLC" in tla_content

    cfg_content = cfg_file.read_text()
    assert "MaxMessages" in cfg_content
    assert "INVARIANTS" in cfg_content

    print(f"\n✓ Extracted {spec.module} to {tla_file}")


@pytest.mark.slow
@pytest.mark.skipif(
    not pytest.importorskip("subprocess").run(
        ["which", "tlc"], capture_output=True
    ).returncode
    == 0,
    reason="TLC not installed (brew install tlaplus)",
)
def test_model_check_redis_mailbox(
    extracted_specs_dir: Path,
    tlc_config: dict[str, str | bool],
) -> None:
    """Model check RedisMailbox specification with TLC (slow).

    This test runs the TLC model checker to verify all invariants hold.
    It may take 10-30 seconds depending on the state space size.

    Mark as slow since it runs TLC exhaustively.
    """
    spec, tla_file, cfg_file, result = extract_and_verify(
        RedisMailbox,
        output_dir=extracted_specs_dir,
        model_check_enabled=True,
        tlc_config=tlc_config,
    )

    # Verify model checking passed
    assert result is not None, "Model check result should be returned"
    assert result.passed, (
        f"Model checking failed for {spec.module}:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert result.states_generated > 0, "Should have generated states"

    print(
        f"\n✓ Model checking passed for {spec.module} "
        f"({result.states_generated} states generated)"
    )
