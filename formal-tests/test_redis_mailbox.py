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
the @formal_spec decorator and optionally validates it using TLC.

FAST by default (~1s):
    pytest formal-tests/test_redis_mailbox.py

Full verification with model checking (~30s):
    pytest formal-tests/test_redis_mailbox.py --model-check

Persist extracted specs:
    pytest formal-tests/test_redis_mailbox.py --persist-specs
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from weakincentives.contrib.mailbox._redis import RedisMailbox
from weakincentives.formal.testing import extract_and_verify


def test_redis_mailbox_spec(
    extracted_specs_dir: Path,
    enable_model_checking: bool,
    tlc_config: dict[str, str | bool],
) -> None:
    """Extract and optionally verify RedisMailbox TLA+ specification.

    By default: Fast extraction only (~1s, temp dir)
    With --model-check: Full TLC verification (~30s)
    With --persist-specs: Write to specs/tla/extracted/
    """
    # Check if TLC is available if model checking is requested
    if enable_model_checking:
        try:
            subprocess.run(
                ["which", "tlc"], capture_output=True, check=True, timeout=5
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("TLC not installed (brew install tlaplus)")

    # Extract and optionally verify
    spec, tla_file, cfg_file, result = extract_and_verify(
        RedisMailbox,
        output_dir=extracted_specs_dir,
        model_check_enabled=enable_model_checking,
        tlc_config=tlc_config if enable_model_checking else None,
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

    # Report results
    if enable_model_checking:
        assert result is not None, "Model check result should be returned"
        assert result.passed, (
            f"Model checking failed:\n{result.stdout}\n{result.stderr}"
        )
        print(
            f"\n✓ Model checking passed ({result.states_generated} states generated)"
        )
        print(f"✓ Spec written to {tla_file}")
    else:
        print(f"\n✓ Extraction validated (skipped model checking for speed)")
        print(f"✓ Spec written to {tla_file}")
        print("\nTo run full verification: pytest formal-tests/ --model-check")
