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
the @formal_spec decorator and validates it using TLC.

Model checking is ENABLED by default (~30s):
    pytest formal-tests/test_redis_mailbox.py

For development speed (extraction only, ~1s):
    pytest formal-tests/test_redis_mailbox.py --skip-model-check

Persist extracted specs:
    pytest formal-tests/test_redis_mailbox.py --persist-specs
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from weakincentives.contrib.mailbox._redis import RedisMailbox
from weakincentives.formal.testing import ModelCheckError, extract_and_verify


def test_redis_mailbox_spec(
    extracted_specs_dir: Path,
    enable_model_checking: bool,
    tlc_config: dict[str, str | bool],
) -> None:
    """Extract and verify RedisMailbox TLA+ specification.

    By default: Full TLC model checking (~30s, temp dir)
    With --skip-model-check: Fast extraction only (~1s, development)
    With --persist-specs: Write to specs/tla/extracted/
    """
    # Check if TLC is available if model checking is requested
    if enable_model_checking:
        try:
            subprocess.run(["which", "tlc"], capture_output=True, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("TLC not installed (brew install tlaplus)")

    # Extract and optionally verify
    try:
        spec, tla_file, cfg_file, result = extract_and_verify(
            RedisMailbox,
            output_dir=extracted_specs_dir,
            model_check_enabled=enable_model_checking,
            tlc_config=tlc_config if enable_model_checking else None,
        )
    except ModelCheckError as e:
        # Skip if TLC is not properly configured (e.g., missing JAR file)
        if "configuration error" in str(e).lower() or "jarfile" in str(e).lower():
            pytest.skip(f"TLC not properly configured: {e}")
        # Re-raise if it's an actual model checking failure (invariant violation)
        raise

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

        # Print TLC output
        print("\n" + "="*80)
        print("TLC MODEL CHECKER OUTPUT")
        print("="*80)
        print(result.stdout)
        if result.stderr:
            print("\n--- STDERR ---")
            print(result.stderr)
        print("="*80)

        assert result.passed, (
            f"Model checking failed:\n{result.stdout}\n{result.stderr}"
        )
        print(f"\n✓ Model checking passed ({result.states_generated} states generated)")
        print(f"✓ Spec written to {tla_file}")
    else:
        print("\n✓ Extraction validated (model checking skipped)")
        print(f"✓ Spec written to {tla_file}")
        print("\n⚠️  Model checking was skipped - use for development only!")
        print("Run without --skip-model-check for full verification")
