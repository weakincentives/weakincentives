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

"""Configuration for formal verification tests.

This directory contains tests that extract and validate TLA+ specifications
embedded in Python code using the @formal_spec decorator.

Tests in this directory:
- Extract TLA+ specs from @formal_spec decorators
- Write them to specs/tla/extracted/
- Optionally run TLC model checker for validation

Run with:
    pytest formal-tests/              # Extract all specs
    pytest formal-tests/ -k model     # Only run model checking tests
    pytest formal-tests/ -v           # Verbose output
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Output directory for extracted TLA+ specifications
EXTRACTED_SPECS_DIR = Path("specs/tla/extracted")

# TLC model checker configuration
TLC_CONFIG = {
    "workers": "auto",  # Use all available CPUs
    "cleanup": True,  # Remove temporary files after checking
}


@pytest.fixture(scope="session")
def extracted_specs_dir() -> Path:
    """Output directory for extracted TLA+ specs.

    This directory is created automatically and specs are written here
    by the formal specification test helpers.
    """
    EXTRACTED_SPECS_DIR.mkdir(parents=True, exist_ok=True)
    return EXTRACTED_SPECS_DIR


@pytest.fixture(scope="session")
def tlc_config() -> dict[str, str | bool]:
    """TLC model checker configuration."""
    return TLC_CONFIG
