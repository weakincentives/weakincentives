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

Tests are FAST by default:
- Use temporary directory (no filesystem pollution)
- Skip model checking (extraction only, ~1s)

For full verification:
    pytest formal-tests/ --model-check           # Run TLC model checker
    pytest formal-tests/ --persist-specs         # Write to specs/tla/extracted/
    pytest formal-tests/ --model-check --persist-specs  # Both

Makefile convenience targets:
    make verify-formal        # Fast extraction (temp dir)
    make verify-formal-full   # Full model checking + persist
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Default TLC model checker configuration (fast settings)
DEFAULT_TLC_CONFIG = {
    "workers": "auto",  # Use all available CPUs
    "cleanup": True,  # Remove temporary files after checking
}


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command-line options for formal verification."""
    group = parser.getgroup("formal", "Formal verification options")

    group.addoption(
        "--persist-specs",
        action="store_true",
        default=False,
        help="Write extracted specs to specs/tla/extracted/ (default: temp dir)",
    )

    group.addoption(
        "--model-check",
        action="store_true",
        default=False,
        help="Run TLC model checking (slow, default: extraction only)",
    )


@pytest.fixture(scope="session")
def extracted_specs_dir(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Output directory for extracted TLA+ specs.

    By default uses a temporary directory for fast, clean runs.
    Use --persist-specs to write to specs/tla/extracted/ instead.
    """
    persist = request.config.getoption("--persist-specs")

    if persist:
        output_dir = Path("specs/tla/extracted")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    else:
        # Use pytest's temp directory (auto-cleaned after session)
        return tmp_path_factory.mktemp("tla_specs")


@pytest.fixture(scope="session")
def enable_model_checking(request: pytest.FixtureRequest) -> bool:
    """Whether to run TLC model checking.

    Disabled by default for fast test runs (extraction only).
    Use --model-check to enable full verification.
    """
    return bool(request.config.getoption("--model-check"))


@pytest.fixture(scope="session")
def tlc_config() -> dict[str, str | bool]:
    """TLC model checker configuration."""
    return DEFAULT_TLC_CONFIG
