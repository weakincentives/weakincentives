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

"""Pytest plugin for extracting and validating TLA+ specifications.

This plugin extracts TLA+ specs embedded in Python code via the
`@formal_spec` decorator and validates them using the TLC model checker.

Usage:
    pytest --extract-tla         # Extract specs to specs/tla/extracted/
    pytest --check-tla           # Extract and validate with TLC
    pytest --tla-output-dir=DIR  # Custom output directory
"""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from weakincentives.formal import FormalSpec


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command-line options for TLA+ extraction and validation."""
    group = parser.getgroup("tla", "TLA+ formal specification extraction")

    group.addoption(
        "--extract-tla",
        action="store_true",
        default=False,
        help="Extract TLA+ specs from @formal_spec decorators",
    )

    group.addoption(
        "--check-tla",
        action="store_true",
        default=False,
        help="Extract and validate TLA+ specs with model checker",
    )

    group.addoption(
        "--tla-output-dir",
        type=Path,
        default=None,
        help="Output directory for extracted TLA+ specs (default: specs/tla/extracted)",
    )

    group.addoption(
        "--tla-checker",
        type=str,
        default="tlc",
        choices=["tlc", "apalache"],
        help="Model checker to use (default: tlc)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Extract and optionally validate TLA+ specs."""
    extract = config.getoption("--extract-tla")
    check = config.getoption("--check-tla")

    if not (extract or check):
        return

    # Determine output directory
    output_dir = config.getoption("--tla-output-dir")
    if output_dir is None:
        output_dir = Path("specs/tla/extracted")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract specs
    extractor = FormalSpecExtractor()
    specs = extractor.extract_all()

    if not specs:
        pytest.skip("No @formal_spec decorators found in codebase")
        return

    # Write to files
    spec_files = []
    for module_name, spec in specs.items():
        tla_file = output_dir / f"{spec.module}.tla"
        cfg_file = output_dir / f"{spec.module}.cfg"

        tla_file.write_text(spec.to_tla())
        cfg_file.write_text(spec.to_tla_config())

        spec_files.append((tla_file, cfg_file))

        print(f"✓ Extracted {spec.module} to {tla_file}")

    # Validate with model checker if requested
    if check:
        checker = config.getoption("--tla-checker")
        for tla_file, cfg_file in spec_files:
            try:
                run_model_checker(tla_file, cfg_file, checker=checker)
                print(f"✓ Model checking passed for {tla_file.name}")
            except ModelCheckError as e:
                pytest.fail(str(e))


class FormalSpecExtractor:
    """Extract TLA+ specs from @formal_spec decorators."""

    def extract_all(self) -> dict[str, FormalSpec]:
        """Extract all formal specs from the codebase.

        Returns:
            Mapping from module name to FormalSpec
        """
        specs: dict[str, FormalSpec] = {}

        # Add src to path if not already there
        src_path = Path("src").resolve()
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        # Import all modules in weakincentives
        for py_file in Path("src/weakincentives").rglob("*.py"):
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                continue

            module_name = (
                py_file.relative_to("src")
                .with_suffix("")
                .as_posix()
                .replace("/", ".")
            )

            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                # Skip modules that fail to import
                print(f"⚠ Skipped {module_name}: {e}")
                continue

            # Extract specs from module
            try:
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    spec = getattr(obj, "__formal_spec__", None)
                    if spec is not None:
                        specs[f"{module_name}.{name}"] = spec
            except Exception:
                # Skip modules where inspection fails (e.g., lazy imports)
                continue

        return specs


class ModelCheckError(Exception):
    """Model checker found an error."""


def run_model_checker(
    tla_file: Path,
    cfg_file: Path,
    *,
    checker: str = "tlc",
) -> None:
    """Run model checker on TLA+ spec.

    Args:
        tla_file: Path to .tla file
        cfg_file: Path to .cfg file
        checker: Model checker to use ("tlc" or "apalache")

    Raises:
        ModelCheckError: If model checking fails
    """
    if checker == "tlc":
        run_tlc(tla_file, cfg_file)
    elif checker == "apalache":
        run_apalache(tla_file)
    else:
        raise ValueError(f"Unknown model checker: {checker}")


def run_tlc(tla_file: Path, cfg_file: Path) -> None:
    """Run TLC model checker.

    Args:
        tla_file: Path to .tla file
        cfg_file: Path to .cfg file

    Raises:
        ModelCheckError: If TLC finds an error
    """
    # Check if tlc is available
    try:
        subprocess.run(["tlc", "-h"], capture_output=True, check=False)
    except FileNotFoundError:
        raise ModelCheckError(
            "TLC not found. Install with: brew install tlaplus (macOS) or "
            "download from https://github.com/tlaplus/tlaplus/releases"
        ) from None

    # Run TLC
    cmd = [
        "tlc",
        str(tla_file),
        "-config",
        str(cfg_file),
        "-workers",
        "auto",
        "-cleanup",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    # Check for errors
    if result.returncode != 0:
        raise ModelCheckError(
            f"TLC model checking failed for {tla_file.name}:\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    # Check for invariant violations in output
    if "Invariant" in result.stdout and "violated" in result.stdout:
        raise ModelCheckError(
            f"Invariant violation in {tla_file.name}:\n{result.stdout}"
        )


def run_apalache(tla_file: Path) -> None:
    """Run Apalache model checker.

    Args:
        tla_file: Path to .tla file

    Raises:
        ModelCheckError: If Apalache finds an error
    """
    # Check if apalache is available
    try:
        subprocess.run(["apalache-mc", "--version"], capture_output=True, check=False)
    except FileNotFoundError:
        raise ModelCheckError(
            "Apalache not found. Install from https://github.com/informalsystems/apalache"
        ) from None

    # Run Apalache
    cmd = ["apalache-mc", "check", str(tla_file)]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise ModelCheckError(
            f"Apalache model checking failed for {tla_file.name}:\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )
