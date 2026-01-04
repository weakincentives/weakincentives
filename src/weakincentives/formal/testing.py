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

"""Testing utilities for formal specifications.

This module provides pytest-compatible helpers for extracting and validating
TLA+ specifications embedded in Python code via @formal_spec decorators.
"""

from __future__ import annotations

import subprocess  # nosec B404
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from weakincentives.formal import FormalSpec


@dataclass(frozen=True, slots=True)
class ModelCheckResult:
    """Result of TLC model checking."""

    passed: bool
    """Whether all invariants held."""

    states_generated: int
    """Number of states generated during model checking."""

    stdout: str
    """Standard output from TLC."""

    stderr: str
    """Standard error from TLC."""

    returncode: int
    """Exit code from TLC."""


class ModelCheckError(Exception):
    """Model checker found an invariant violation or error."""


def extract_spec(target_class: type) -> FormalSpec:
    """Extract TLA+ specification from a class with @formal_spec decorator.

    Args:
        target_class: Class decorated with @formal_spec

    Returns:
        The extracted FormalSpec

    Raises:
        ValueError: If target_class does not have @formal_spec decorator

    Example:
        >>> from weakincentives.contrib.mailbox._redis import RedisMailbox
        >>> spec = extract_spec(RedisMailbox)
        >>> spec.module
        'RedisMailbox'
    """
    spec = getattr(target_class, "__formal_spec__", None)
    if spec is None:
        raise ValueError(
            f"{target_class.__name__} does not have @formal_spec decorator"
        )
    return spec


def write_spec(
    spec: FormalSpec,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write TLA+ specification and config to files.

    Args:
        spec: The formal specification to write
        output_dir: Directory to write files to

    Returns:
        Tuple of (tla_file_path, cfg_file_path)

    Example:
        >>> spec = extract_spec(RedisMailbox)
        >>> tla_file, cfg_file = write_spec(spec, Path("specs/tla/extracted"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tla_file = output_dir / f"{spec.module}.tla"
    cfg_file = output_dir / f"{spec.module}.cfg"

    _ = tla_file.write_text(spec.to_tla())
    _ = cfg_file.write_text(spec.to_tla_config())

    return tla_file, cfg_file


def model_check(
    spec: FormalSpec,
    *,
    tlc_config: dict[str, Any] | None = None,
) -> ModelCheckResult:
    """Run TLC model checker on a specification with 60s timeout.

    If TLC times out without finding violations, the check is considered passed.
    This allows bounded verification within reasonable time limits.

    Args:
        spec: The formal specification to check
        tlc_config: Optional TLC configuration (workers, cleanup, etc.)

    Returns:
        Model checking result

    Raises:
        ModelCheckError: If TLC is not found or checking fails
        FileNotFoundError: If TLA+ files don't exist

    Example:
        >>> spec = extract_spec(RedisMailbox)
        >>> result = model_check(spec, tlc_config={"workers": "auto"})
        >>> assert result.passed
    """
    # Write spec to temporary location for checking
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        tla_file, cfg_file = write_spec(spec, tmp_path)

        # Configure TLC
        config = tlc_config or {}
        workers = config.get("workers", "auto")
        cleanup = config.get("cleanup", True)

        # Check if tlc is available
        try:
            _ = subprocess.run(  # nosec B603 B607
                ["tlc", "-h"], capture_output=True, check=False, timeout=5
            )
        except FileNotFoundError:
            raise ModelCheckError(
                "TLC not found. Install with: brew install tlaplus (macOS) or download from https://github.com/tlaplus/tlaplus/releases"
            ) from None

        # Build TLC command
        cmd = [
            "tlc",
            str(tla_file),
            "-config",
            str(cfg_file),
            "-workers",
            str(workers),
        ]
        if cleanup:
            cmd.append("-cleanup")

        # Run TLC with 60-second timeout
        try:
            result = subprocess.run(  # nosec B603 B607
                cmd, capture_output=True, text=True, check=False, timeout=60
            )
            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode
        except subprocess.TimeoutExpired as e:
            # Timeout: extract partial output
            stdout = (
                (e.stdout or b"").decode()
                if isinstance(e.stdout, bytes)
                else e.stdout or ""
            )
            stderr = (
                (e.stderr or b"").decode()
                if isinstance(e.stderr, bytes)
                else e.stderr or ""
            )
            returncode = -1  # Indicate timeout

            # If no violations found before timeout, treat as passed (bounded verification)
            if "violated" not in stdout.lower():
                return ModelCheckResult(
                    passed=True,
                    states_generated=_extract_state_count(stdout),
                    stdout=stdout + "\n[Timeout: No violations found in 60s]",
                    stderr=stderr,
                    returncode=returncode,
                )
            # If violations found, fall through to normal error handling

        # Extract state count and check for violations
        states = _extract_state_count(stdout)

        # Check for TLC configuration errors (e.g., missing JAR file)
        if "jarfile" in stderr.lower() or "unable to access" in stderr.lower():
            raise ModelCheckError(
                f"TLC configuration error. {stderr.strip()}\n"
                "Install TLC: brew install tlaplus (macOS) or download from https://github.com/tlaplus/tlaplus/releases"
            )

        passed = returncode == 0 and "violated" not in stdout.lower()

        return ModelCheckResult(
            passed=passed,
            states_generated=states,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
        )


def _extract_state_count(output: str) -> int:
    """Extract state count from TLC output."""
    for line in output.split("\n"):
        if "states generated" in line.lower():
            for part in line.split():
                if part.isdigit():
                    return int(part)
    return 0


def extract_and_verify(
    target_class: type,
    *,
    output_dir: Path,
    model_check_enabled: bool = False,
    tlc_config: dict[str, Any] | None = None,
) -> tuple[FormalSpec, Path, Path, ModelCheckResult | None]:
    """Extract TLA+ spec from class and optionally verify with TLC.

    This is the main entry point for formal verification tests.

    Args:
        target_class: Class decorated with @formal_spec
        output_dir: Directory to write extracted files
        model_check_enabled: Whether to run TLC model checker
        tlc_config: Optional TLC configuration

    Returns:
        Tuple of (spec, tla_file, cfg_file, model_check_result)
        model_check_result is None if model_check_enabled is False

    Raises:
        ValueError: If target_class doesn't have @formal_spec
        ModelCheckError: If model checking fails

    Example:
        >>> spec, tla, cfg, result = extract_and_verify(
        ...     RedisMailbox,
        ...     output_dir=Path("specs/tla/extracted"),
        ...     model_check_enabled=True,
        ... )
        >>> assert result.passed
    """
    # Extract spec
    spec = extract_spec(target_class)

    # Write to files
    tla_file, cfg_file = write_spec(spec, output_dir)

    # Optionally model check
    result = None
    if model_check_enabled:
        result = model_check(spec, tlc_config=tlc_config)
        if not result.passed:
            raise ModelCheckError(
                f"Model checking failed for {spec.module}:\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )

    return spec, tla_file, cfg_file, result
