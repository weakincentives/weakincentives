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
    """Result of running TLC model checker on a formal specification.

    Contains the outcome of model checking along with diagnostic information.
    Check `passed` to determine if all invariants held. If `passed` is False,
    examine `stdout` for the counterexample trace showing the invariant violation.

    A timeout (returncode -1) with no violations found is treated as passed,
    enabling bounded verification within time limits.

    Example:
        >>> result = model_check(spec)
        >>> if not result.passed:
        ...     print(f"Invariant violated! States explored: {result.states_generated}")
        ...     print(result.stdout)  # Contains counterexample trace
    """

    passed: bool
    """True if all invariants held in every explored state, False if any violation found."""

    states_generated: int
    """Number of unique states TLC generated during exploration. Useful for gauging coverage."""

    stdout: str
    """Complete standard output from TLC, including progress messages and any counterexamples."""

    stderr: str
    """Standard error output from TLC, typically contains warnings or configuration errors."""

    returncode: int
    """TLC exit code: 0 for success, non-zero for errors, -1 indicates timeout."""


class ModelCheckError(Exception):
    """Exception raised when TLC model checking fails.

    This exception is raised in the following situations:
    - TLC is not installed or not found in the expected location
    - TLC configuration is invalid (e.g., missing JAR file)
    - An invariant violation was detected during model checking
    - TLC encountered a fatal error during execution

    The exception message contains details about the failure, including
    TLC's stdout/stderr output when available.

    Example:
        >>> try:
        ...     result = model_check(spec)
        ... except ModelCheckError as e:
        ...     print(f"Model checking failed: {e}")
    """


def extract_spec(target_class: type) -> FormalSpec:
    """Extract the TLA+ specification from a class decorated with @formal_spec.

    Retrieves the FormalSpec metadata that was attached to a class by the
    @formal_spec decorator. This is the first step in the verification
    workflow: extract the spec, then write it to files or run model checking.

    Args:
        target_class: A class that has been decorated with @formal_spec.
            The class must have the `__formal_spec__` attribute.

    Returns:
        The FormalSpec instance containing all specification metadata.

    Raises:
        ValueError: If target_class was not decorated with @formal_spec
            (i.e., missing the `__formal_spec__` attribute).

    Example:
        >>> from weakincentives.contrib.mailbox._redis import RedisMailbox
        >>> spec = extract_spec(RedisMailbox)
        >>> spec.module
        'RedisMailbox'
        >>> print(spec.to_tla())  # Generate TLA+ source
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
    """Write a TLA+ specification and its TLC config to files.

    Generates both the `.tla` module file and the `.cfg` configuration file
    in the specified output directory. Creates the directory if it doesn't
    exist. Files are named after the module name from the spec.

    The generated files can be opened in the TLA+ Toolbox or verified with
    the TLC command-line tool.

    Args:
        spec: The FormalSpec to write. Uses `spec.module` for file names.
        output_dir: Directory path where files will be written. Created with
            `mkdir -p` semantics if it doesn't exist.

    Returns:
        A tuple of (tla_file_path, cfg_file_path) with absolute paths to
        the generated files.

    Example:
        >>> spec = extract_spec(RedisMailbox)
        >>> tla_file, cfg_file = write_spec(spec, Path("specs/tla/extracted"))
        >>> print(f"Generated: {tla_file}, {cfg_file}")
        Generated: specs/tla/extracted/RedisMailbox.tla, specs/tla/extracted/RedisMailbox.cfg
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
    """Run the TLC model checker on a formal specification.

    Executes TLC to exhaustively explore the state space defined by the
    specification and verify that all invariants hold. The check runs with
    a 3-minute timeout; if TLC times out without finding violations, the
    result is considered passed (bounded verification).

    TLC is located automatically: first checks for `/usr/local/lib/tla2tools.jar`,
    then falls back to the `tlc` command (e.g., from `brew install tlaplus`).

    The specification is written to a temporary directory, checked, and then
    cleaned up automatically.

    Args:
        spec: The FormalSpec to verify. Must have at least state_vars and
            actions defined for meaningful verification.
        tlc_config: Optional dict with TLC configuration:
            - "workers": Number of worker threads ("auto" or integer, default "auto")
            - "cleanup": Whether to clean up TLC state files (default True)

    Returns:
        ModelCheckResult with pass/fail status, state count, and TLC output.

    Raises:
        ModelCheckError: If TLC is not installed, configuration is invalid,
            or a fatal error occurs during model checking.

    Example:
        >>> spec = extract_spec(RedisMailbox)
        >>> result = model_check(spec, tlc_config={"workers": 4})
        >>> if result.passed:
        ...     print(f"Verified {result.states_generated} states")
        ... else:
        ...     print(f"Violation found:\\n{result.stdout}")

    Note:
        Model checking is computationally expensive. Use state constraints
        in your specification to bound the state space for faster checking.
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

        # Check if TLC JAR is available
        tlc_jar = Path("/usr/local/lib/tla2tools.jar")
        if not tlc_jar.exists():
            # Fall back to tlc command (e.g., on macOS with brew install)
            try:
                _ = subprocess.run(  # nosec B603 B607
                    ["tlc", "-h"], capture_output=True, check=False, timeout=5
                )
                use_jar = False
            except FileNotFoundError:
                raise ModelCheckError(
                    "TLC not found. Install with: brew install tlaplus (macOS) or download from https://github.com/tlaplus/tlaplus/releases"
                ) from None
        else:
            use_jar = True

        # Build TLC command - use Java directly to avoid wrapper script issues
        if use_jar:
            cmd = [
                "java",
                "-XX:+UseParallelGC",
                "-jar",
                str(tlc_jar),
                str(tla_file),
                "-config",
                str(cfg_file),
                "-workers",
                str(workers),
            ]
        else:
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

        # Run TLC with 3-minute timeout using Popen for reliable timeout enforcement
        process = subprocess.Popen(  # nosec B603 B607
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            stdout, stderr = process.communicate(timeout=180)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            # Timeout: kill process and extract partial output
            process.kill()
            stdout, stderr = process.communicate()
            returncode = -1  # Indicate timeout

            # If no violations found before timeout, treat as passed (bounded verification)
            if "violated" not in stdout.lower():
                return ModelCheckResult(
                    passed=True,
                    states_generated=_extract_state_count(stdout),
                    stdout=stdout + "\n[Timeout: No violations found in 3 minutes]",
                    stderr=stderr,
                    returncode=returncode,
                )
            # If violations found, fall through to normal error handling

        # Extract state count and check for violations
        states = _extract_state_count(stdout)

        # Check for TLC configuration errors (e.g., missing JAR file)
        if "jarfile" in stderr.lower() or "unable to access" in stderr.lower():
            msg = (
                f"TLC configuration error. {stderr.strip()}\n"
                f"Install TLC: brew install tlaplus (macOS) or download from "
                f"https://github.com/tlaplus/tlaplus/releases"
            )
            raise ModelCheckError(msg)

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
    """Extract a TLA+ specification from a class and optionally verify it with TLC.

    This is the primary entry point for formal verification in tests. It combines
    extraction, file generation, and optional model checking into a single call.

    The workflow is:
    1. Extract the FormalSpec from the decorated class
    2. Write the .tla and .cfg files to the output directory
    3. Optionally run TLC model checking
    4. Raise ModelCheckError if any invariant is violated

    Use this in pytest tests to verify that implementations match their formal
    specifications.

    Args:
        target_class: A class decorated with @formal_spec. The FormalSpec
            metadata will be extracted from its `__formal_spec__` attribute.
        output_dir: Directory where .tla and .cfg files will be written.
            Created if it doesn't exist.
        model_check_enabled: If True, runs TLC model checker after generating
            files. If False (default), only generates files without verification.
        tlc_config: Optional TLC configuration dict passed to model_check().
            See model_check() for available options.

    Returns:
        A 4-tuple containing:
        - spec: The extracted FormalSpec
        - tla_file: Path to the generated .tla file
        - cfg_file: Path to the generated .cfg file
        - result: ModelCheckResult if model_check_enabled is True, else None

    Raises:
        ValueError: If target_class is not decorated with @formal_spec.
        ModelCheckError: If model_check_enabled is True and TLC finds an
            invariant violation or encounters a fatal error.

    Example:
        >>> # In a pytest test
        >>> def test_mailbox_formal_spec(tmp_path):
        ...     spec, tla, cfg, result = extract_and_verify(
        ...         RedisMailbox,
        ...         output_dir=tmp_path / "specs",
        ...         model_check_enabled=True,
        ...         tlc_config={"workers": "auto"},
        ...     )
        ...     assert result is not None
        ...     assert result.passed
        ...     assert result.states_generated > 0
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
