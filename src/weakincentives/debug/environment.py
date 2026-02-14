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

"""Reproducibility envelope capture for debug bundles.

Captures execution environment metadata needed to reproduce and debug issues:
- System information (OS, architecture, CPU, memory)
- Python runtime details
- Package dependencies
- Environment variables (with redaction)
- Git repository state
- Command invocation details
- Container runtime info (if applicable)

Example::

    from weakincentives.debug.environment import capture_environment

    env = capture_environment()
    print(env.system.os_name)
    print(env.python.version)
    print(env.git.commit_sha if env.git else "Not in a git repo")
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess  # nosec B404 - needed for system introspection
import sys
from collections.abc import Mapping
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING

from ..clock import SYSTEM_CLOCK
from ..dataclasses import FrozenDataclass
from ._git import GitInfo, capture_git_diff, capture_git_info

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)

# Environment variables to capture (case-insensitive prefixes)
_ALLOWED_ENV_PREFIXES = (
    "PYTHON",
    "PATH",
    "LANG",
    "LC_",
    "TZ",
    "HOME",
    "USER",
    "SHELL",
    "TERM",
    "VIRTUAL_ENV",
    "CONDA_",
    "UV_",
    "PIP_",
    "POETRY_",
    "NODE_",
    "NPM_",
    "CUDA_",
    "LD_LIBRARY_PATH",
    "DYLD_",
    "CI",
    "GITHUB_",
    "GITLAB_",
    "JENKINS_",
    "TRAVIS_",
    "CIRCLECI",
    "AZURE_",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "DOCKER_",
    "KUBERNETES_",
    "K8S_",
    "POD_",
    "CONTAINER_",
)

# Patterns for values that should be redacted
_REDACT_PATTERNS = (
    re.compile(r"(api[_-]?key|apikey)", re.IGNORECASE),
    re.compile(r"(secret|password|passwd|token|credential)", re.IGNORECASE),
    re.compile(r"(auth|bearer)", re.IGNORECASE),
    re.compile(r"(private[_-]?key)", re.IGNORECASE),
)

# Minimum expected parts in meminfo output lines
_MIN_MEMINFO_PARTS = 2

# Docker container ID length (64-character hex string)
_CONTAINER_ID_LENGTH = 64


@FrozenDataclass()
class SystemInfo:
    """System/OS information.

    Attributes:
        os_name: Operating system name (e.g., 'Linux', 'Darwin', 'Windows').
        os_release: OS version/release string.
        kernel_version: Kernel version (Linux) or system version.
        architecture: CPU architecture (e.g., 'x86_64', 'arm64').
        processor: Processor identifier.
        cpu_count: Number of logical CPU cores.
        memory_total_bytes: Total system memory in bytes, or None if unavailable.
        hostname: Machine hostname.
    """

    os_name: str = ""
    os_release: str = ""
    kernel_version: str = ""
    architecture: str = ""
    processor: str = ""
    cpu_count: int | None = None
    memory_total_bytes: int | None = None
    hostname: str = ""


@FrozenDataclass()
class PythonInfo:
    """Python runtime information.

    Attributes:
        version: Full Python version string.
        version_info: Tuple of (major, minor, micro).
        implementation: Python implementation (e.g., 'CPython', 'PyPy').
        executable: Path to the Python executable.
        prefix: sys.prefix value.
        base_prefix: sys.base_prefix value (differs from prefix in venvs).
        is_virtualenv: True if running in a virtual environment.
    """

    version: str = ""
    version_info: tuple[int, int, int] = (0, 0, 0)
    implementation: str = ""
    executable: str = ""
    prefix: str = ""
    base_prefix: str = ""
    is_virtualenv: bool = False


@FrozenDataclass()
class ContainerInfo:
    """Container runtime information.

    Attributes:
        runtime: Container runtime (e.g., 'docker', 'podman', 'containerd').
        container_id: Container ID if running inside a container.
        image: Container image name/tag.
        image_digest: Image digest (sha256:...) if available.
        cgroup_path: Cgroup path for the process.
        is_containerized: True if running inside a container.
    """

    runtime: str | None = None
    container_id: str | None = None
    image: str | None = None
    image_digest: str | None = None
    cgroup_path: str | None = None
    is_containerized: bool = False


@FrozenDataclass()
class CommandInfo:
    """Command invocation details.

    Attributes:
        argv: Command line arguments.
        working_dir: Current working directory.
        entrypoint: The main entry point script/module.
        executable: Python executable used.
    """

    argv: tuple[str, ...] = ()
    working_dir: str = ""
    entrypoint: str = ""
    executable: str = ""


@FrozenDataclass()
class EnvironmentCapture:
    """Complete environment capture for reproducibility.

    This is the main data structure capturing all environment information
    needed to reproduce a debug bundle's execution context.

    Attributes:
        system: System/OS information.
        python: Python runtime information.
        packages: Frozen package list (pip freeze output).
        env_vars: Filtered and redacted environment variables.
        git: Git repository state, or None if not in a repo.
        git_diff: Uncommitted changes diff (capped), or None.
        command: Command invocation details.
        container: Container runtime info, or None if not containerized.
        captured_at: ISO timestamp when capture occurred.
    """

    system: SystemInfo = field(default_factory=SystemInfo)
    python: PythonInfo = field(default_factory=PythonInfo)
    packages: str = ""
    env_vars: Mapping[str, str] = field(default_factory=lambda: {})
    git: GitInfo | None = None
    git_diff: str | None = None
    command: CommandInfo = field(default_factory=CommandInfo)
    container: ContainerInfo | None = None
    captured_at: str = ""


def _get_linux_memory_bytes() -> int | None:
    """Get total memory bytes from /proc/meminfo on Linux."""
    meminfo_path = Path("/proc/meminfo")
    try:
        with meminfo_path.open() as f:
            for line in f:
                if line.startswith("MemTotal:"):  # pragma: no cover
                    # Parse "MemTotal:    12345678 kB"
                    parts = line.split()
                    if len(parts) >= _MIN_MEMINFO_PARTS:
                        return int(parts[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _get_darwin_memory_bytes() -> int | None:
    """Get total memory bytes using sysctl on macOS."""
    result = subprocess.run(  # nosec B603 B607 - trusted system command
        ["sysctl", "-n", "hw.memsize"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode == 0:
        return int(result.stdout.strip())
    return None


def _capture_memory_bytes() -> int | None:
    """Capture total system memory in bytes."""
    platform_name = sys.platform
    try:
        if platform_name == "linux":
            return _get_linux_memory_bytes()
        if platform_name == "darwin":
            return _get_darwin_memory_bytes()
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _capture_system_info() -> SystemInfo:
    """Capture system/OS information."""
    return SystemInfo(
        os_name=platform.system(),
        os_release=platform.release(),
        kernel_version=platform.version(),
        architecture=platform.machine(),
        processor=platform.processor(),
        cpu_count=os.cpu_count(),
        memory_total_bytes=_capture_memory_bytes(),
        hostname=platform.node(),
    )


def _capture_python_info() -> PythonInfo:
    """Capture Python runtime information."""
    vi = sys.version_info
    return PythonInfo(
        version=sys.version,
        version_info=(vi.major, vi.minor, vi.micro),
        implementation=platform.python_implementation(),
        executable=sys.executable,
        prefix=sys.prefix,
        base_prefix=sys.base_prefix,
        is_virtualenv=sys.prefix != sys.base_prefix,
    )


def _capture_packages() -> str:
    """Capture installed packages via pip freeze or similar."""
    # Try uv first (if available)
    if shutil.which("uv"):
        try:
            result = subprocess.run(  # nosec B603 B607 - trusted package manager
                ["uv", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except subprocess.SubprocessError:
            pass

    # Fall back to pip freeze
    try:
        result = subprocess.run(  # nosec B603 - trusted Python executable
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except subprocess.SubprocessError:
        pass

    return ""


def _should_capture_env_var(name: str) -> bool:
    """Check if an environment variable should be captured."""
    upper_name = name.upper()
    return any(
        upper_name.startswith(prefix.upper()) for prefix in _ALLOWED_ENV_PREFIXES
    )


def _should_redact_value(name: str, _value: str) -> bool:
    """Check if an environment variable value should be redacted."""
    return any(pattern.search(name) for pattern in _REDACT_PATTERNS)


def _capture_env_vars() -> dict[str, str]:
    """Capture filtered and redacted environment variables."""
    result: dict[str, str] = {}

    for name, value in sorted(os.environ.items()):
        if _should_capture_env_var(name):
            if _should_redact_value(name, value):
                result[name] = "[REDACTED]"
            else:
                result[name] = value

    return result


def _is_valid_container_id(s: str) -> bool:
    """Check if a string looks like a container ID (64-char hex)."""
    return len(s) == _CONTAINER_ID_LENGTH and all(c in "0123456789abcdef" for c in s)


def _extract_container_id_from_cgroup(content: str) -> tuple[str | None, str | None]:
    """Extract container ID and runtime from cgroup content."""
    for line in content.splitlines():
        if "docker" not in line and "containerd" not in line:
            continue

        runtime = "docker" if "docker" in line else "containerd"
        parts = line.split("/")
        for part in reversed(parts):
            if _is_valid_container_id(part):
                return part, runtime

        return None, runtime

    return None, None


def _capture_container_info() -> ContainerInfo | None:
    """Capture container runtime information."""
    is_containerized = False
    container_id: str | None = None
    cgroup_path: str | None = None
    runtime: str | None = None

    try:
        # Check /.dockerenv
        if Path("/.dockerenv").exists():
            is_containerized = True
            runtime = "docker"

        # Check cgroup for container ID
        cgroup_file = Path("/proc/1/cgroup")
        if cgroup_file.exists():  # pragma: no cover
            content = cgroup_file.read_text()
            cgroup_path = content.strip()
            container_id, cgroup_runtime = _extract_container_id_from_cgroup(content)
            if cgroup_runtime:
                is_containerized = True
                runtime = cgroup_runtime

        # Check for Kubernetes
        if Path("/var/run/secrets/kubernetes.io").exists():  # pragma: no cover
            is_containerized = True
            runtime = "kubernetes"

    except (OSError, PermissionError):  # pragma: no cover
        pass

    if not is_containerized:
        return None

    # Try to get image info from environment
    image = os.environ.get("CONTAINER_IMAGE") or os.environ.get("IMAGE_NAME")
    image_digest = os.environ.get("CONTAINER_IMAGE_DIGEST") or os.environ.get(
        "IMAGE_DIGEST"
    )

    return ContainerInfo(
        runtime=runtime,
        container_id=container_id,
        image=image,
        image_digest=image_digest,
        cgroup_path=cgroup_path,
        is_containerized=is_containerized,
    )


def _capture_command_info() -> CommandInfo:
    """Capture command invocation details."""
    argv = tuple(sys.argv)
    entrypoint = sys.argv[0] if sys.argv else ""

    return CommandInfo(
        argv=argv,
        working_dir=str(Path.cwd()),
        entrypoint=entrypoint,
        executable=sys.executable,
    )


def capture_environment(
    *,
    working_dir: Path | None = None,
    include_packages: bool = True,
    include_git_diff: bool = True,
) -> EnvironmentCapture:
    """Capture complete environment information.

    Args:
        working_dir: Working directory for git operations. Defaults to cwd.
        include_packages: Whether to capture installed packages (slower).
        include_git_diff: Whether to capture git diff (may be large).

    Returns:
        EnvironmentCapture containing all collected information.
    """
    # Capture in order of speed (fast first, slow last)
    system = _capture_system_info()
    python = _capture_python_info()
    env_vars = _capture_env_vars()
    command = _capture_command_info()
    container = _capture_container_info()

    # Git operations
    git = capture_git_info(working_dir)
    git_diff = capture_git_diff(working_dir) if include_git_diff and git else None

    # Package capture is slowest
    packages = _capture_packages() if include_packages else ""

    return EnvironmentCapture(
        system=system,
        python=python,
        packages=packages,
        env_vars=env_vars,
        git=git,
        git_diff=git_diff,
        command=command,
        container=container,
        captured_at=SYSTEM_CLOCK.utcnow().isoformat(),
    )


__all__ = [
    "CommandInfo",
    "ContainerInfo",
    "EnvironmentCapture",
    "GitInfo",
    "PythonInfo",
    "SystemInfo",
    "capture_environment",
]
