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

"""Git utilities for the verification toolbox."""

from __future__ import annotations

import subprocess
from pathlib import Path


def is_git_repo(path: Path) -> bool:
    """Check if the path is inside a git repository.

    Args:
        path: The path to check.

    Returns:
        True if inside a git repository, False otherwise.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def tracked_files(
    root: Path,
    *,
    pattern: str = "*",
    exclude_parts: frozenset[str] = frozenset(),
    exclude_prefixes: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    """Get git-tracked files matching a pattern.

    Args:
        root: The repository root directory.
        pattern: Glob pattern for filtering files (e.g., "*.md").
        exclude_parts: Directory parts to exclude (e.g., {"test-repositories"}).
        exclude_prefixes: Path prefixes to exclude.

    Returns:
        A tuple of absolute paths to matching tracked files.

    Raises:
        RuntimeError: If git ls-files fails.
    """
    result = subprocess.run(
        ["git", "ls-files", pattern],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git ls-files failed: {message}")

    files: list[Path] = []
    for line in result.stdout.splitlines():
        if not line:
            continue

        path = Path(line)

        # Check exclusions by directory parts
        if exclude_parts.intersection(path.parts):
            continue

        # Check exclusions by prefix
        if any(path.is_relative_to(prefix) for prefix in exclude_prefixes):
            continue

        # Resolve to absolute path and verify existence
        candidate = root / path
        if candidate.exists():
            files.append(candidate)

    return tuple(sorted(files))


def get_repo_root(path: Path) -> Path | None:
    """Get the root directory of the git repository.

    Args:
        path: A path inside the repository.

    Returns:
        The repository root, or None if not in a git repository.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return None

    return Path(result.stdout.strip())
