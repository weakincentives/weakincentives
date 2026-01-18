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

"""Project path discovery utilities."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root by looking for pyproject.toml.

    Walks up the directory tree from the starting point looking for
    a pyproject.toml file, which indicates the project root.

    Args:
        start: Starting directory for the search. Defaults to cwd.

    Returns:
        The project root directory.

    Raises:
        FileNotFoundError: If no pyproject.toml is found.
    """
    current = (start or Path.cwd()).resolve()

    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # Check the root directory too
    if (current / "pyproject.toml").exists():
        return current

    raise FileNotFoundError(
        f"Could not find pyproject.toml starting from {start or Path.cwd()}"
    )


def get_package_dir(project_root: Path, package_name: str = "weakincentives") -> Path:
    """Get the main package directory.

    Args:
        project_root: The project root directory.
        package_name: The name of the package.

    Returns:
        The path to the package directory.

    Raises:
        FileNotFoundError: If the package directory doesn't exist.
    """
    package_dir = project_root / "src" / package_name
    if not package_dir.is_dir():
        raise FileNotFoundError(f"Package directory not found: {package_dir}")
    return package_dir


def get_tests_dir(project_root: Path) -> Path:
    """Get the tests directory.

    Args:
        project_root: The project root directory.

    Returns:
        The path to the tests directory.

    Raises:
        FileNotFoundError: If the tests directory doesn't exist.
    """
    tests_dir = project_root / "tests"
    if not tests_dir.is_dir():
        raise FileNotFoundError(f"Tests directory not found: {tests_dir}")
    return tests_dir


def get_specs_dir(project_root: Path) -> Path:
    """Get the specs directory.

    Args:
        project_root: The project root directory.

    Returns:
        The path to the specs directory.

    Raises:
        FileNotFoundError: If the specs directory doesn't exist.
    """
    specs_dir = project_root / "specs"
    if not specs_dir.is_dir():
        raise FileNotFoundError(f"Specs directory not found: {specs_dir}")
    return specs_dir


def get_guides_dir(project_root: Path) -> Path:
    """Get the guides directory.

    Args:
        project_root: The project root directory.

    Returns:
        The path to the guides directory.

    Raises:
        FileNotFoundError: If the guides directory doesn't exist.
    """
    guides_dir = project_root / "guides"
    if not guides_dir.is_dir():
        raise FileNotFoundError(f"Guides directory not found: {guides_dir}")
    return guides_dir
