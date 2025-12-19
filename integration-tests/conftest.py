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

"""Integration test hooks."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

# When pytest is invoked via the console script the CWD isn't automatically
# added to sys.path, so ensure the repository root (and thus the tests helpers)
# stay importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def isolated_workspace(tmp_path: Path) -> Path:
    """Create an isolated workspace directory with test files.

    This fixture provides a temporary directory with a test README.md file,
    avoiding any operations on the actual repository. Use this instead of
    Path.cwd() for tests that need to read/write files.

    Returns:
        Path to the isolated workspace directory.
    """
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Test Repository\n\n"
        "This is an isolated test workspace for integration tests.\n\n"
        "## Overview\n\n"
        "This file exists to verify file reading capabilities without\n"
        "affecting the actual repository.\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def isolated_project(tmp_path: Path) -> Iterator[Path]:
    """Create an isolated project directory with multiple test files.

    This fixture provides a more complete project structure for tests
    that need multiple files or directories.

    Yields:
        Path to the isolated project directory.
    """
    # Create project structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "# Main module\ndef main():\n    print('Hello')\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "utils.py").write_text(
        "# Utilities\ndef helper():\n    return 42\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# Test Project\n\nA test project for integration tests.\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "test-project"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )

    yield tmp_path
