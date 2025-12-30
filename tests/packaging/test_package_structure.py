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

"""Tests for package build structure validation.

These tests ensure that the built wheel contains all expected files,
particularly the bundled documentation for the wink docs CLI command.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def wheel_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a wheel and return its path.

    This is module-scoped to avoid rebuilding for each test.
    """
    output_dir = tmp_path_factory.mktemp("dist")

    # Build wheel using uv/pip
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(output_dir),
            ".",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        pytest.fail(f"Failed to build wheel: {result.stderr}")

    wheels = list(output_dir.glob("*.whl"))
    if not wheels:
        pytest.fail("No wheel file was produced")

    return wheels[0]


@pytest.fixture(scope="module")
def wheel_contents(wheel_path: Path) -> frozenset[str]:
    """Extract file names from the wheel."""
    with zipfile.ZipFile(wheel_path, "r") as whl:
        return frozenset(whl.namelist())


class TestDocsPackaging:
    """Tests that documentation is correctly bundled in the wheel."""

    def test_docs_package_exists(self, wheel_contents: frozenset[str]) -> None:
        """The weakincentives.docs package must exist."""
        assert "weakincentives/docs/__init__.py" in wheel_contents

    def test_llms_md_bundled(self, wheel_contents: frozenset[str]) -> None:
        """llms.md (API reference) must be bundled."""
        assert "weakincentives/docs/llms.md" in wheel_contents

    def test_wink_guide_bundled(self, wheel_contents: frozenset[str]) -> None:
        """WINK_GUIDE.md (usage guide) must be bundled."""
        assert "weakincentives/docs/WINK_GUIDE.md" in wheel_contents

    def test_specs_directory_exists(self, wheel_contents: frozenset[str]) -> None:
        """Specs directory must contain markdown files."""
        specs_files = [
            f for f in wheel_contents if f.startswith("weakincentives/docs/specs/")
        ]
        assert len(specs_files) > 0, "No spec files found in wheel"

    @pytest.mark.parametrize(
        "spec_file",
        [
            "ADAPTERS.md",
            "DATACLASSES.md",
            "DBC.md",
            "PROMPTS.md",
            "SESSIONS.md",
            "TOOLS.md",
            "WINK_DOCS.md",
        ],
    )
    def test_core_specs_bundled(
        self,
        wheel_contents: frozenset[str],
        spec_file: str,
    ) -> None:
        """Core spec files must be bundled."""
        expected_path = f"weakincentives/docs/specs/{spec_file}"
        assert expected_path in wheel_contents, f"{spec_file} not found in wheel"


class TestCorePackaging:
    """Tests that core package structure is correct."""

    def test_cli_module_exists(self, wheel_contents: frozenset[str]) -> None:
        """CLI module must exist for wink command."""
        assert "weakincentives/cli/__init__.py" in wheel_contents
        assert "weakincentives/cli/wink.py" in wheel_contents

    def test_typed_marker_exists(self, wheel_contents: frozenset[str]) -> None:
        """py.typed marker must exist for type checking support."""
        assert "weakincentives/py.typed" in wheel_contents

    def test_prompt_module_exists(self, wheel_contents: frozenset[str]) -> None:
        """Prompt module must exist."""
        assert "weakincentives/prompt/__init__.py" in wheel_contents

    def test_runtime_module_exists(self, wheel_contents: frozenset[str]) -> None:
        """Runtime module must exist."""
        assert "weakincentives/runtime/__init__.py" in wheel_contents

    def test_adapters_module_exists(self, wheel_contents: frozenset[str]) -> None:
        """Adapters module must exist."""
        assert "weakincentives/adapters/__init__.py" in wheel_contents
