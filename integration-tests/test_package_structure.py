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
def project_root() -> Path:
    """Return the project root directory."""
    # integration-tests/test_package_structure.py -> parent.parent = project root
    return Path(__file__).parent.parent


@pytest.fixture(scope="module")
def wheel_path(tmp_path_factory: pytest.TempPathFactory, project_root: Path) -> Path:
    """Build a wheel and return its path.

    This is module-scoped to avoid rebuilding for each test.
    Uses project_root to ensure we build from the correct directory,
    even when tests run from a different working directory (e.g., mutants/).
    """
    output_dir = tmp_path_factory.mktemp("dist")

    # Build wheel using uv/pip from the project root
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(output_dir),
            str(project_root),
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

    def test_guides_bundled(
        self,
        wheel_contents: frozenset[str],
        project_root: Path,
    ) -> None:
        """All guide files from guides/ directory must be bundled in the wheel."""
        # Get all .md files from source guides/ directory
        source_guides = {f.name for f in (project_root / "guides").glob("*.md")}
        assert len(source_guides) > 0, (
            "No guide files found in source guides/ directory"
        )

        # Get all .md files bundled in the wheel
        bundled_guides = {
            Path(f).name
            for f in wheel_contents
            if f.startswith("weakincentives/docs/guides/") and f.endswith(".md")
        }

        # Every source guide must be bundled
        missing = source_guides - bundled_guides
        assert not missing, f"Guide files missing from wheel: {sorted(missing)}"

        # Bundled guides should match source (no extra files)
        extra = bundled_guides - source_guides
        assert not extra, f"Unexpected guide files in wheel: {sorted(extra)}"

    def test_all_specs_bundled(
        self,
        wheel_contents: frozenset[str],
        project_root: Path,
    ) -> None:
        """All spec files from specs/ directory must be bundled in the wheel."""
        # Get all .md files from source specs/ directory
        source_specs = {f.name for f in (project_root / "specs").glob("*.md")}
        assert len(source_specs) > 0, "No spec files found in source specs/ directory"

        # Get all .md files bundled in the wheel
        bundled_specs = {
            Path(f).name
            for f in wheel_contents
            if f.startswith("weakincentives/docs/specs/") and f.endswith(".md")
        }

        # Every source spec must be bundled
        missing = source_specs - bundled_specs
        assert not missing, f"Spec files missing from wheel: {sorted(missing)}"

        # Bundled specs should match source (no extra files)
        extra = bundled_specs - source_specs
        assert not extra, f"Unexpected spec files in wheel: {sorted(extra)}"


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
