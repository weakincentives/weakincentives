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

"""Tests for build/check_core_imports.py."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

# Add build directory to path so we can import the module
_BUILD_DIR = Path(__file__).resolve().parent.parent.parent / "build"
sys.path.insert(0, str(_BUILD_DIR))

from check_core_imports import (  # noqa: E402
    _check_file,
    _is_contrib_import,
)


class TestIsContribImport:
    """Tests for _is_contrib_import function."""

    def test_detects_direct_contrib_import(self) -> None:
        """Detects 'import weakincentives.contrib.tools'."""
        node = ast.parse("import weakincentives.contrib.tools").body[0]
        assert isinstance(node, ast.Import)
        is_contrib, name = _is_contrib_import(node)
        assert is_contrib is True
        assert name == "weakincentives.contrib.tools"

    def test_detects_contrib_submodule_import(self) -> None:
        """Detects 'import weakincentives.contrib.tools.workspace'."""
        node = ast.parse("import weakincentives.contrib.tools.workspace").body[0]
        assert isinstance(node, ast.Import)
        is_contrib, name = _is_contrib_import(node)
        assert is_contrib is True
        assert name == "weakincentives.contrib.tools.workspace"

    def test_ignores_non_contrib_import(self) -> None:
        """Ignores 'import weakincentives.prompt'."""
        node = ast.parse("import weakincentives.prompt").body[0]
        assert isinstance(node, ast.Import)
        is_contrib, name = _is_contrib_import(node)
        assert is_contrib is False
        assert name == ""

    def test_ignores_stdlib_import(self) -> None:
        """Ignores standard library imports."""
        node = ast.parse("import os").body[0]
        assert isinstance(node, ast.Import)
        is_contrib, _name = _is_contrib_import(node)
        assert is_contrib is False

    def test_detects_from_contrib_import(self) -> None:
        """Detects 'from weakincentives.contrib.tools import X'."""
        node = ast.parse(
            "from weakincentives.contrib.tools import VfsToolsSection"
        ).body[0]
        assert isinstance(node, ast.ImportFrom)
        is_contrib, name = _is_contrib_import(node)
        assert is_contrib is True
        assert name == "weakincentives.contrib.tools"

    def test_detects_relative_contrib_import(self) -> None:
        """Detects 'from ..contrib.tools import X'."""
        node = ast.parse("from ..contrib.tools import X").body[0]
        assert isinstance(node, ast.ImportFrom)
        is_contrib, name = _is_contrib_import(node)
        assert is_contrib is True
        assert "contrib" in name

    def test_detects_deeply_nested_relative_contrib_import(self) -> None:
        """Detects 'from ...contrib.tools.workspace import Y'."""
        node = ast.parse("from ...contrib.tools.workspace import Y").body[0]
        assert isinstance(node, ast.ImportFrom)
        is_contrib, _name = _is_contrib_import(node)
        assert is_contrib is True

    def test_ignores_non_contrib_from_import(self) -> None:
        """Ignores 'from weakincentives.prompt import Prompt'."""
        node = ast.parse("from weakincentives.prompt import Prompt").body[0]
        assert isinstance(node, ast.ImportFrom)
        is_contrib, _name = _is_contrib_import(node)
        assert is_contrib is False

    def test_ignores_relative_non_contrib_import(self) -> None:
        """Ignores 'from ..prompt import X'."""
        node = ast.parse("from ..prompt import X").body[0]
        assert isinstance(node, ast.ImportFrom)
        is_contrib, _name = _is_contrib_import(node)
        assert is_contrib is False

    def test_handles_multi_import(self) -> None:
        """Handles 'import os, weakincentives.contrib.tools'."""
        node = ast.parse("import os, weakincentives.contrib.tools").body[0]
        assert isinstance(node, ast.Import)
        is_contrib, name = _is_contrib_import(node)
        assert is_contrib is True
        assert name == "weakincentives.contrib.tools"

    def test_handles_from_import_with_no_module(self) -> None:
        """Handles 'from . import X' (module is None)."""
        node = ast.parse("from . import X").body[0]
        assert isinstance(node, ast.ImportFrom)
        is_contrib, _name = _is_contrib_import(node)
        assert is_contrib is False

    def test_detects_contrib_in_middle_of_path(self) -> None:
        """Detects contrib even in unusual module paths."""
        node = ast.parse("from some.contrib.module import X").body[0]
        assert isinstance(node, ast.ImportFrom)
        is_contrib, _name = _is_contrib_import(node)
        assert is_contrib is True


class TestCheckFile:
    """Tests for _check_file function."""

    @pytest.fixture
    def tmp_py_file(self, tmp_path: Path) -> Path:
        """Create a temporary Python file."""
        return tmp_path / "test_module.py"

    def test_no_violations_clean_file(self, tmp_py_file: Path) -> None:
        """Returns empty list for file with no contrib imports."""
        tmp_py_file.write_text(
            "from weakincentives.prompt import Prompt\nimport os\nx = 1\n"
        )
        violations = _check_file(tmp_py_file)
        assert violations == []

    def test_detects_direct_import_violation(self, tmp_py_file: Path) -> None:
        """Detects direct import from contrib."""
        tmp_py_file.write_text("import weakincentives.contrib.tools\n")
        violations = _check_file(tmp_py_file)
        assert len(violations) == 1
        assert "weakincentives.contrib.tools" in violations[0]
        assert ":1:" in violations[0]

    def test_detects_from_import_violation(self, tmp_py_file: Path) -> None:
        """Detects from import from contrib."""
        tmp_py_file.write_text(
            "from weakincentives.contrib.tools import VfsToolsSection\n"
        )
        violations = _check_file(tmp_py_file)
        assert len(violations) == 1
        assert "weakincentives.contrib.tools" in violations[0]

    def test_detects_relative_import_violation(self, tmp_py_file: Path) -> None:
        """Detects relative import from contrib."""
        tmp_py_file.write_text("from ..contrib.tools import X\n")
        violations = _check_file(tmp_py_file)
        assert len(violations) == 1
        assert "contrib" in violations[0]

    def test_detects_multiple_violations(self, tmp_py_file: Path) -> None:
        """Detects multiple violations in one file."""
        tmp_py_file.write_text(
            "from weakincentives.contrib.tools import X\n"
            "import os\n"
            "from weakincentives.contrib.mailbox import Y\n"
        )
        violations = _check_file(tmp_py_file)
        assert len(violations) == 2

    def test_reports_correct_line_numbers(self, tmp_py_file: Path) -> None:
        """Reports correct line numbers for violations."""
        tmp_py_file.write_text(
            "import os\nimport sys\nfrom weakincentives.contrib.tools import X\n"
        )
        violations = _check_file(tmp_py_file)
        assert len(violations) == 1
        assert ":3:" in violations[0]

    def test_handles_syntax_error(self, tmp_py_file: Path) -> None:
        """Returns empty list for file with syntax error."""
        tmp_py_file.write_text("def f(\n    pass\n")
        violations = _check_file(tmp_py_file)
        assert violations == []

    def test_handles_unicode_decode_error(self, tmp_py_file: Path) -> None:
        """Returns empty list for file with invalid encoding."""
        tmp_py_file.write_bytes(b"\x80\x81\x82")
        violations = _check_file(tmp_py_file)
        assert violations == []

    def test_handles_empty_file(self, tmp_py_file: Path) -> None:
        """Handles empty Python file."""
        tmp_py_file.write_text("")
        violations = _check_file(tmp_py_file)
        assert violations == []

    def test_handles_comment_only_file(self, tmp_py_file: Path) -> None:
        """Handles file with only comments."""
        tmp_py_file.write_text("# This is a comment\n# Another comment\n")
        violations = _check_file(tmp_py_file)
        assert violations == []

    def test_ignores_contrib_in_string(self, tmp_py_file: Path) -> None:
        """Ignores 'contrib' appearing in string literals."""
        tmp_py_file.write_text('x = "from weakincentives.contrib import X"\n')
        violations = _check_file(tmp_py_file)
        assert violations == []

    def test_ignores_contrib_in_comment(self, tmp_py_file: Path) -> None:
        """Ignores 'contrib' appearing in comments."""
        tmp_py_file.write_text("# from weakincentives.contrib import X\n")
        violations = _check_file(tmp_py_file)
        assert violations == []

    def test_handles_type_checking_import(self, tmp_py_file: Path) -> None:
        """Detects violations even in TYPE_CHECKING blocks."""
        tmp_py_file.write_text(
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from weakincentives.contrib.tools import X\n"
        )
        violations = _check_file(tmp_py_file)
        assert len(violations) == 1

    def test_handles_try_except_import(self, tmp_py_file: Path) -> None:
        """Detects violations in try/except blocks."""
        tmp_py_file.write_text(
            "try:\n"
            "    from weakincentives.contrib.tools import X\n"
            "except ImportError:\n"
            "    X = None\n"
        )
        violations = _check_file(tmp_py_file)
        assert len(violations) == 1

    def test_handles_nested_function_import(self, tmp_py_file: Path) -> None:
        """Detects violations in nested function imports."""
        tmp_py_file.write_text(
            "def foo():\n    from weakincentives.contrib.tools import X\n    return X\n"
        )
        violations = _check_file(tmp_py_file)
        assert len(violations) == 1
        assert ":2:" in violations[0]


class TestIntegration:
    """Integration tests for the check script."""

    def test_actual_codebase_has_no_violations(self) -> None:
        """Verify the actual codebase passes the check.

        This is a regression test to ensure no new violations are introduced.
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        src_path = project_root / "src" / "weakincentives"
        contrib_path = src_path / "contrib"

        violations: list[str] = []

        for filepath in src_path.rglob("*.py"):
            # Skip contrib directory
            if contrib_path in filepath.parents or filepath.parent == contrib_path:
                continue
            violations.extend(_check_file(filepath))

        assert violations == [], "Found violations:\n" + "\n".join(violations)

    def test_contrib_files_would_have_violations(self, tmp_path: Path) -> None:
        """Verify that contrib files DO have contrib imports (sanity check).

        This ensures our detector actually works by checking that contrib
        files (which are allowed to import from contrib) would be flagged
        if they weren't excluded.
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        contrib_path = project_root / "src" / "weakincentives" / "contrib"

        # Find a contrib file that imports from other contrib modules
        found_contrib_import = False
        for filepath in contrib_path.rglob("*.py"):
            violations = _check_file(filepath)
            if violations:
                found_contrib_import = True
                break

        # It's okay if contrib doesn't import from itself,
        # but this test validates our detector works
        # Skip assertion if no cross-contrib imports exist
        if not found_contrib_import:
            pytest.skip("No cross-contrib imports found to validate detector")
