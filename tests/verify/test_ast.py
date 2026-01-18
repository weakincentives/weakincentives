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

"""Tests for AST utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.verify._ast import (
    extract_imports,
    get_top_level_package,
    module_to_path,
    patch_ast_for_legacy_tools,
    path_to_module,
)


class TestExtractImports:
    """Tests for extract_imports function."""

    def test_simple_import(self) -> None:
        """Extract simple import statement."""
        source = "import os"
        imports = extract_imports(source, "test.module")
        assert len(imports) == 1
        assert imports[0].imported_from == "os"
        assert imports[0].items == ()
        assert imports[0].is_relative is False

    def test_from_import(self) -> None:
        """Extract from-import statement."""
        source = "from os.path import join, exists"
        imports = extract_imports(source, "test.module")
        assert len(imports) == 1
        assert imports[0].imported_from == "os.path"
        assert imports[0].items == ("join", "exists")
        assert imports[0].is_relative is False

    def test_relative_import(self) -> None:
        """Extract relative import statement."""
        source = "from .sibling import something"
        imports = extract_imports(source, "package.submodule")
        assert len(imports) == 1
        assert imports[0].is_relative is True
        assert "sibling" in imports[0].imported_from

    def test_multiple_imports(self) -> None:
        """Extract multiple import statements."""
        source = """\
import os
import sys
from pathlib import Path
from typing import Any
"""
        imports = extract_imports(source, "test.module")
        assert len(imports) == 4

    def test_syntax_error(self) -> None:
        """Raise SyntaxError for invalid code."""
        source = "import ("
        with pytest.raises(SyntaxError):
            extract_imports(source, "test.module")


class TestPathToModule:
    """Tests for path_to_module function."""

    def test_simple_module(self, tmp_path: Path) -> None:
        """Convert simple path to module."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        module_path = src_dir / "mypackage" / "submodule.py"

        result = path_to_module(module_path, src_dir)
        assert result == "mypackage.submodule"

    def test_init_module(self, tmp_path: Path) -> None:
        """Convert __init__.py to package module."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        init_path = src_dir / "mypackage" / "__init__.py"

        result = path_to_module(init_path, src_dir)
        assert result == "mypackage"

    def test_nested_module(self, tmp_path: Path) -> None:
        """Convert deeply nested path."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        nested_path = src_dir / "pkg" / "sub" / "deep" / "module.py"

        result = path_to_module(nested_path, src_dir)
        assert result == "pkg.sub.deep.module"


class TestModuleToPath:
    """Tests for module_to_path function."""

    def test_package_module(self, tmp_path: Path) -> None:
        """Convert package module to path."""
        src_dir = tmp_path / "src"
        pkg_dir = src_dir / "weakincentives" / "subpkg"
        pkg_dir.mkdir(parents=True)
        init_file = pkg_dir / "__init__.py"
        init_file.write_text("")

        result = module_to_path("weakincentives.subpkg", src_dir)
        assert result == init_file

    def test_file_module(self, tmp_path: Path) -> None:
        """Convert file module to path."""
        src_dir = tmp_path / "src"
        pkg_dir = src_dir / "weakincentives"
        pkg_dir.mkdir(parents=True)
        module_file = pkg_dir / "module.py"
        module_file.write_text("")

        result = module_to_path("weakincentives.module", src_dir)
        assert result == module_file

    def test_nonexistent_module(self, tmp_path: Path) -> None:
        """Return None for nonexistent module."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        result = module_to_path("nonexistent.module", src_dir)
        assert result is None


class TestGetTopLevelPackage:
    """Tests for get_top_level_package function."""

    def test_subpackage(self) -> None:
        """Get top-level package from subpackage."""
        result = get_top_level_package("weakincentives.contrib.tools")
        assert result == "contrib"

    def test_direct_submodule(self) -> None:
        """Get top-level package from direct submodule."""
        result = get_top_level_package("weakincentives.runtime")
        assert result == "runtime"

    def test_non_matching_root(self) -> None:
        """Return None for non-matching root package."""
        result = get_top_level_package("otherpackage.module")
        assert result is None

    def test_root_only(self) -> None:
        """Return None for root package only."""
        result = get_top_level_package("weakincentives")
        assert result is None


class TestPatchAstForLegacyTools:
    """Tests for patch_ast_for_legacy_tools function."""

    def test_patch_succeeds(self) -> None:
        """Patching doesn't raise errors."""
        # Just verify it doesn't crash
        patch_ast_for_legacy_tools()
