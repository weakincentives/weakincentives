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

"""Tests for architecture checkers."""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.verify._types import CheckContext, Severity
from weakincentives.verify.checkers.architecture import (
    CoreContribSeparationChecker,
    LayerViolationsChecker,
)


class TestLayerViolationsChecker:
    """Tests for LayerViolationsChecker."""

    def test_checker_properties(self) -> None:
        """Checker has required properties."""
        checker = LayerViolationsChecker()
        assert checker.name == "layer_violations"
        assert checker.category == "architecture"
        assert len(checker.description) > 0

    def test_no_package_dir(self, tmp_path: Path) -> None:
        """Error when package directory doesn't exist."""
        (tmp_path / "src").mkdir()
        ctx = CheckContext.from_project_root(tmp_path)
        checker = LayerViolationsChecker()

        result = checker.check(ctx)

        assert not result.passed
        assert any("not found" in f.message for f in result.findings)

    def test_empty_package(self, tmp_path: Path) -> None:
        """Pass with empty package."""
        pkg_dir = tmp_path / "src" / "weakincentives"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "__init__.py").write_text("")
        ctx = CheckContext.from_project_root(tmp_path)
        checker = LayerViolationsChecker()

        result = checker.check(ctx)

        assert result.passed

    def test_valid_layer_import(self, tmp_path: Path) -> None:
        """Pass with valid layer imports (low to high)."""
        pkg_dir = tmp_path / "src" / "weakincentives"
        (pkg_dir / "runtime").mkdir(parents=True)
        (pkg_dir / "types").mkdir(parents=True)
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "types" / "__init__.py").write_text("")
        # runtime (core) importing from types (foundation) is valid
        (pkg_dir / "runtime" / "__init__.py").write_text(
            "from weakincentives.types import SomeType"
        )
        ctx = CheckContext.from_project_root(tmp_path)
        checker = LayerViolationsChecker()

        result = checker.check(ctx)

        assert result.passed

    def test_invalid_layer_import(self, tmp_path: Path) -> None:
        """Fail with invalid layer import (high to low)."""
        pkg_dir = tmp_path / "src" / "weakincentives"
        (pkg_dir / "types").mkdir(parents=True)
        (pkg_dir / "contrib").mkdir(parents=True)
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "contrib" / "__init__.py").write_text("")
        # types (foundation) importing from contrib (high_level) is invalid
        (pkg_dir / "types" / "__init__.py").write_text(
            "from weakincentives.contrib.tools import Plan"
        )
        ctx = CheckContext.from_project_root(tmp_path)
        checker = LayerViolationsChecker()

        result = checker.check(ctx)

        assert not result.passed
        assert any("LAYER" in f.message.upper() or "layer" in f.message for f in result.findings)


class TestCoreContribSeparationChecker:
    """Tests for CoreContribSeparationChecker."""

    def test_checker_properties(self) -> None:
        """Checker has required properties."""
        checker = CoreContribSeparationChecker()
        assert checker.name == "core_contrib_separation"
        assert checker.category == "architecture"
        assert len(checker.description) > 0

    def test_no_package_dir(self, tmp_path: Path) -> None:
        """Error when package directory doesn't exist."""
        (tmp_path / "src").mkdir()
        ctx = CheckContext.from_project_root(tmp_path)
        checker = CoreContribSeparationChecker()

        result = checker.check(ctx)

        assert not result.passed

    def test_empty_package(self, tmp_path: Path) -> None:
        """Pass with empty package."""
        pkg_dir = tmp_path / "src" / "weakincentives"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "__init__.py").write_text("")
        ctx = CheckContext.from_project_root(tmp_path)
        checker = CoreContribSeparationChecker()

        result = checker.check(ctx)

        assert result.passed

    def test_core_importing_contrib(self, tmp_path: Path) -> None:
        """Fail when core imports from contrib."""
        pkg_dir = tmp_path / "src" / "weakincentives"
        (pkg_dir / "runtime").mkdir(parents=True)
        (pkg_dir / "contrib" / "tools").mkdir(parents=True)
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "contrib" / "__init__.py").write_text("")
        (pkg_dir / "contrib" / "tools" / "__init__.py").write_text("")
        # Core module importing from contrib
        (pkg_dir / "runtime" / "__init__.py").write_text(
            "from weakincentives.contrib.tools import Plan"
        )
        ctx = CheckContext.from_project_root(tmp_path)
        checker = CoreContribSeparationChecker()

        result = checker.check(ctx)

        assert not result.passed
        assert any("contrib" in f.message.lower() for f in result.findings)

    def test_contrib_importing_contrib(self, tmp_path: Path) -> None:
        """Pass when contrib imports from contrib."""
        pkg_dir = tmp_path / "src" / "weakincentives"
        (pkg_dir / "contrib" / "tools").mkdir(parents=True)
        (pkg_dir / "contrib" / "mailbox").mkdir(parents=True)
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "contrib" / "__init__.py").write_text("")
        (pkg_dir / "contrib" / "tools" / "__init__.py").write_text("")
        # Contrib importing from other contrib is fine
        (pkg_dir / "contrib" / "mailbox" / "__init__.py").write_text(
            "from weakincentives.contrib.tools import Plan"
        )
        ctx = CheckContext.from_project_root(tmp_path)
        checker = CoreContribSeparationChecker()

        result = checker.check(ctx)

        assert result.passed
