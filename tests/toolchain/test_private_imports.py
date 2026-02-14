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

"""Tests for private module import checker."""

from __future__ import annotations

import tempfile
from pathlib import Path

from toolchain.checkers.private_imports import (
    PrivateImportChecker,
    _find_private_segment,
    _is_private,
)


class TestIsPrivate:
    """Tests for _is_private helper."""

    def test_single_underscore_prefix_is_private(self) -> None:
        assert _is_private("_foo") is True

    def test_dunder_is_not_private(self) -> None:
        assert _is_private("__init__") is False
        assert _is_private("__future__") is False
        assert _is_private("__pycache__") is False

    def test_no_underscore_is_not_private(self) -> None:
        assert _is_private("foo") is False
        assert _is_private("section") is False

    def test_double_underscore_prefix_only_is_not_private(self) -> None:
        assert _is_private("__foo") is False


class TestFindPrivateSegment:
    """Tests for _find_private_segment helper."""

    def test_no_private_segment(self) -> None:
        assert _find_private_segment(["weakincentives", "prompt", "section"]) is None

    def test_finds_first_private_segment(self) -> None:
        assert (
            _find_private_segment(["weakincentives", "serde", "_scope"]) == 2
        )

    def test_finds_first_when_multiple(self) -> None:
        parts = ["weakincentives", "adapters", "_shared", "_types"]
        assert _find_private_segment(parts) == 2

    def test_ignores_dunder_segments(self) -> None:
        parts = ["weakincentives", "__init__", "section"]
        assert _find_private_segment(parts) is None

    def test_private_at_start(self) -> None:
        assert _find_private_segment(["_internal"]) == 0


class TestPrivateImportChecker:
    """Tests for PrivateImportChecker."""

    def test_name_and_description(self) -> None:
        checker = PrivateImportChecker()
        assert checker.name == "private-imports"
        assert "private" in checker.description.lower()

    def test_passes_on_valid_codebase(self) -> None:
        root = Path(__file__).parents[2]
        checker = PrivateImportChecker(src_dir=root / "src")
        result = checker.run()
        assert result.status == "passed"
        assert result.name == "private-imports"

    def test_fails_on_missing_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checker = PrivateImportChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "failed"
            assert any("not found" in d.message.lower() for d in result.diagnostics)

    def test_detects_cross_package_private_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            # Create serde package with private module
            serde = pkg / "serde"
            serde.mkdir()
            (serde / "__init__.py").write_text("")
            (serde / "_scope.py").write_text("class SerdeScope: pass")

            # Create prompt package importing from serde's private module
            prompt = pkg / "prompt"
            prompt.mkdir()
            (prompt / "__init__.py").write_text("")
            (prompt / "output.py").write_text(
                "from weakincentives.serde._scope import SerdeScope"
            )

            checker = PrivateImportChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "failed"
            assert len(result.diagnostics) == 1
            diag = result.diagnostics[0]
            assert "weakincentives.serde._scope" in diag.message
            assert "Import:" in diag.message

    def test_allows_within_package_private_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            # Create serde package with private module
            serde = pkg / "serde"
            serde.mkdir()
            (serde / "__init__.py").write_text(
                "from ._scope import SerdeScope"
            )
            (serde / "_scope.py").write_text("class SerdeScope: pass")
            (serde / "parse.py").write_text(
                "from weakincentives.serde._scope import SerdeScope"
            )

            checker = PrivateImportChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "passed"

    def test_allows_adapter_shared_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            # Create adapters/_shared pattern
            adapters = pkg / "adapters"
            adapters.mkdir()
            (adapters / "__init__.py").write_text("")
            shared = adapters / "_shared"
            shared.mkdir()
            (shared / "__init__.py").write_text("")
            (shared / "_bridge.py").write_text("class Bridge: pass")

            # Adapter sub-package importing from _shared
            claude = adapters / "claude_sdk"
            claude.mkdir()
            (claude / "__init__.py").write_text("")
            (claude / "adapter.py").write_text(
                "from weakincentives.adapters._shared._bridge import Bridge"
            )

            checker = PrivateImportChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "passed"

    def test_allows_dunder_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            core = pkg / "core.py"
            core.write_text("from __future__ import annotations")

            checker = PrivateImportChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "passed"

    def test_skips_pycache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            pycache = pkg / "__pycache__"
            pycache.mkdir()
            (pycache / "bad.py").write_text(
                "from weakincentives.serde._scope import X"
            )

            checker = PrivateImportChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "passed"

    def test_handles_syntax_errors_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "broken.py").write_text("def broken(")

            checker = PrivateImportChecker(src_dir=Path(tmpdir))
            result = checker.run()
            # Should not crash; syntax errors handled by architecture checker
            assert result.status == "passed"

    def test_diagnostic_has_location(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            serde = pkg / "serde"
            serde.mkdir()
            (serde / "__init__.py").write_text("")
            (serde / "_scope.py").write_text("")

            prompt = pkg / "prompt"
            prompt.mkdir()
            (prompt / "__init__.py").write_text("")
            (prompt / "output.py").write_text(
                "from weakincentives.serde._scope import Scope"
            )

            checker = PrivateImportChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "failed"
            diag = result.diagnostics[0]
            assert diag.location is not None
            assert diag.location.line == 1
            assert "output.py" in diag.location.file

    def test_relative_import_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            serde = pkg / "serde"
            serde.mkdir()
            (serde / "__init__.py").write_text("")
            (serde / "_scope.py").write_text("")

            prompt = pkg / "prompt"
            prompt.mkdir()
            (prompt / "__init__.py").write_text("")
            (prompt / "output.py").write_text(
                "from ..serde._scope import Scope"
            )

            checker = PrivateImportChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "failed"
            assert len(result.diagnostics) == 1

    def test_duration_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            checker = PrivateImportChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.duration_ms >= 0
