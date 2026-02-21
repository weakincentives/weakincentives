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

"""Tests for ArchitectureChecker and BannedTimeImportsChecker."""

from __future__ import annotations

import tempfile
from pathlib import Path

from toolchain.checkers.architecture import ArchitectureChecker
from toolchain.checkers.banned_time_imports import BannedTimeImportsChecker


class TestArchitectureChecker:
    """Tests for ArchitectureChecker."""

    def test_name_and_description(self) -> None:
        checker = ArchitectureChecker()
        assert checker.name == "architecture"
        assert "core/contrib" in checker.description.lower()

    def test_passes_on_valid_structure(self) -> None:
        # Use actual src directory
        root = Path(__file__).parents[2]
        checker = ArchitectureChecker(src_dir=root / "src")
        result = checker.run()
        # Should pass if codebase is properly structured
        assert result.name == "architecture"
        assert result.status == "passed", [d.message for d in result.diagnostics]

    def test_fails_on_missing_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checker = ArchitectureChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "failed"
            assert any("not found" in d.message.lower() for d in result.diagnostics)

    def test_detects_core_importing_contrib(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            # Create a runtime (core layer) module that imports contrib (high-level)
            runtime = pkg / "runtime"
            runtime.mkdir()
            (runtime / "__init__.py").write_text("")
            (runtime / "core.py").write_text(
                "from weakincentives.contrib import something"
            )

            # Create contrib directory
            contrib = pkg / "contrib"
            contrib.mkdir()
            (contrib / "__init__.py").write_text("")

            checker = ArchitectureChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "failed"
            assert any("contrib" in d.message.lower() for d in result.diagnostics)
            # Verify that the import statement is included in the diagnostic
            assert any(
                "Import: from weakincentives.contrib import something" in d.message
                for d in result.diagnostics
            )

    def test_allows_contrib_importing_contrib(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            # Create contrib module that imports other contrib
            contrib = pkg / "contrib"
            contrib.mkdir()
            (contrib / "__init__.py").write_text("")
            (contrib / "tools.py").write_text("from weakincentives.contrib import other")

            checker = ArchitectureChecker(src_dir=Path(tmpdir))
            result = checker.run()
            # Should pass - contrib can import contrib
            assert not any("layer violation" in d.message.lower() for d in result.diagnostics)

    def test_handles_syntax_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            # Put the broken file in a known-layer package so it's checked
            runtime = pkg / "runtime"
            runtime.mkdir()
            (runtime / "__init__.py").write_text("")
            (runtime / "broken.py").write_text("def broken(")  # Syntax error

            checker = ArchitectureChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "failed"
            assert any("syntax" in d.message.lower() for d in result.diagnostics)

    def test_skips_pycache_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            # Create a __pycache__ directory with a .py file (glob finds *.py)
            pycache = pkg / "__pycache__"
            pycache.mkdir()
            # This .py file would cause issues if not skipped
            (pycache / "bad.py").write_text("from weakincentives.contrib import x")

            checker = ArchitectureChecker(src_dir=Path(tmpdir))
            result = checker.run()
            # Should pass - pycache is skipped
            assert result.status == "passed"

    # ------------------------------------------------------------------
    # Four-layer model tests
    # ------------------------------------------------------------------

    def test_detects_core_importing_adapters(self) -> None:
        """Core (layer 2) must not import Adapters (layer 3) at runtime."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            runtime = pkg / "runtime"
            runtime.mkdir()
            (runtime / "__init__.py").write_text("")
            (runtime / "loop.py").write_text(
                "from weakincentives.adapters.core import PromptEvaluationError"
            )

            adapters = pkg / "adapters"
            adapters.mkdir()
            (adapters / "__init__.py").write_text("")
            (adapters / "core.py").write_text("")

            checker = ArchitectureChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "failed"
            assert any("layer violation" in d.message.lower() for d in result.diagnostics)

    def test_allows_type_checking_imports_across_layers(self) -> None:
        """TYPE_CHECKING imports are exempt from layer checks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            runtime = pkg / "runtime"
            runtime.mkdir()
            (runtime / "__init__.py").write_text("")
            (runtime / "loop.py").write_text(
                "from __future__ import annotations\n"
                "from typing import TYPE_CHECKING\n"
                "if TYPE_CHECKING:\n"
                "    from weakincentives.adapters.core import ProviderAdapter\n"
            )

            adapters = pkg / "adapters"
            adapters.mkdir()
            (adapters / "__init__.py").write_text("")
            (adapters / "core.py").write_text("")

            checker = ArchitectureChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "passed"

    def test_allows_same_layer_imports(self) -> None:
        """Core modules can import from other core modules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            runtime = pkg / "runtime"
            runtime.mkdir()
            (runtime / "__init__.py").write_text("")
            (runtime / "loop.py").write_text(
                "from weakincentives.prompt import Prompt"
            )

            prompt = pkg / "prompt"
            prompt.mkdir()
            (prompt / "__init__.py").write_text("")

            checker = ArchitectureChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "passed"

    def test_allows_lower_layer_imports(self) -> None:
        """Core modules can import from Foundation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            runtime = pkg / "runtime"
            runtime.mkdir()
            (runtime / "__init__.py").write_text("")
            (runtime / "loop.py").write_text(
                "from weakincentives.errors import WinkError"
            )

            checker = ArchitectureChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "passed"

    def test_detects_foundation_importing_core(self) -> None:
        """Foundation (layer 1) must not import Core (layer 2)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            types_dir = pkg / "types"
            types_dir.mkdir()
            (types_dir / "__init__.py").write_text(
                "from weakincentives.runtime import Session"
            )

            checker = ArchitectureChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "failed"
            assert any("layer violation" in d.message.lower() for d in result.diagnostics)

    def test_allows_adapters_importing_core(self) -> None:
        """Adapters (layer 3) can import from Core (layer 2)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            adapters = pkg / "adapters"
            adapters.mkdir()
            (adapters / "__init__.py").write_text("")
            (adapters / "sdk.py").write_text(
                "from weakincentives.runtime import Session"
            )

            runtime = pkg / "runtime"
            runtime.mkdir()
            (runtime / "__init__.py").write_text("")

            checker = ArchitectureChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "passed"

    def test_detects_adapters_importing_high_level(self) -> None:
        """Adapters (layer 3) must not import High-level (layer 4)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            adapters = pkg / "adapters"
            adapters.mkdir()
            (adapters / "__init__.py").write_text("")
            (adapters / "sdk.py").write_text(
                "from weakincentives.contrib import tool"
            )

            contrib = pkg / "contrib"
            contrib.mkdir()
            (contrib / "__init__.py").write_text("")

            checker = ArchitectureChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "failed"
            assert any("layer violation" in d.message.lower() for d in result.diagnostics)

    def test_real_codebase_passes(self) -> None:
        """The actual codebase must pass layer checks after violation fixes."""
        root = Path(__file__).parents[2]
        checker = ArchitectureChecker(src_dir=root / "src")
        result = checker.run()
        assert result.status == "passed", [d.message for d in result.diagnostics]


class TestBannedTimeImportsChecker:
    """Tests for BannedTimeImportsChecker."""

    def test_name_and_description(self) -> None:
        checker = BannedTimeImportsChecker()
        assert checker.name == "banned-time-imports"
        assert "clock" in checker.description.lower()

    def test_passes_on_real_codebase(self) -> None:
        root = Path(__file__).parents[2]
        checker = BannedTimeImportsChecker(src_dir=root / "src")
        result = checker.run()
        assert result.status == "passed", [d.message for d in result.diagnostics]

    def test_fails_on_missing_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checker = BannedTimeImportsChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "failed"
            assert any("not found" in d.message.lower() for d in result.diagnostics)

    def test_detects_import_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "bad.py").write_text("import time\n\nx = time.monotonic()\n")

            checker = BannedTimeImportsChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "failed"
            assert len(result.diagnostics) == 1
            assert "import time" in result.diagnostics[0].message
            assert result.diagnostics[0].location is not None
            assert result.diagnostics[0].location.line == 1

    def test_detects_from_time_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "bad.py").write_text("from time import sleep\n")

            checker = BannedTimeImportsChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "failed"
            assert any("from time import" in d.message for d in result.diagnostics)

    def test_allows_clock_py(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "clock.py").write_text("import time as _time\n")

            checker = BannedTimeImportsChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "passed"

    def test_skips_pycache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            pycache = pkg / "__pycache__"
            pycache.mkdir()
            (pycache / "bad.py").write_text("import time\n")

            checker = BannedTimeImportsChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "passed"

    def test_handles_unreadable_file(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "bad.py").write_text("import time\n")

            original_read = Path.read_text

            def mock_read(self: Path, *args: object, **kwargs: object) -> str:
                if self.name == "bad.py":
                    raise OSError("Permission denied")
                return original_read(self, *args, **kwargs)  # type: ignore[arg-type]

            with patch.object(Path, "read_text", mock_read):
                checker = BannedTimeImportsChecker(src_dir=Path(tmpdir))
                result = checker.run()
            # Should pass - unreadable files are skipped gracefully
            assert result.status == "passed"

    def test_multiple_violations_in_one_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir) / "weakincentives"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "bad.py").write_text("import time\nfrom time import sleep\n")

            checker = BannedTimeImportsChecker(src_dir=Path(tmpdir))
            result = checker.run()
            assert result.status == "failed"
            assert len(result.diagnostics) == 2
