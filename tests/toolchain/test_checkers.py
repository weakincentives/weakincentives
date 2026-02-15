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

"""Tests for toolchain checker implementations."""

from __future__ import annotations

import tempfile
from pathlib import Path

from toolchain.checkers import (
    create_all_checkers,
    create_architecture_checker,
    create_bandit_checker,
    create_banned_time_imports_checker,
    create_deptry_checker,
    create_docs_checker,
    create_format_checker,
    create_lint_checker,
    create_markdown_checker,
    create_pip_audit_checker,
    create_private_imports_checker,
    create_test_checker,
    create_typecheck_checker,
)
from toolchain.checkers.architecture import ArchitectureChecker
from toolchain.checkers.banned_time_imports import BannedTimeImportsChecker
from toolchain.checkers.docs import DocsChecker
from toolchain.checkers.private_imports import PrivateImportChecker


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


class TestDocsChecker:
    """Tests for DocsChecker."""

    def test_name_and_description(self) -> None:
        checker = DocsChecker()
        assert checker.name == "docs"
        assert "documentation" in checker.description.lower()

    def test_runs_on_real_codebase(self) -> None:
        root = Path(__file__).parents[2]
        checker = DocsChecker(root=root)
        result = checker.run()
        assert result.name == "docs"
        # Duration should be recorded
        assert result.duration_ms >= 0

    def test_detects_broken_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create a README with a broken link
            readme = root / "README.md"
            readme.write_text("[Guide](./nonexistent.md)")

            # Initialize as git repo with the file tracked
            import subprocess

            subprocess.run(["git", "init"], cwd=root, capture_output=True)
            subprocess.run(["git", "add", "README.md"], cwd=root, capture_output=True)

            checker = DocsChecker(root=root)
            result = checker.run()
            # Should detect broken link
            assert any("broken" in d.message.lower() for d in result.diagnostics)

    def test_is_api_reference(self) -> None:
        checker = DocsChecker()

        # API reference (mostly signatures) - pattern expects standalone function names
        api_ref = """method(arg: str) -> int
other(x: int) -> str
.property"""
        assert checker._is_api_reference(api_ref) is True

        # Lines starting with . are also counted
        dot_api = """.method() -> int
.property -> str
.other()"""
        assert checker._is_api_reference(dot_api) is True

        # Real code with imports
        real_code = """from foo import bar

def example():
    return bar()"""
        assert checker._is_api_reference(real_code) is False

        # Empty code
        assert checker._is_api_reference("") is False

        # Only comments
        assert checker._is_api_reference("# comment") is False

    def test_is_noise(self) -> None:
        checker = DocsChecker()
        assert checker._is_noise("Variable not allowed in type expression") is True
        assert checker._is_noise("Type error: int expected") is False
        assert checker._is_noise("is obscured by a declaration") is True
        assert checker._is_noise("could not be resolved") is True
        assert checker._is_noise("Unexpected indentation") is True
        assert checker._is_noise("must return value") is True
        assert checker._is_noise("Expression value is unused") is True

    def test_check_examples_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # No README.md or llms.md, no guides
            checker = DocsChecker(root=Path(tmpdir))
            result = checker.run()
            # Should pass - no files to check
            assert result.status == "passed"

    def test_check_examples_no_python_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            readme = root / "README.md"
            # Markdown with no Python code blocks
            readme.write_text("# README\n\nJust text, no code.")

            import subprocess

            subprocess.run(["git", "init"], cwd=root, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=root, capture_output=True)

            checker = DocsChecker(root=root)
            result = checker.run()
            # Should pass - no code blocks to check
            assert result.status == "passed"

    def test_map_diagnostic_returns_none_out_of_range(self) -> None:
        from toolchain.utils import CodeBlock

        checker = DocsChecker()
        # Test when diagnostic line is out of range
        diag = {"range": {"start": {"line": 9999}}}
        blocks: list[CodeBlock] = []
        module = "# short module\npass"
        result = checker._map_diagnostic(diag, blocks, module)
        assert result is None

    def test_map_diagnostic_no_line(self) -> None:
        from toolchain.utils import CodeBlock

        checker = DocsChecker()
        # Test when diagnostic has no line info
        diag = {"range": {"start": {}}}
        blocks: list[CodeBlock] = []
        module = "# short module"
        result = checker._map_diagnostic(diag, blocks, module)
        assert result is None

    def test_handles_pyright_json_decode_error(self) -> None:
        """Test that JSON decode errors are properly reported."""
        import subprocess
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            readme = root / "README.md"
            readme.write_text("```python\nx = 1\n```")

            subprocess.run(["git", "init"], cwd=root, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=root, capture_output=True)

            # Mock subprocess.run to return malformed JSON from pyright
            original_run = subprocess.run

            def mock_run(
                cmd: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                if "pyright" in cmd:
                    return subprocess.CompletedProcess(
                        args=cmd,
                        returncode=1,
                        stdout="This is not valid JSON output from pyright",
                        stderr="Some error occurred",
                    )
                return original_run(cmd, **kwargs)

            with patch("subprocess.run", side_effect=mock_run):
                checker = DocsChecker(root=root)
                result = checker.run()

            # Should have a diagnostic about JSON parse failure
            assert any(
                "Failed to parse pyright output" in d.message for d in result.diagnostics
            )
            assert any("exit code 1" in d.message for d in result.diagnostics)
            assert any("This is not valid JSON" in d.message for d in result.diagnostics)

    def test_handles_pyright_failure_no_output(self) -> None:
        """Test that pyright failures with no output are properly reported."""
        import subprocess
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            readme = root / "README.md"
            readme.write_text("```python\nx = 1\n```")

            subprocess.run(["git", "init"], cwd=root, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=root, capture_output=True)

            # Mock subprocess.run to return empty output but non-zero exit code
            original_run = subprocess.run

            def mock_run(
                cmd: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                if "pyright" in cmd:
                    return subprocess.CompletedProcess(
                        args=cmd,
                        returncode=2,
                        stdout="",
                        stderr="pyright: command not found or crashed",
                    )
                return original_run(cmd, **kwargs)

            with patch("subprocess.run", side_effect=mock_run):
                checker = DocsChecker(root=root)
                result = checker.run()

            # Should have a diagnostic about pyright failure
            assert any(
                "Pyright failed with exit code 2" in d.message for d in result.diagnostics
            )
            assert any(
                "pyright: command not found" in d.message for d in result.diagnostics
            )


class TestParseTypecheck:
    """Tests for _parse_typecheck helper."""

    def test_parses_combined_output(self) -> None:
        from toolchain.checkers import _parse_typecheck

        # Combined ty + pyright output
        output = """error[invalid-type]: Type mismatch
  --> src/foo.py:10:5
  src/bar.py:20:10 - error: Cannot assign str to int"""
        diagnostics = _parse_typecheck(output, 1)
        assert len(diagnostics) == 2

    def test_prefixes_ty_diagnostics(self) -> None:
        from toolchain.checkers import _parse_typecheck

        # ty-style output
        output = """error[invalid-type]: Type mismatch
  --> src/foo.py:10:5"""
        diagnostics = _parse_typecheck(output, 1)
        assert len(diagnostics) == 1
        assert diagnostics[0].message.startswith("[ty] ")
        assert "Type mismatch" in diagnostics[0].message

    def test_prefixes_pyright_diagnostics(self) -> None:
        from toolchain.checkers import _parse_typecheck

        # pyright-style output (note: pyright parser expects 2 leading spaces)
        output = """  src/bar.py:20:10 - error: Cannot assign str to int"""
        diagnostics = _parse_typecheck(output, 1)
        assert len(diagnostics) == 1
        assert diagnostics[0].message.startswith("[pyright] ")
        assert "Cannot assign str to int" in diagnostics[0].message

    def test_combined_output_has_both_prefixes(self) -> None:
        from toolchain.checkers import _parse_typecheck

        # Combined ty + pyright output
        output = """error[invalid-type]: Type mismatch
  --> src/foo.py:10:5
  src/bar.py:20:10 - error: Cannot assign str to int"""
        diagnostics = _parse_typecheck(output, 1)
        assert len(diagnostics) == 2
        # Find the ty diagnostic
        ty_diags = [d for d in diagnostics if d.message.startswith("[ty] ")]
        assert len(ty_diags) == 1
        assert "Type mismatch" in ty_diags[0].message
        # Find the pyright diagnostic
        pyright_diags = [d for d in diagnostics if d.message.startswith("[pyright] ")]
        assert len(pyright_diags) == 1
        assert "Cannot assign str to int" in pyright_diags[0].message

    def test_empty_output(self) -> None:
        from toolchain.checkers import _parse_typecheck

        diagnostics = _parse_typecheck("", 0)
        assert len(diagnostics) == 0


class TestFactoryFunctions:
    """Tests for checker factory functions."""

    def test_create_format_checker(self) -> None:
        checker = create_format_checker()
        assert checker.name == "format"
        assert "ruff" in checker.check_command
        assert "ruff" in checker.fix_command
        assert "--check" in checker.check_command
        assert "--check" not in checker.fix_command

    def test_create_lint_checker(self) -> None:
        checker = create_lint_checker()
        assert checker.name == "lint"
        assert "--preview" in checker.command

    def test_create_typecheck_checker(self) -> None:
        checker = create_typecheck_checker()
        assert checker.name == "typecheck"
        assert "ty" in str(checker.command)
        assert "pyright" in str(checker.command)

    def test_create_test_checker(self) -> None:
        checker = create_test_checker()
        assert checker.name == "test"
        assert "pytest" in checker.command
        assert checker.timeout == 600  # 10 minutes

    def test_create_bandit_checker(self) -> None:
        checker = create_bandit_checker()
        assert checker.name == "bandit"
        assert "bandit" in checker.command

    def test_create_deptry_checker(self) -> None:
        checker = create_deptry_checker()
        assert checker.name == "deptry"
        assert "deptry" in checker.command

    def test_create_pip_audit_checker(self) -> None:
        checker = create_pip_audit_checker()
        assert checker.name == "pip-audit"
        assert "pip-audit" in checker.command

    def test_create_markdown_checker(self) -> None:
        checker = create_markdown_checker()
        assert checker.name == "markdown"
        assert "mdformat" in checker.check_command
        assert "mdformat" in checker.fix_command
        assert "--check" in checker.check_command
        assert "--check" not in checker.fix_command
        # Should have file list parser for text output
        assert checker.file_list_parser is not None

    def test_create_architecture_checker(self) -> None:
        checker = create_architecture_checker()
        assert checker.name == "architecture"
        assert isinstance(checker, ArchitectureChecker)

    def test_create_docs_checker(self) -> None:
        checker = create_docs_checker()
        assert checker.name == "docs"
        assert isinstance(checker, DocsChecker)

    def test_create_private_imports_checker(self) -> None:
        checker = create_private_imports_checker()
        assert checker.name == "private-imports"
        assert isinstance(checker, PrivateImportChecker)

    def test_create_banned_time_imports_checker(self) -> None:
        checker = create_banned_time_imports_checker()
        assert checker.name == "banned-time-imports"
        assert isinstance(checker, BannedTimeImportsChecker)

    def test_create_all_checkers(self) -> None:
        checkers = create_all_checkers()
        assert len(checkers) >= 10  # At least 10 checkers
        names = [c.name for c in checkers]
        assert "format" in names
        assert "lint" in names
        assert "typecheck" in names
        assert "test" in names
        assert "architecture" in names
        assert "private-imports" in names
        assert "banned-time-imports" in names
        assert "docs" in names


class TestParseMdformatFileList:
    """Tests for _parse_mdformat_file_list helper."""

    def test_parses_single_file(self) -> None:
        from toolchain.checkers import _parse_mdformat_file_list

        output = 'Error: File "README.md" is not formatted.'
        files = _parse_mdformat_file_list(output)
        assert files == ["README.md"]

    def test_parses_multiple_files(self) -> None:
        from toolchain.checkers import _parse_mdformat_file_list

        output = (
            'Error: File "README.md" is not formatted.\n'
            'Error: File "docs/guide.md" is not formatted.\n'
            'Error: File "specs/DESIGN.md" is not formatted.'
        )
        files = _parse_mdformat_file_list(output)
        assert files == ["README.md", "docs/guide.md", "specs/DESIGN.md"]

    def test_returns_sorted_files(self) -> None:
        from toolchain.checkers import _parse_mdformat_file_list

        output = (
            'Error: File "z.md" is not formatted.\n'
            'Error: File "a.md" is not formatted.\n'
            'Error: File "m.md" is not formatted.'
        )
        files = _parse_mdformat_file_list(output)
        assert files == ["a.md", "m.md", "z.md"]

    def test_returns_empty_for_no_errors(self) -> None:
        from toolchain.checkers import _parse_mdformat_file_list

        output = ""
        files = _parse_mdformat_file_list(output)
        assert files == []

    def test_ignores_non_matching_lines(self) -> None:
        from toolchain.checkers import _parse_mdformat_file_list

        output = (
            "Some other output\n"
            'Error: File "docs/guide.md" is not formatted.\n'
            "More text here"
        )
        files = _parse_mdformat_file_list(output)
        assert files == ["docs/guide.md"]

    def test_handles_paths_with_spaces(self) -> None:
        from toolchain.checkers import _parse_mdformat_file_list

        output = 'Error: File "docs/my guide.md" is not formatted.'
        files = _parse_mdformat_file_list(output)
        assert files == ["docs/my guide.md"]
