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

"""Tests for DocsChecker, typecheck parsing, factory functions, and mdformat helpers."""

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
