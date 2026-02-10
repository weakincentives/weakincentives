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

"""Tests for CodeLengthChecker."""

from __future__ import annotations

from pathlib import Path

from toolchain.checkers.code_length import CodeLengthChecker


class TestCodeLengthChecker:
    """Tests for file and function length checking."""

    def test_name_and_description(self) -> None:
        checker = CodeLengthChecker()
        assert checker.name == "code-length"
        assert "length" in checker.description.lower()

    def test_passes_on_short_files(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "ok.py").write_text("x = 1\n" * 10)
        checker = CodeLengthChecker(src_dir=src, test_dir=tmp_path / "tests")
        result = checker.run()
        assert result.status == "passed"
        assert len(result.diagnostics) == 0

    def test_warns_on_long_file(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "long.py").write_text("x = 1\n" * 700)
        checker = CodeLengthChecker(
            src_dir=src, test_dir=tmp_path / "tests", max_file_lines=620
        )
        result = checker.run()
        # File warnings don't fail the check
        assert result.status == "passed"
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].severity == "warning"
        assert "700 lines" in result.diagnostics[0].message

    def test_warns_on_long_function(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        lines = ["def big():\n"] + ["    x = 1\n"] * 130
        (src / "funcs.py").write_text("".join(lines))
        checker = CodeLengthChecker(
            src_dir=src, test_dir=tmp_path / "tests", max_function_lines=120
        )
        result = checker.run()
        # Default function severity is warning (doesn't fail the check)
        assert result.status == "passed"
        assert any(d.severity == "warning" for d in result.diagnostics)
        assert any("big" in d.message for d in result.diagnostics)

    def test_errors_on_long_function_when_configured(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        lines = ["def big():\n"] + ["    x = 1\n"] * 130
        (src / "funcs.py").write_text("".join(lines))
        checker = CodeLengthChecker(
            src_dir=src,
            test_dir=tmp_path / "tests",
            max_function_lines=120,
            function_length_severity="error",
        )
        result = checker.run()
        assert result.status == "failed"
        assert any(d.severity == "error" for d in result.diagnostics)
        assert any("big" in d.message for d in result.diagnostics)

    def test_warns_on_long_method(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        lines = ["class Foo:\n", "    def bar(self):\n"] + [
            "        x = 1\n"
        ] * 125
        (src / "cls.py").write_text("".join(lines))
        checker = CodeLengthChecker(
            src_dir=src, test_dir=tmp_path / "tests", max_function_lines=120
        )
        result = checker.run()
        assert result.status == "passed"
        diag = [d for d in result.diagnostics if d.severity == "warning"]
        assert any("Foo.bar" in d.message for d in diag)

    def test_passes_when_under_limits(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        lines = ["def ok():\n"] + ["    x = 1\n"] * 50
        (src / "small.py").write_text("".join(lines))
        checker = CodeLengthChecker(
            src_dir=src, test_dir=tmp_path / "tests"
        )
        result = checker.run()
        assert result.status == "passed"
        assert len(result.diagnostics) == 0

    def test_checks_test_directory_too(self, tmp_path: Path) -> None:
        tests = tmp_path / "tests"
        tests.mkdir()
        lines = ["def test_huge():\n"] + ["    x = 1\n"] * 130
        (tests / "test_big.py").write_text("".join(lines))
        checker = CodeLengthChecker(
            src_dir=tmp_path / "src", test_dir=tests, max_function_lines=120
        )
        result = checker.run()
        # Warnings only — doesn't fail
        assert result.status == "passed"
        assert any("test_huge" in d.message for d in result.diagnostics)

    def test_skips_pycache(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        cache = src / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "mod.py").write_text("x = 1\n" * 800)
        checker = CodeLengthChecker(
            src_dir=src, test_dir=tmp_path / "tests"
        )
        result = checker.run()
        assert result.status == "passed"

    def test_handles_syntax_error(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "bad.py").write_text("def (\n" * 700)
        checker = CodeLengthChecker(
            src_dir=src, test_dir=tmp_path / "tests", max_file_lines=620
        )
        result = checker.run()
        # Should still report file length warning even with syntax error
        assert any(d.severity == "warning" for d in result.diagnostics)

    def test_missing_directories(self, tmp_path: Path) -> None:
        checker = CodeLengthChecker(
            src_dir=tmp_path / "nonexistent_src",
            test_dir=tmp_path / "nonexistent_tests",
        )
        result = checker.run()
        assert result.status == "passed"
        assert len(result.diagnostics) == 0

    def test_both_file_and_function_violations(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        # Long file with a long function
        lines = ["def huge():\n"] + ["    x = 1\n"] * 699
        (src / "both.py").write_text("".join(lines))
        checker = CodeLengthChecker(
            src_dir=src,
            test_dir=tmp_path / "tests",
            max_file_lines=620,
            max_function_lines=120,
        )
        result = checker.run()
        # Both are warnings by default — doesn't fail
        assert result.status == "passed"
        warnings = [d for d in result.diagnostics if d.severity == "warning"]
        assert len(warnings) == 2  # file + function

    def test_async_function_checked(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        lines = ["async def big_async():\n"] + ["    x = 1\n"] * 130
        (src / "async_mod.py").write_text("".join(lines))
        checker = CodeLengthChecker(
            src_dir=src,
            test_dir=tmp_path / "tests",
            max_function_lines=120,
        )
        result = checker.run()
        assert result.status == "passed"
        assert any("big_async" in d.message for d in result.diagnostics)

    def test_severity_escalation(self, tmp_path: Path) -> None:
        """Setting severity to 'error' makes the check fail."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "long.py").write_text("x = 1\n" * 700)
        checker = CodeLengthChecker(
            src_dir=src,
            test_dir=tmp_path / "tests",
            max_file_lines=620,
            file_length_severity="error",
        )
        result = checker.run()
        assert result.status == "failed"
        assert any(d.severity == "error" for d in result.diagnostics)
