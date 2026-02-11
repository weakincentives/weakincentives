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

from toolchain.checkers.code_length import CodeLengthChecker, load_baseline


def _write_baseline(tmp_path: Path, entries: list[str]) -> Path:
    """Write a baseline file and return its path."""
    bl = tmp_path / "baseline.txt"
    bl.write_text("\n".join(entries) + "\n")
    return bl


class TestLoadBaseline:
    """Tests for baseline file loading."""

    def test_loads_entries(self, tmp_path: Path) -> None:
        path = _write_baseline(tmp_path, ["src/big.py", "src/a.py:Foo.bar"])
        result = load_baseline(path)
        assert result == frozenset({"src/big.py", "src/a.py:Foo.bar"})

    def test_skips_comments_and_blanks(self, tmp_path: Path) -> None:
        bl = tmp_path / "bl.txt"
        bl.write_text("# comment\n\nsrc/ok.py\n  # indented comment\n")
        result = load_baseline(bl)
        assert result == frozenset({"src/ok.py"})

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = load_baseline(tmp_path / "nope.txt")
        assert result == frozenset()


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
        bl = _write_baseline(tmp_path, [])
        checker = CodeLengthChecker(
            src_dir=src, test_dir=tmp_path / "tests", baseline_path=bl
        )
        result = checker.run()
        assert result.status == "passed"
        assert len(result.diagnostics) == 0

    # ── baseline: known violations are warnings ─────────────────────

    def test_baselined_file_is_warning(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        long_file = src / "long.py"
        long_file.write_text("x = 1\n" * 700)
        bl = _write_baseline(tmp_path, [str(long_file)])
        checker = CodeLengthChecker(
            src_dir=src,
            test_dir=tmp_path / "tests",
            max_file_lines=620,
            baseline_path=bl,
        )
        result = checker.run()
        assert result.status == "passed"
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].severity == "warning"
        assert "700 lines" in result.diagnostics[0].message

    def test_baselined_function_is_warning(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        lines = ["def big():\n"] + ["    x = 1\n"] * 130
        (src / "funcs.py").write_text("".join(lines))
        bl = _write_baseline(tmp_path, [f"{src / 'funcs.py'}:big"])
        checker = CodeLengthChecker(
            src_dir=src,
            test_dir=tmp_path / "tests",
            max_function_lines=120,
            baseline_path=bl,
        )
        result = checker.run()
        assert result.status == "passed"
        assert any(d.severity == "warning" for d in result.diagnostics)
        assert any("big" in d.message for d in result.diagnostics)

    # ── new violations are errors ───────────────────────────────────

    def test_new_file_violation_is_error(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "long.py").write_text("x = 1\n" * 700)
        bl = _write_baseline(tmp_path, [])  # empty baseline
        checker = CodeLengthChecker(
            src_dir=src,
            test_dir=tmp_path / "tests",
            max_file_lines=620,
            baseline_path=bl,
        )
        result = checker.run()
        assert result.status == "failed"
        assert result.diagnostics[0].severity == "error"

    def test_new_function_violation_is_error(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        lines = ["def big():\n"] + ["    x = 1\n"] * 130
        (src / "funcs.py").write_text("".join(lines))
        bl = _write_baseline(tmp_path, [])  # empty baseline
        checker = CodeLengthChecker(
            src_dir=src,
            test_dir=tmp_path / "tests",
            max_function_lines=120,
            baseline_path=bl,
        )
        result = checker.run()
        assert result.status == "failed"
        assert any(d.severity == "error" for d in result.diagnostics)
        assert any("big" in d.message for d in result.diagnostics)

    def test_new_method_violation_is_error(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        lines = ["class Foo:\n", "    def bar(self):\n"] + [
            "        x = 1\n"
        ] * 125
        (src / "cls.py").write_text("".join(lines))
        bl = _write_baseline(tmp_path, [])
        checker = CodeLengthChecker(
            src_dir=src,
            test_dir=tmp_path / "tests",
            max_function_lines=120,
            baseline_path=bl,
        )
        result = checker.run()
        assert result.status == "failed"
        diag = [d for d in result.diagnostics if d.severity == "error"]
        assert any("Foo.bar" in d.message for d in diag)

    # ── mixed: baselined + new in same run ──────────────────────────

    def test_mixed_baseline_and_new(self, tmp_path: Path) -> None:
        """Baselined file warns, new file errors, overall fails."""
        src = tmp_path / "src"
        src.mkdir()
        known = src / "known.py"
        known.write_text("x = 1\n" * 700)
        fresh = src / "fresh.py"
        fresh.write_text("x = 1\n" * 700)
        bl = _write_baseline(tmp_path, [str(known)])
        checker = CodeLengthChecker(
            src_dir=src,
            test_dir=tmp_path / "tests",
            max_file_lines=620,
            baseline_path=bl,
        )
        result = checker.run()
        assert result.status == "failed"
        warnings = [d for d in result.diagnostics if d.severity == "warning"]
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(warnings) == 1
        assert len(errors) == 1

    # ── other behavior ──────────────────────────────────────────────

    def test_passes_when_under_limits(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        lines = ["def ok():\n"] + ["    x = 1\n"] * 50
        (src / "small.py").write_text("".join(lines))
        bl = _write_baseline(tmp_path, [])
        checker = CodeLengthChecker(
            src_dir=src, test_dir=tmp_path / "tests", baseline_path=bl
        )
        result = checker.run()
        assert result.status == "passed"
        assert len(result.diagnostics) == 0

    def test_checks_test_directory_too(self, tmp_path: Path) -> None:
        tests = tmp_path / "tests"
        tests.mkdir()
        lines = ["def test_huge():\n"] + ["    x = 1\n"] * 130
        (tests / "test_big.py").write_text("".join(lines))
        bl = _write_baseline(tmp_path, [])
        checker = CodeLengthChecker(
            src_dir=tmp_path / "src",
            test_dir=tests,
            max_function_lines=120,
            baseline_path=bl,
        )
        result = checker.run()
        assert result.status == "failed"
        assert any("test_huge" in d.message for d in result.diagnostics)

    def test_skips_pycache(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        cache = src / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "mod.py").write_text("x = 1\n" * 800)
        bl = _write_baseline(tmp_path, [])
        checker = CodeLengthChecker(
            src_dir=src, test_dir=tmp_path / "tests", baseline_path=bl
        )
        result = checker.run()
        assert result.status == "passed"

    def test_handles_syntax_error(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        bad = src / "bad.py"
        bad.write_text("def (\n" * 700)
        bl = _write_baseline(tmp_path, [str(bad)])
        checker = CodeLengthChecker(
            src_dir=src,
            test_dir=tmp_path / "tests",
            max_file_lines=620,
            baseline_path=bl,
        )
        result = checker.run()
        assert any(d.severity == "warning" for d in result.diagnostics)

    def test_missing_directories(self, tmp_path: Path) -> None:
        bl = _write_baseline(tmp_path, [])
        checker = CodeLengthChecker(
            src_dir=tmp_path / "nonexistent_src",
            test_dir=tmp_path / "nonexistent_tests",
            baseline_path=bl,
        )
        result = checker.run()
        assert result.status == "passed"
        assert len(result.diagnostics) == 0

    def test_both_file_and_function_violations(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        both = src / "both.py"
        lines = ["def huge():\n"] + ["    x = 1\n"] * 699
        both.write_text("".join(lines))
        # Baseline the file but not the function
        bl = _write_baseline(tmp_path, [str(both)])
        checker = CodeLengthChecker(
            src_dir=src,
            test_dir=tmp_path / "tests",
            max_file_lines=620,
            max_function_lines=120,
            baseline_path=bl,
        )
        result = checker.run()
        assert result.status == "failed"  # function is not baselined
        warnings = [d for d in result.diagnostics if d.severity == "warning"]
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(warnings) == 1  # file is baselined
        assert len(errors) == 1  # function is new

    def test_async_function_checked(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        lines = ["async def big_async():\n"] + ["    x = 1\n"] * 130
        (src / "async_mod.py").write_text("".join(lines))
        bl = _write_baseline(tmp_path, [])
        checker = CodeLengthChecker(
            src_dir=src,
            test_dir=tmp_path / "tests",
            max_function_lines=120,
            baseline_path=bl,
        )
        result = checker.run()
        assert result.status == "failed"
        assert any("big_async" in d.message for d in result.diagnostics)

    def test_unreadable_file_skipped(self, tmp_path: Path) -> None:
        """Files that can't be read are silently skipped."""
        src = tmp_path / "src"
        src.mkdir()
        bad = src / "unreadable.py"
        bad.write_bytes(b"\x80\x81\x82" * 100)
        bl = _write_baseline(tmp_path, [])
        checker = CodeLengthChecker(
            src_dir=src, test_dir=tmp_path / "tests", baseline_path=bl
        )
        result = checker.run()
        assert result.status == "passed"
        assert len(result.diagnostics) == 0

    def test_standalone_function_with_class_present(self, tmp_path: Path) -> None:
        """A long module-level function in a file that also has a class."""
        src = tmp_path / "src"
        src.mkdir()
        code = (
            "class Foo:\n"
            "    def short(self):\n"
            "        pass\n"
            "\n"
            "def standalone():\n"
        ) + "    x = 1\n" * 130
        (src / "mixed.py").write_text(code)
        bl = _write_baseline(tmp_path, [])
        checker = CodeLengthChecker(
            src_dir=src,
            test_dir=tmp_path / "tests",
            max_function_lines=120,
            baseline_path=bl,
        )
        result = checker.run()
        func_diags = [d for d in result.diagnostics if "standalone" in d.message]
        assert len(func_diags) == 1
        assert "Foo." not in func_diags[0].message

    def test_function_without_end_lineno(self, tmp_path: Path) -> None:
        """Functions where end_lineno is None are skipped gracefully."""
        import ast
        import unittest.mock

        src = tmp_path / "src"
        src.mkdir()
        code = "def f():\n    pass\n"
        (src / "mod.py").write_text(code)

        bl = _write_baseline(tmp_path, [])
        checker = CodeLengthChecker(
            src_dir=src,
            test_dir=tmp_path / "tests",
            max_function_lines=1,
            baseline_path=bl,
        )

        orig_parse = ast.parse

        def patched_parse(source: str, filename: str = "") -> ast.Module:
            tree = orig_parse(source, filename=filename)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    node.end_lineno = None  # type: ignore[assignment]
            return tree

        with unittest.mock.patch("ast.parse", side_effect=patched_parse):
            result = checker.run()

        func_diags = [
            d
            for d in result.diagnostics
            if "lines" in d.message and "max" in d.message
        ]
        assert len(func_diags) == 0

    def test_no_baseline_file_treats_all_as_new(self, tmp_path: Path) -> None:
        """When baseline file doesn't exist, every violation is an error."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "big.py").write_text("x = 1\n" * 700)
        checker = CodeLengthChecker(
            src_dir=src,
            test_dir=tmp_path / "tests",
            max_file_lines=620,
            baseline_path=tmp_path / "missing_baseline.txt",
        )
        result = checker.run()
        assert result.status == "failed"
        assert result.diagnostics[0].severity == "error"

    # ── integration: real baseline file ─────────────────────────────

    def test_real_baseline_all_known(self) -> None:
        """The shipped baseline covers all current violations (no errors)."""
        checker = CodeLengthChecker()
        result = checker.run()
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert errors == [], (
            "New code-length violations found that are not in the baseline. "
            "Either refactor them or add them to "
            "toolchain/checkers/code_length_baseline.txt:\n"
            + "\n".join(d.message for d in errors)
        )
