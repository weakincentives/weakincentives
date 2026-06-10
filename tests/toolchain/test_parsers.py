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

"""Tests for toolchain diagnostic parsers."""

from __future__ import annotations

import json
from pathlib import Path

from toolchain.parsers import (
    parse_biome,
    parse_biome_files,
    parse_pyright,
    parse_ruff_format,
    parse_ruff_format_files,
    parse_ruff_json,
    parse_ty,
    parse_vulture,
    relativize_path,
)


def _ruff_json_item(
    *,
    filename: str = "src/foo.py",
    code: str | None = "F401",
    message: str = "`os` imported but unused",
    row: int = 1,
    column: int = 8,
    fix_applicability: str | None = "safe",
) -> dict[str, object]:
    """Build a ruff JSON diagnostic entry."""
    fix = {"applicability": fix_applicability} if fix_applicability else None
    return {
        "filename": filename,
        "code": code,
        "message": message,
        "location": {"row": row, "column": column},
        "fix": fix,
    }


class TestRelativizePath:
    """Tests for relativize_path."""

    def test_relativizes_path_under_cwd(self) -> None:
        absolute = str(Path.cwd() / "src" / "foo.py")
        assert relativize_path(absolute) == str(Path("src") / "foo.py")

    def test_keeps_path_outside_cwd(self) -> None:
        assert relativize_path("/somewhere/else/foo.py") == "/somewhere/else/foo.py"


class TestParseRuffJson:
    """Tests for parse_ruff_json."""

    def test_parses_single_error(self) -> None:
        output = json.dumps([_ruff_json_item(fix_applicability=None)])
        diagnostics = parse_ruff_json(output, 1)
        assert len(diagnostics) == 1
        assert diagnostics[0].message == "[F401] `os` imported but unused"
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "src/foo.py"
        assert diagnostics[0].location.line == 1
        assert diagnostics[0].location.column == 8

    def test_marks_fixable_and_prepends_summary(self) -> None:
        output = json.dumps(
            [
                _ruff_json_item(fix_applicability="safe"),
                _ruff_json_item(
                    filename="src/bar.py",
                    code="E721",
                    message="Do not compare types",
                    fix_applicability=None,
                ),
            ]
        )
        diagnostics = parse_ruff_json(output, 1)
        assert len(diagnostics) == 3  # summary + 2 issues
        assert diagnostics[0].severity == "info"
        assert "1 of 2 issue(s) auto-fixable" in diagnostics[0].message
        assert "make lint-fix" in diagnostics[0].message
        assert diagnostics[1].message.endswith("(fixable)")
        assert "(fixable)" not in diagnostics[2].message

    def test_unsafe_fix_not_marked_fixable(self) -> None:
        output = json.dumps([_ruff_json_item(fix_applicability="unsafe")])
        diagnostics = parse_ruff_json(output, 1)
        assert len(diagnostics) == 1
        assert "(fixable)" not in diagnostics[0].message

    def test_no_summary_when_nothing_fixable(self) -> None:
        output = json.dumps([_ruff_json_item(fix_applicability=None)])
        diagnostics = parse_ruff_json(output, 1)
        assert len(diagnostics) == 1
        assert "auto-fixable" not in diagnostics[0].message

    def test_exit_zero_returns_empty(self) -> None:
        assert parse_ruff_json("[]", 0) == ()

    def test_unparseable_output_returns_empty(self) -> None:
        assert parse_ruff_json("ruff crashed: panic", 1) == ()
        assert parse_ruff_json("", 1) == ()

    def test_non_list_json_returns_empty(self) -> None:
        assert parse_ruff_json('{"filename": "x.py"}', 1) == ()

    def test_tolerates_trailing_noise_after_json(self) -> None:
        output = json.dumps([_ruff_json_item()]) + "\nwarning: something on stderr"
        diagnostics = parse_ruff_json(output, 1)
        assert len(diagnostics) == 2  # summary + issue

    def test_skips_non_dict_items(self) -> None:
        output = json.dumps(["not a dict", _ruff_json_item(fix_applicability=None)])
        diagnostics = parse_ruff_json(output, 1)
        assert len(diagnostics) == 1

    def test_skips_items_without_filename_or_message(self) -> None:
        output = json.dumps(
            [
                {"code": "F401", "message": "no filename"},
                {"filename": "src/foo.py", "code": "F401", "message": None},
            ]
        )
        assert parse_ruff_json(output, 1) == ()

    def test_handles_missing_code_and_location(self) -> None:
        output = json.dumps(
            [
                {
                    "filename": "src/foo.py",
                    "code": None,
                    "message": "SyntaxError: unexpected token",
                    "location": None,
                    "fix": None,
                }
            ]
        )
        diagnostics = parse_ruff_json(output, 1)
        assert len(diagnostics) == 1
        assert diagnostics[0].message == "SyntaxError: unexpected token"
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "src/foo.py"
        assert diagnostics[0].location.line is None

    def test_relativizes_absolute_filenames(self) -> None:
        absolute = str(Path.cwd() / "src" / "foo.py")
        output = json.dumps(
            [_ruff_json_item(filename=absolute, fix_applicability=None)]
        )
        diagnostics = parse_ruff_json(output, 1)
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == str(Path("src") / "foo.py")


class TestParseRuffFormat:
    """Tests for parse_ruff_format and parse_ruff_format_files."""

    def test_extracts_file_list(self) -> None:
        output = (
            "Would reformat: src/zoo.py\n"
            "Would reformat: src/bar.py\n"
            "2 files would be reformatted, 531 files already formatted\n"
        )
        assert parse_ruff_format_files(output) == ["src/bar.py", "src/zoo.py"]

    def test_file_list_empty_when_no_matches(self) -> None:
        assert parse_ruff_format_files("All done!") == []

    def test_diagnostics_per_file(self) -> None:
        output = "Would reformat: src/foo.py\n1 file would be reformatted"
        diagnostics = parse_ruff_format(output, 1)
        assert len(diagnostics) == 1
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "src/foo.py"
        assert "make format" in diagnostics[0].message

    def test_exit_zero_returns_empty(self) -> None:
        assert parse_ruff_format("anything", 0) == ()


class TestParseBiome:
    """Tests for parse_biome."""

    def test_parses_error_line(self) -> None:
        output = (
            "::error title=lint/suspicious/noDoubleEquals,"
            "file=src/weakincentives/cli/static/app.js,"
            "line=2,endLine=2,col=7,endColumn=9::Using == may be unsafe."
        )
        diagnostics = parse_biome(output, 1)
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == "error"
        assert diagnostics[0].message == (
            "[lint/suspicious/noDoubleEquals] Using == may be unsafe."
        )
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "src/weakincentives/cli/static/app.js"
        assert diagnostics[0].location.line == 2
        assert diagnostics[0].location.column == 7
        assert diagnostics[0].location.end_line == 2
        assert diagnostics[0].location.end_column == 9

    def test_parses_warning_and_notice(self) -> None:
        output = (
            "::warning title=lint/style/useConst,file=a.js,"
            "line=1,endLine=1,col=1,endColumn=4::Use const.\n"
            "::notice title=,file=b.js,"
            "line=3,endLine=3,col=2,endColumn=5::Informational."
        )
        diagnostics = parse_biome(output, 1)
        assert len(diagnostics) == 2
        assert diagnostics[0].severity == "warning"
        assert diagnostics[1].severity == "info"
        # Empty title omits the bracket prefix
        assert diagnostics[1].message == "Informational."

    def test_exit_zero_returns_empty(self) -> None:
        assert (
            parse_biome(
                "::error title=x,file=a.js,line=1,endLine=1,col=1,endColumn=2::m", 0
            )
            == ()
        )

    def test_unmatched_output_returns_empty(self) -> None:
        assert parse_biome("some unrelated failure", 1) == ()


class TestParseBiomeFiles:
    """Tests for parse_biome_files."""

    def test_extracts_unique_sorted_files(self) -> None:
        output = (
            "::error title=a,file=z.js,line=1,endLine=1,col=1,endColumn=2::m\n"
            "::error title=b,file=a.js,line=2,endLine=2,col=1,endColumn=2::m\n"
            "::warning title=c,file=z.js,line=3,endLine=3,col=1,endColumn=2::m"
        )
        assert parse_biome_files(output) == ["a.js", "z.js"]

    def test_empty_when_no_matches(self) -> None:
        assert parse_biome_files("nothing to see") == []


class TestParsePyright:
    """Tests for parse_pyright."""

    def test_parses_error(self) -> None:
        output = '  src/foo.py:42:10 - error: Argument of type "str" cannot be assigned to "int"'
        diagnostics = parse_pyright(output, 1)
        assert len(diagnostics) == 1
        assert "cannot be assigned" in diagnostics[0].message
        assert diagnostics[0].severity == "error"
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "src/foo.py"
        assert diagnostics[0].location.line == 42

    def test_parses_error_with_rule_name(self) -> None:
        """Test that rule names like (reportGeneralTypeIssues) are captured."""
        output = '  src/foo.py:42:10 - error: Argument of type "str" cannot be assigned to "int" (reportGeneralTypeIssues)'
        diagnostics = parse_pyright(output, 1)
        assert len(diagnostics) == 1
        assert "[reportGeneralTypeIssues]" in diagnostics[0].message
        assert "cannot be assigned" in diagnostics[0].message
        # Rule name should be at the start, not end
        assert not diagnostics[0].message.endswith("(reportGeneralTypeIssues)")

    def test_parses_warning(self) -> None:
        output = "  src/foo.py:10:5 - warning: Variable is unused"
        diagnostics = parse_pyright(output, 1)
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == "warning"

    def test_parses_info(self) -> None:
        output = "  src/foo.py:10:5 - info: Consider using type annotation"
        diagnostics = parse_pyright(output, 1)
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == "info"

    def test_empty_output(self) -> None:
        diagnostics = parse_pyright("", 0)
        assert len(diagnostics) == 0


class TestParseTy:
    """Tests for parse_ty."""

    def test_parses_error(self) -> None:
        output = """error[invalid-type]: Type "str" is not assignable to "int"
  --> src/foo.py:42:10"""
        diagnostics = parse_ty(output, 1)
        assert len(diagnostics) == 1
        assert "not assignable" in diagnostics[0].message
        # Error code should be included in message
        assert "[invalid-type]" in diagnostics[0].message
        assert diagnostics[0].severity == "error"
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "src/foo.py"
        assert diagnostics[0].location.line == 42

    def test_parses_warning(self) -> None:
        output = """warning[unused]: Variable 'x' is never used
  --> src/foo.py:10:5"""
        diagnostics = parse_ty(output, 1)
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == "warning"
        # Error code should be included in message
        assert "[unused]" in diagnostics[0].message

    def test_empty_output(self) -> None:
        diagnostics = parse_ty("", 0)
        assert len(diagnostics) == 0


class TestParseVulture:
    """Tests for parse_vulture."""

    def test_parses_single_item(self) -> None:
        output = "src/foo.py:42: unused function 'bar' (80% confidence)"
        diagnostics = parse_vulture(output, 1)
        assert len(diagnostics) == 1
        assert diagnostics[0].message == "unused function 'bar' (80% confidence)"
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "src/foo.py"
        assert diagnostics[0].location.line == 42

    def test_parses_multiple_items(self) -> None:
        output = (
            "src/foo.py:10: unused variable 'x' (80% confidence)\n"
            "src/bar.py:20: unused class 'MyClass' (90% confidence)\n"
            "src/baz.py:30: unused import 'os' (100% confidence)"
        )
        diagnostics = parse_vulture(output, 1)
        assert len(diagnostics) == 3
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "src/foo.py"
        assert diagnostics[1].location is not None
        assert diagnostics[1].location.file == "src/bar.py"
        assert diagnostics[2].location is not None
        assert diagnostics[2].location.file == "src/baz.py"

    def test_empty_output_no_issues(self) -> None:
        diagnostics = parse_vulture("", 0)
        assert len(diagnostics) == 0

    def test_exit_code_ignored(self) -> None:
        """Exit code does not affect parsing; only output content matters."""
        output = "src/foo.py:1: unused attribute 'x' (80% confidence)"
        diagnostics_fail = parse_vulture(output, 1)
        diagnostics_pass = parse_vulture(output, 0)
        assert len(diagnostics_fail) == len(diagnostics_pass) == 1
