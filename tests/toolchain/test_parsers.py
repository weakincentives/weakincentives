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

from toolchain.parsers import (
    parse_pyright,
    parse_ruff,
    parse_ty,
    parse_vulture,
)


class TestParseRuff:
    """Tests for parse_ruff."""

    def test_parses_single_error(self) -> None:
        output = "src/foo.py:42:10: E501 Line too long (120 > 88)"
        diagnostics = parse_ruff(output, 1)
        assert len(diagnostics) == 1
        assert diagnostics[0].message == "[E501] Line too long (120 > 88)"
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "src/foo.py"
        assert diagnostics[0].location.line == 42
        assert diagnostics[0].location.column == 10

    def test_parses_multiple_errors(self) -> None:
        output = """src/foo.py:42:10: E501 Line too long
src/bar.py:17:1: F401 'os' imported but unused
src/baz.py:8:5: E721 Do not compare types"""
        diagnostics = parse_ruff(output, 1)
        assert len(diagnostics) == 3

    def test_empty_output(self) -> None:
        diagnostics = parse_ruff("", 0)
        assert len(diagnostics) == 0


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
