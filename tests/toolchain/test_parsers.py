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
    _extract_test_files,
    _extract_uncovered_files,
    _find_test_failure_line,
    parse_bandit,
    parse_bun_test,
    parse_deptry,
    parse_mdformat,
    parse_pip_audit,
    parse_pyright,
    parse_pytest,
    parse_ruff,
    parse_ty,
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


class TestParsePytest:
    """Tests for parse_pytest."""

    def test_returns_empty_on_success(self) -> None:
        diagnostics = parse_pytest("all tests passed", 0)
        assert len(diagnostics) == 0

    def test_parses_failed_test(self) -> None:
        output = "FAILED tests/test_foo.py::test_something - AssertionError: assert 1 == 2"
        diagnostics = parse_pytest(output, 1)
        assert len(diagnostics) == 1
        assert "test_something" in diagnostics[0].message
        assert "AssertionError" in diagnostics[0].message
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "tests/test_foo.py"

    def test_parses_multiple_failures(self) -> None:
        output = """FAILED tests/test_a.py::test_one - AssertionError
FAILED tests/test_b.py::test_two - ValueError"""
        diagnostics = parse_pytest(output, 1)
        assert len(diagnostics) == 2

    def test_parses_failed_test_without_reason(self) -> None:
        # Test case where FAILED line has no reason (no " - ")
        output = "FAILED tests/test_foo.py::test_something"
        diagnostics = parse_pytest(output, 1)
        assert len(diagnostics) == 1
        assert "test_something" in diagnostics[0].message
        # No AssertionError in message since no reason given
        assert "AssertionError" not in diagnostics[0].message

    def test_parses_collection_error(self) -> None:
        output = """ERROR collecting tests/test_broken.py
SyntaxError: invalid syntax"""
        diagnostics = parse_pytest(output, 1)
        assert len(diagnostics) == 1
        assert "Collection error" in diagnostics[0].message

    def test_parses_coverage_failure(self) -> None:
        output = "FAIL Required test coverage of 100% not reached. Total coverage: 95.5%"
        diagnostics = parse_pytest(output, 1)
        assert len(diagnostics) == 1
        assert "95.5%" in diagnostics[0].message
        assert "100%" in diagnostics[0].message

    def test_parses_coverage_failure_with_uncovered_files(self) -> None:
        """Test that uncovered files/lines are extracted from coverage report."""
        output = """Name                                      Stmts   Miss  Cover
-------------------------------------------------------------
src/weakincentives/module.py                100     10    90%
src/weakincentives/other.py                  50      5    90%
TOTAL                                        150     15    90%
FAIL Required test coverage of 100% not reached. Total coverage: 90%"""
        diagnostics = parse_pytest(output, 1)
        assert len(diagnostics) == 1
        assert "90%" in diagnostics[0].message
        assert "100%" in diagnostics[0].message
        assert "Uncovered" in diagnostics[0].message
        assert "module.py" in diagnostics[0].message
        assert "other.py" in diagnostics[0].message

    def test_parses_coverage_failure_with_missing_lines(self) -> None:
        """Test coverage report that includes specific missing line numbers."""
        output = """Name                                      Stmts   Miss  Cover   Missing
-------------------------------------------------------------------------
src/weakincentives/module.py                100     10    90%   1-5, 10, 20-30
TOTAL                                        100     10    90%
FAIL Required test coverage of 100% not reached. Total coverage: 90%"""
        diagnostics = parse_pytest(output, 1)
        assert len(diagnostics) == 1
        assert "lines 1-5, 10, 20-30" in diagnostics[0].message

    def test_parses_coverage_failure_with_branch_arrow_notation(self) -> None:
        """Test coverage report with -> branch notation in missing lines."""
        output = """Name                                                                    Stmts   Miss Branch BrPart  Cover   Missing
-------------------------------------------------------------------------------------------------------------------
src/weakincentives/adapters/claude_agent_sdk/adapter.py                   490     16    166     10    96%   195, 207, 214-222, 231->239, 240, 257
-------------------------------------------------------------------------------------------------------------------
TOTAL                                                                   16772     16   4462     10    99%
FAIL Required test coverage of 100% not reached. Total coverage: 99.87%"""
        diagnostics = parse_pytest(output, 1)
        assert len(diagnostics) == 1
        assert "99.87%" in diagnostics[0].message
        assert "Uncovered" in diagnostics[0].message
        assert "adapter.py" in diagnostics[0].message
        assert "231->239" in diagnostics[0].message

    def test_generic_failure_message(self) -> None:
        output = "some random failure output"
        diagnostics = parse_pytest(output, 1)
        assert len(diagnostics) == 1
        assert "Tests failed" in diagnostics[0].message

    def test_generic_failure_extracts_test_files(self) -> None:
        """Test that test files are extracted from generic failure output."""
        output = """Error occurred in tests/test_foo.py
Something went wrong in tests/unit/test_bar.py
Also failed: test_baz.py"""
        diagnostics = parse_pytest(output, 1)
        assert len(diagnostics) == 1
        assert "Tests failed" in diagnostics[0].message
        assert "Files involved" in diagnostics[0].message
        assert "test_foo.py" in diagnostics[0].message or "tests/test_foo.py" in diagnostics[0].message

    def test_generic_failure_limits_test_files_to_five(self) -> None:
        """Test that more than 5 test files are truncated."""
        output = """tests/test_a.py failed
tests/test_b.py failed
tests/test_c.py failed
tests/test_d.py failed
tests/test_e.py failed
tests/test_f.py failed
tests/test_g.py failed"""
        diagnostics = parse_pytest(output, 1)
        assert len(diagnostics) == 1
        assert "Tests failed" in diagnostics[0].message
        assert "(and 2 more)" in diagnostics[0].message

    def test_no_tests_ran(self) -> None:
        output = "no tests ran in 0.5s"
        diagnostics = parse_pytest(output, 1)
        assert len(diagnostics) == 1
        assert "No tests ran" in diagnostics[0].message

    def test_parses_with_traceback_context(self) -> None:
        """Test that traceback context is included in the diagnostic message."""
        output = """FAILED tests/test_foo.py::test_something
_ test_something _
tests/test_foo.py:42: in test_something
>   assert x == y
E   AssertionError: Values differ
E   assert 1 == 2
_ test_other _"""
        diagnostics = parse_pytest(output, 1)
        assert len(diagnostics) == 1
        assert "test_something" in diagnostics[0].message
        # Should include traceback details
        assert ("assert" in diagnostics[0].message or "AssertionError" in diagnostics[0].message)


class TestFindTestFailureLine:
    """Tests for _find_test_failure_line helper."""

    def test_finds_line_in_traceback(self) -> None:
        output = """tests/test_foo.py:42: in test_something
>       assert x == y"""
        line, msg = _find_test_failure_line(output, "tests/test_foo.py", "test_something")
        assert line == 42

    def test_finds_assertion_line(self) -> None:
        output = """tests/test_foo.py:42: AssertionError"""
        line, msg = _find_test_failure_line(output, "tests/test_foo.py", "test_other")
        assert line == 42

    def test_returns_none_when_not_found(self) -> None:
        output = "no traceback here"
        line, msg = _find_test_failure_line(output, "tests/test_foo.py", "test_something")
        assert line is None

    def test_extracts_traceback_message(self) -> None:
        """Test that we extract assertion details from traceback."""
        output = """_ test_something _
tests/test_foo.py:42: in test_something
>   assert x == y
E   AssertionError: x != y
_ test_other _"""
        line, msg = _find_test_failure_line(output, "tests/test_foo.py", "test_something")
        assert line == 42
        assert msg is not None
        assert "assert x == y" in msg or "AssertionError" in msg

    def test_test_section_without_error_lines(self) -> None:
        """Test section found but no error lines extracted."""
        output = """_ test_something _
tests/test_foo.py:42: in test_something
Some other output
_ test_other _"""
        line, msg = _find_test_failure_line(output, "tests/test_foo.py", "test_something")
        assert line == 42
        assert msg is None


class TestExtractUncoveredFiles:
    """Tests for _extract_uncovered_files helper."""

    def test_extracts_uncovered_files(self) -> None:
        output = """Name                                      Stmts   Miss  Cover
-------------------------------------------------------------
src/module.py                               100     10    90%
src/other.py                                 50      5    90%
TOTAL                                        150     15    90%"""
        result = _extract_uncovered_files(output)
        assert result is not None
        assert "module.py" in result
        assert "other.py" in result
        assert "90%" in result

    def test_extracts_files_with_missing_lines(self) -> None:
        output = """Name                                      Stmts   Miss  Cover   Missing
-------------------------------------------------------------------------
src/module.py                               100     10    90%   1-5, 10
TOTAL                                        100     10    90%"""
        result = _extract_uncovered_files(output)
        assert result is not None
        assert "lines 1-5, 10" in result

    def test_returns_none_for_full_coverage(self) -> None:
        output = """Name                                      Stmts   Miss  Cover
-------------------------------------------------------------
src/module.py                               100      0   100%
TOTAL                                        100      0   100%"""
        result = _extract_uncovered_files(output)
        assert result is None

    def test_returns_none_for_no_coverage_table(self) -> None:
        output = "No coverage table here"
        result = _extract_uncovered_files(output)
        assert result is None

    def test_skips_files_with_100_percent_coverage(self) -> None:
        """Test that files showing 100% are skipped even if miss count seems inconsistent."""
        # Edge case: coverage report shows 100% despite non-zero miss (shouldn't happen
        # in practice, but code should handle it gracefully)
        output = """Name                                      Stmts   Miss  Cover
-------------------------------------------------------------
src/partial.py                              100     10    90%
src/full.py                                 100      1   100%
TOTAL                                        200     11    95%"""
        result = _extract_uncovered_files(output)
        assert result is not None
        assert "partial.py" in result
        assert "full.py" not in result  # 100% coverage skipped

    def test_limits_output_to_ten_files(self) -> None:
        lines = [
            "Name                                      Stmts   Miss  Cover",
            "-------------------------------------------------------------",
        ]
        for i in range(15):
            lines.append(f"src/module{i}.py                           100     10    90%")
        lines.append("TOTAL                                        1500    150    90%")
        output = "\n".join(lines)
        result = _extract_uncovered_files(output)
        assert result is not None
        assert "... and 5 more files" in result

    def test_extracts_files_with_branch_coverage_columns(self) -> None:
        """Test parsing coverage output with branch coverage columns."""
        output = """Name                                      Stmts   Miss Branch BrPart  Cover   Missing
---------------------------------------------------------------------------------------
src/weakincentives/module.py                 17      5      2      0    63%   54-56, 65-66
src/weakincentives/other.py                 147     95     34      0    29%   48, 56, 64
TOTAL                                        164    100     36      0    35%"""
        result = _extract_uncovered_files(output)
        assert result is not None
        assert "module.py" in result
        assert "63%" in result
        assert "lines 54-56, 65-66" in result
        assert "other.py" in result
        assert "29%" in result

    def test_extracts_files_with_branch_arrow_notation(self) -> None:
        """Test parsing coverage output with -> branch notation in missing lines."""
        output = """Name                                                                    Stmts   Miss Branch BrPart  Cover   Missing
-------------------------------------------------------------------------------------------------------------------
src/weakincentives/adapters/claude_agent_sdk/adapter.py                   490     16    166     10    96%   195, 207, 214-222, 231->239, 240, 257, 893-894, 902, 927, 1263-1273, 1292
-------------------------------------------------------------------------------------------------------------------
TOTAL                                                                   16772     16   4462     10    99%"""
        result = _extract_uncovered_files(output)
        assert result is not None
        assert "adapter.py" in result
        assert "96%" in result
        assert "lines 195, 207, 214-222, 231->239" in result


class TestExtractTestFiles:
    """Tests for _extract_test_files helper."""

    def test_extracts_test_files(self) -> None:
        output = """Error in tests/test_foo.py
Also: tests/unit/test_bar.py"""
        result = _extract_test_files(output)
        assert "tests/test_foo.py" in result
        assert "tests/unit/test_bar.py" in result

    def test_extracts_standalone_test_files(self) -> None:
        output = "Running test_something.py"
        result = _extract_test_files(output)
        assert "test_something.py" in result

    def test_returns_empty_list_when_no_test_files(self) -> None:
        output = "No test files mentioned here"
        result = _extract_test_files(output)
        assert result == []

    def test_deduplicates_files(self) -> None:
        output = """tests/test_foo.py failed
tests/test_foo.py error"""
        result = _extract_test_files(output)
        assert len(result) == 1


class TestParseBunTest:
    """Tests for parse_bun_test."""

    def test_returns_empty_on_success(self) -> None:
        output = """bun test v1.3.6 (d530ed99)
--------------------------------------|---------|---------|-------------------
File                                  | % Funcs | % Lines | Uncovered Line #s
--------------------------------------|---------|---------|-------------------
All files                             |  100.00 |  100.00 |
--------------------------------------|---------|---------|-------------------

 78 pass
 0 fail
Ran 78 tests across 1 file. [42.00ms]"""
        diagnostics = parse_bun_test(output, 0)
        assert len(diagnostics) == 0

    def test_parses_failed_test(self) -> None:
        output = """bun test v1.3.6 (d530ed99)

tests/js/lib.test.js:
1 | import { expect, test } from "bun:test"; test("should fail", () => { expect(1).toBe(2); });
                                                                                   ^
error: expect(received).toBe(expected)

Expected: 2
Received: 1

      at <anonymous> (/home/user/tests/js/lib.test.js:1:80)
(fail) should fail [0.47ms]

 0 pass
 1 fail
Ran 1 test across 1 file. [22.00ms]"""
        diagnostics = parse_bun_test(output, 1)
        assert len(diagnostics) == 1
        assert "should fail" in diagnostics[0].message
        assert "Expected: 2" in diagnostics[0].message
        assert "Received: 1" in diagnostics[0].message
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "/home/user/tests/js/lib.test.js"
        assert diagnostics[0].location.line == 1

    def test_parses_multiple_failures(self) -> None:
        output = """bun test v1.3.6 (d530ed99)

error: expect(received).toBe(expected)

Expected: 2
Received: 1

      at <anonymous> (/home/user/tests/js/test1.js:10:5)
(fail) test one [0.1ms]

error: expect(received).toBe(expected)

Expected: "hello"
Received: "world"

      at <anonymous> (/home/user/tests/js/test2.js:20:10)
(fail) test two [0.2ms]

 0 pass
 2 fail
Ran 2 tests across 2 files. [50.00ms]"""
        diagnostics = parse_bun_test(output, 1)
        assert len(diagnostics) == 2
        assert "test one" in diagnostics[0].message
        assert "test two" in diagnostics[1].message

    def test_parses_failed_test_without_error_details(self) -> None:
        output = """bun test v1.3.6 (d530ed99)

(fail) test something [0.5ms]

 0 pass
 1 fail
Ran 1 test across 1 file. [22.00ms]"""
        diagnostics = parse_bun_test(output, 1)
        assert len(diagnostics) == 1
        assert "test something" in diagnostics[0].message

    def test_generic_failure_message(self) -> None:
        output = """bun test v1.3.6 (d530ed99)

 5 pass
 3 fail
Ran 8 tests across 2 files. [100.00ms]"""
        diagnostics = parse_bun_test(output, 1)
        assert len(diagnostics) == 1
        assert "JS tests failed" in diagnostics[0].message
        assert "3 failed" in diagnostics[0].message
        assert "5 passed" in diagnostics[0].message

    def test_skipped_message_returns_empty(self) -> None:
        """Test that 'bun not installed' message returns no diagnostics."""
        output = "bun not installed, skipping"
        diagnostics = parse_bun_test(output, 0)
        assert len(diagnostics) == 0

    def test_parses_import_error(self) -> None:
        """Test parsing import/syntax errors."""
        output = """bun test v1.3.6 (d530ed99)

error: Could not resolve: "nonexistent-module"
      at /home/user/tests/js/broken.test.js:1:0

 0 pass
 0 fail
Ran 0 tests across 1 file. [10.00ms]"""
        diagnostics = parse_bun_test(output, 1)
        assert len(diagnostics) == 1
        assert "Import/syntax error" in diagnostics[0].message
        assert "Could not resolve" in diagnostics[0].message
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "/home/user/tests/js/broken.test.js"
        assert diagnostics[0].location.line == 1

    def test_deduplicates_error_already_in_diagnostics(self) -> None:
        """Test that syntax errors aren't duplicated if already captured in failure.

        The syntax_pattern matches 'error: ...' immediately followed by 'at file:line:col'
        on the next line. If this error is already captured in the (fail) diagnostic,
        it should not be added again.
        """
        # This output has the error pattern matching both:
        # 1. The (fail) line's error extraction
        # 2. The syntax_pattern regex
        output = """bun test v1.3.6 (d530ed99)

error: TypeError: undefined is not a function
      at testFunc (/home/user/tests/js/lib.test.js:5:10)
(fail) test with thrown error [0.47ms]

 0 pass
 1 fail
Ran 1 test across 1 file. [22.00ms]"""
        diagnostics = parse_bun_test(output, 1)
        # Should have exactly 1 diagnostic (the test failure), not 2
        # The error "TypeError: undefined is not a function" appears in the
        # (fail) diagnostic and would also match syntax_pattern
        assert len(diagnostics) == 1
        assert "test with thrown error" in diagnostics[0].message
        # Error should be included in the message
        assert "TypeError" in diagnostics[0].message

    def test_generic_failure_without_summary(self) -> None:
        """Test fallback when no pass/fail summary is present."""
        output = """bun test v1.3.6 (d530ed99)

Something went wrong"""
        diagnostics = parse_bun_test(output, 1)
        assert len(diagnostics) == 1
        assert "JS tests failed" in diagnostics[0].message
        assert "Reproduce" in diagnostics[0].message


class TestParseBandit:
    """Tests for parse_bandit."""

    def test_returns_empty_on_success(self) -> None:
        diagnostics = parse_bandit("No issues found", 0)
        assert len(diagnostics) == 0

    def test_parses_security_issue(self) -> None:
        output = """>> Issue: [B101:ASSERT_USED:LOW] Use of assert detected
   Location: src/foo.py:42"""
        diagnostics = parse_bandit(output, 1)
        assert len(diagnostics) == 1
        assert "B101" in diagnostics[0].message
        assert "assert" in diagnostics[0].message.lower()
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "src/foo.py"
        assert diagnostics[0].location.line == 42


class TestParseDeptry:
    """Tests for parse_deptry."""

    def test_returns_empty_on_success(self) -> None:
        diagnostics = parse_deptry("", 0)
        assert len(diagnostics) == 0

    def test_parses_dependency_issue(self) -> None:
        output = "src/foo.py:1:0: DEP001 'missing_package' imported but not declared"
        diagnostics = parse_deptry(output, 1)
        assert len(diagnostics) == 1
        assert "DEP001" in diagnostics[0].message
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "src/foo.py"

    def test_fallback_parsing(self) -> None:
        output = """Found dependency issues:
some issue description"""
        diagnostics = parse_deptry(output, 1)
        assert len(diagnostics) >= 1


class TestParsePipAudit:
    """Tests for parse_pip_audit."""

    def test_returns_empty_on_success(self) -> None:
        diagnostics = parse_pip_audit("No vulnerabilities found", 0)
        assert len(diagnostics) == 0

    def test_parses_vulnerability(self) -> None:
        output = "requests 2.25.0 has CVE-2023-12345"
        diagnostics = parse_pip_audit(output, 1)
        assert len(diagnostics) == 1
        assert "requests" in diagnostics[0].message
        assert "CVE-2023-12345" in diagnostics[0].message

    def test_parses_ghsa_vulnerability(self) -> None:
        output = "urllib3 1.26.0 affected by GHSA-abcd-1234-efgh"
        diagnostics = parse_pip_audit(output, 1)
        assert len(diagnostics) == 1
        assert "GHSA-abcd-1234-efgh" in diagnostics[0].message

    def test_generic_failure(self) -> None:
        output = "audit failed for unknown reasons"
        diagnostics = parse_pip_audit(output, 1)
        assert len(diagnostics) == 1
        assert "Vulnerability check failed" in diagnostics[0].message
        # Should include the output in the message
        assert "audit failed" in diagnostics[0].message

    def test_generic_failure_includes_output_preview(self) -> None:
        """Test that fallback includes first few lines of output."""
        output = """Error: Could not resolve dependencies
Package foo requires bar>=2.0
Package baz requires bar<2.0
Conflict detected"""
        diagnostics = parse_pip_audit(output, 1)
        assert len(diagnostics) == 1
        assert "Vulnerability check failed" in diagnostics[0].message
        assert "Output:" in diagnostics[0].message
        assert "Could not resolve dependencies" in diagnostics[0].message

    def test_generic_failure_with_empty_output(self) -> None:
        """Test fallback with empty/whitespace-only output."""
        output = "   \n   \n   "
        diagnostics = parse_pip_audit(output, 1)
        assert len(diagnostics) == 1
        assert "Vulnerability check failed" in diagnostics[0].message
        # Should not have "Output:" section when output is empty
        assert "Output:" not in diagnostics[0].message

    def test_generic_failure_truncates_long_output(self) -> None:
        """Test that fallback truncates output longer than 5 lines."""
        output = """Line 1: Some error
Line 2: Another error
Line 3: More info
Line 4: Additional details
Line 5: Even more
Line 6: Extra line
Line 7: Yet another"""
        diagnostics = parse_pip_audit(output, 1)
        assert len(diagnostics) == 1
        assert "..." in diagnostics[0].message
        assert "Line 6" not in diagnostics[0].message


class TestParseMdformat:
    """Tests for parse_mdformat."""

    def test_returns_empty_on_success(self) -> None:
        diagnostics = parse_mdformat("", 0)
        assert len(diagnostics) == 0

    def test_parses_files_needing_format(self) -> None:
        output = """README.md
docs/guide.md"""
        diagnostics = parse_mdformat(output, 1)
        assert len(diagnostics) == 2
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "README.md"
        # Check improved message format
        assert "Markdown formatting required" in diagnostics[0].message
        assert "mdformat README.md" in diagnostics[0].message

    def test_message_includes_fix_command(self) -> None:
        """Test that each file diagnostic includes the fix command."""
        output = "docs/api.md"
        diagnostics = parse_mdformat(output, 1)
        assert len(diagnostics) == 1
        assert "Fix:" in diagnostics[0].message
        assert "mdformat docs/api.md" in diagnostics[0].message

    def test_parses_error_format(self) -> None:
        """Test parsing the Error: File format used by modern mdformat."""
        output = 'Error: File "/home/user/project/README.md" is not formatted.'
        diagnostics = parse_mdformat(output, 1)
        assert len(diagnostics) == 1
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "/home/user/project/README.md"
        assert "Markdown formatting required" in diagnostics[0].message
        assert "mdformat /home/user/project/README.md" in diagnostics[0].message

    def test_parses_multiple_error_format_files(self) -> None:
        """Test parsing multiple Error: File lines."""
        output = """Error: File "docs/api.md" is not formatted.
Error: File "guides/tutorial.md" is not formatted."""
        diagnostics = parse_mdformat(output, 1)
        assert len(diagnostics) == 2
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "docs/api.md"
        assert diagnostics[1].location is not None
        assert diagnostics[1].location.file == "guides/tutorial.md"

    def test_fallback_skips_non_md_lines(self) -> None:
        """Test that fallback parser skips lines that aren't .md files."""
        output = """Some random output
README.md
Another line that is not a markdown file
docs/guide.md"""
        diagnostics = parse_mdformat(output, 1)
        assert len(diagnostics) == 2
        assert diagnostics[0].location is not None
        assert diagnostics[0].location.file == "README.md"
        assert diagnostics[1].location is not None
        assert diagnostics[1].location.file == "docs/guide.md"
