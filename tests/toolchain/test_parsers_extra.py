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

"""Tests for bun_test, bandit, deptry, pip_audit, and mdformat parsers."""

from __future__ import annotations

from toolchain.parsers import (
    parse_bandit,
    parse_bun_test,
    parse_deptry,
    parse_mdformat,
    parse_pip_audit,
)


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
