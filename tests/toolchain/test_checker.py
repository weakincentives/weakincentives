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

"""Tests for toolchain checker module."""

from __future__ import annotations

from toolchain.checker import SubprocessChecker
from toolchain.result import Diagnostic, Location


class TestSubprocessChecker:
    """Tests for SubprocessChecker."""

    def test_successful_command(self) -> None:
        checker = SubprocessChecker(
            name="echo",
            description="Echo test",
            command=["echo", "hello"],
        )
        result = checker.run()
        assert result.name == "echo"
        assert result.status == "passed"
        assert result.duration_ms >= 0
        assert "hello" in result.output

    def test_failing_command(self) -> None:
        checker = SubprocessChecker(
            name="false",
            description="Always fails",
            command=["false"],
        )
        result = checker.run()
        assert result.name == "false"
        assert result.status == "failed"

    def test_command_with_stderr(self) -> None:
        checker = SubprocessChecker(
            name="stderr",
            description="Writes to stderr",
            command=["bash", "-c", "echo error >&2"],
        )
        result = checker.run()
        assert "error" in result.output

    def test_command_with_stdout_and_stderr(self) -> None:
        checker = SubprocessChecker(
            name="both",
            description="Writes to both",
            command=["bash", "-c", "echo out; echo err >&2"],
        )
        result = checker.run()
        assert "out" in result.output
        assert "err" in result.output

    def test_parser_extracts_diagnostics(self) -> None:
        def parse_output(output: str, code: int) -> tuple[Diagnostic, ...]:
            return (Diagnostic(message=f"Parsed: {output.strip()}"),)

        checker = SubprocessChecker(
            name="parsed",
            description="Uses parser",
            command=["echo", "test message"],
            parser=parse_output,
        )
        result = checker.run()
        assert len(result.diagnostics) == 1
        assert "Parsed: test message" in result.diagnostics[0].message

    def test_timeout_handling(self) -> None:
        checker = SubprocessChecker(
            name="slow",
            description="Times out",
            command=["sleep", "10"],
            timeout=1,
        )
        result = checker.run()
        assert result.status == "failed"
        assert any("Timed out" in d.message for d in result.diagnostics)

    def test_command_not_found(self) -> None:
        checker = SubprocessChecker(
            name="missing",
            description="Command not found",
            command=["nonexistent_command_12345"],
        )
        result = checker.run()
        assert result.status == "failed"
        assert any("Command not found" in d.message for d in result.diagnostics)

    def test_custom_environment(self) -> None:
        checker = SubprocessChecker(
            name="env",
            description="Custom env",
            command=["bash", "-c", "echo $MY_VAR"],
            env={"MY_VAR": "custom_value"},
        )
        result = checker.run()
        assert "custom_value" in result.output

    def test_name_and_description_properties(self) -> None:
        checker = SubprocessChecker(
            name="my-checker",
            description="My checker description",
            command=["true"],
        )
        assert checker.name == "my-checker"
        assert checker.description == "My checker description"

    def test_output_stripped(self) -> None:
        checker = SubprocessChecker(
            name="whitespace",
            description="Test whitespace",
            command=["echo", "  hello  "],
        )
        result = checker.run()
        # Output should be stripped
        assert result.output == "hello"
