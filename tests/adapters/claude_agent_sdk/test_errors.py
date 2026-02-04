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

"""Tests for Claude Agent SDK error normalization."""

from __future__ import annotations

from tests.adapters.claude_agent_sdk.error_mocks import (
    MockCLIConnectionError,
    MockCLIJSONDecodeError,
    MockCLINotFoundError,
    MockCodeError,
    MockMaxTurnsExceededError,
    MockProcessError,
    MockProcessErrorMinimal,
    MockSDKError,
)
from weakincentives.adapters.claude_agent_sdk._errors import normalize_sdk_error
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.adapters.throttle import ThrottleError


class TestNormalizeSDKError:
    def test_cli_not_found_error(self) -> None:
        error = MockCLINotFoundError("CLI not found")
        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, PromptEvaluationError)
        assert "Claude Code CLI not found" in result.message
        assert "npm install" in result.message
        assert result.prompt_name == "test_prompt"
        assert result.phase == "request"

    def test_cli_connection_error(self) -> None:
        error = MockCLIConnectionError("Connection timed out")
        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, ThrottleError)
        assert result.prompt_name == "test_prompt"
        assert result.phase == "request"
        assert result.details.retry_safe is True

    def test_process_error_with_details(self) -> None:
        error = MockProcessError("Process failed", exit_code=1, stderr="Error output")
        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, PromptEvaluationError)
        assert "Claude Code process failed" in result.message
        assert result.prompt_name == "test_prompt"
        assert result.phase == "request"
        assert result.provider_payload is not None
        assert result.provider_payload["exit_code"] == 1
        assert result.provider_payload["stderr"] == "Error output"

    def test_json_decode_error(self) -> None:
        error = MockCLIJSONDecodeError("Invalid JSON")
        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, PromptEvaluationError)
        assert "Failed to parse SDK response" in result.message
        assert result.prompt_name == "test_prompt"
        assert result.phase == "response"

    def test_max_turns_exceeded_error(self) -> None:
        error = MockMaxTurnsExceededError("Exceeded 10 turns")
        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, PromptEvaluationError)
        assert "exceeded maximum turns" in result.message
        assert result.prompt_name == "test_prompt"
        assert result.phase == "response"

    def test_generic_sdk_error(self) -> None:
        error = MockSDKError("Some SDK error")
        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, PromptEvaluationError)
        assert "Claude Agent SDK error" in result.message
        assert result.prompt_name == "test_prompt"
        assert result.phase == "request"

    def test_claude_code_module_error(self) -> None:
        error = MockCodeError("Code client error")
        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, PromptEvaluationError)
        assert "Claude Agent SDK error" in result.message

    def test_unknown_error(self) -> None:
        error = RuntimeError("Something went wrong")
        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, PromptEvaluationError)
        assert result.message == "Something went wrong"
        assert result.prompt_name == "test_prompt"
        assert result.phase == "request"

    def test_process_error_with_captured_stderr(self) -> None:
        """ProcessError with stderr_output captures stderr in payload."""
        error = MockProcessError("Process failed", exit_code=1, stderr="Error output")
        result = normalize_sdk_error(
            error, "test_prompt", stderr_output="Captured stderr from process"
        )

        assert isinstance(result, PromptEvaluationError)
        assert result.provider_payload is not None
        assert result.provider_payload["exit_code"] == 1
        # Original stderr from error attribute
        assert result.provider_payload["stderr"] == "Error output"
        # Captured stderr is also included
        assert (
            result.provider_payload["stderr_captured"] == "Captured stderr from process"
        )

    def test_process_error_with_captured_stderr_no_error_stderr(self) -> None:
        """ProcessError with captured stderr but no error.stderr attribute."""
        error = MockProcessErrorMinimal("Process failed", exit_code=1)
        result = normalize_sdk_error(
            error, "test_prompt", stderr_output="Captured stderr from process"
        )

        assert isinstance(result, PromptEvaluationError)
        assert result.provider_payload is not None
        assert result.provider_payload["exit_code"] == 1
        # Captured stderr becomes the primary stderr
        assert result.provider_payload["stderr"] == "Captured stderr from process"
        assert (
            result.provider_payload["stderr_captured"] == "Captured stderr from process"
        )

    def test_json_decode_error_with_stderr(self) -> None:
        """JSONDecodeError with stderr_output includes stderr in payload."""
        error = MockCLIJSONDecodeError("Invalid JSON")
        result = normalize_sdk_error(
            error, "test_prompt", stderr_output="CLI output before crash"
        )

        assert isinstance(result, PromptEvaluationError)
        assert "Failed to parse SDK response" in result.message
        assert result.provider_payload is not None
        assert result.provider_payload["stderr"] == "CLI output before crash"

    def test_unknown_error_with_stderr(self) -> None:
        """Unknown error with stderr_output includes stderr in payload."""
        error = RuntimeError("Something went wrong")
        result = normalize_sdk_error(
            error, "test_prompt", stderr_output="Unexpected stderr output"
        )

        assert isinstance(result, PromptEvaluationError)
        assert result.provider_payload is not None
        assert result.provider_payload["stderr"] == "Unexpected stderr output"

    def test_exception_group_with_cli_connection_errors(self) -> None:
        """ExceptionGroup with CLIConnectionError during cleanup is handled."""
        # Create mock sub-exceptions
        sub_error1 = MockCLIConnectionError("ProcessTransport is not ready for writing")
        sub_error2 = MockCLIConnectionError("ProcessTransport is not ready for writing")

        # Create ExceptionGroup (Python 3.11+)
        error = ExceptionGroup(
            "unhandled errors in a TaskGroup (2 sub-exceptions)",
            [sub_error1, sub_error2],
        )

        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, PromptEvaluationError)
        assert "SDK cleanup error" in result.message
        assert "Transport closed while processing" in result.message
        assert "2 pending control requests" in result.message
        assert result.phase == "response"
        assert result.prompt_name == "test_prompt"
        # Check that provider_payload is None when no stderr_output provided
        assert result.provider_payload is None

    def test_exception_group_with_mixed_errors(self) -> None:
        """ExceptionGroup with mixed error types is handled generically."""
        # Create mock sub-exceptions of different types
        sub_error1 = MockCLIConnectionError("Connection error")
        sub_error2 = RuntimeError("Some other error")

        # Create ExceptionGroup
        error = ExceptionGroup(
            "Multiple errors occurred",
            [sub_error1, sub_error2],
        )

        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, PromptEvaluationError)
        # The error message includes the sub-exception count
        assert result.message == "Multiple errors occurred (2 sub-exceptions)"
        assert result.phase == "request"
        assert result.prompt_name == "test_prompt"
        assert result.provider_payload is not None
        assert "sub_exceptions" in result.provider_payload
        assert len(result.provider_payload["sub_exceptions"]) == 2
        assert (
            result.provider_payload["sub_exceptions"][0]["type"] == "CLIConnectionError"
        )
        assert result.provider_payload["sub_exceptions"][1]["type"] == "RuntimeError"

    def test_exception_group_without_exceptions_attribute(self) -> None:
        """ExceptionGroup without exceptions attribute is handled as generic error."""

        # Create a mock ExceptionGroup without the exceptions attribute
        class MockExceptionGroup(Exception):
            pass

        MockExceptionGroup.__name__ = "ExceptionGroup"
        error = MockExceptionGroup("Some exception group")

        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, PromptEvaluationError)
        # Empty sub_exceptions should be treated as generic exception group,
        # not cleanup errors (vacuous truth protection)
        assert result.message == "Some exception group"  # Uses str(error)
        assert "SDK cleanup error" not in result.message  # NOT a cleanup error
        assert result.phase == "request"  # Generic errors use "request" phase
        # provider_payload is None when sub_exceptions is empty and no stderr
        assert result.provider_payload is None

    def test_exception_group_with_stderr_output(self) -> None:
        """ExceptionGroup includes stderr_output in payload."""
        sub_error = MockCLIConnectionError("ProcessTransport is not ready for writing")
        error = ExceptionGroup(
            "unhandled errors in a TaskGroup (1 sub-exception)",
            [sub_error],
        )

        result = normalize_sdk_error(
            error, "test_prompt", stderr_output="Debug output from SDK"
        )

        assert isinstance(result, PromptEvaluationError)
        assert "SDK cleanup error" in result.message
        assert result.phase == "response"
        assert result.provider_payload is not None
        assert result.provider_payload.get("stderr") == "Debug output from SDK"
        assert result.provider_payload["error_type"] == "TaskGroupCleanupError"
        assert result.provider_payload["sub_exception_count"] == 1
