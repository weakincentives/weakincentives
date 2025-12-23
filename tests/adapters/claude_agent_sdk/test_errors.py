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

from weakincentives.adapters.claude_agent_sdk._errors import normalize_sdk_error
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.adapters.exceptions import ClaudeAgentSDKError
from weakincentives.adapters.throttle import ThrottleError


class MockCLINotFoundError(Exception):
    pass


class MockCLIConnectionError(Exception):
    pass


class MockProcessError(Exception):
    def __init__(self, message: str, exit_code: int, stderr: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


class MockCLIJSONDecodeError(Exception):
    pass


class MockMaxTurnsExceededError(Exception):
    pass


MockCLINotFoundError.__name__ = "CLINotFoundError"
MockCLIConnectionError.__name__ = "CLIConnectionError"
MockProcessError.__name__ = "ProcessError"
MockCLIJSONDecodeError.__name__ = "CLIJSONDecodeError"
MockMaxTurnsExceededError.__name__ = "MaxTurnsExceededError"


class TestNormalizeSDKError:
    def test_cli_not_found_error(self) -> None:
        error = MockCLINotFoundError("CLI not found")
        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, ClaudeAgentSDKError)
        assert isinstance(result, PromptEvaluationError)
        assert "Claude Code CLI not found" in result.message
        assert "npm install" in result.message
        assert result.prompt_name == "test_prompt"
        assert result.phase == "request"
        assert result.error_type == "CLINotFoundError"

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

        assert isinstance(result, ClaudeAgentSDKError)
        assert isinstance(result, PromptEvaluationError)
        assert "Claude Code process failed" in result.message
        assert result.prompt_name == "test_prompt"
        assert result.phase == "request"
        assert result.error_type == "ProcessError"
        assert result.provider_payload is not None
        assert result.provider_payload["exit_code"] == 1
        assert result.provider_payload["stderr"] == "Error output"

    def test_json_decode_error(self) -> None:
        error = MockCLIJSONDecodeError("Invalid JSON")
        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, ClaudeAgentSDKError)
        assert isinstance(result, PromptEvaluationError)
        assert "Failed to parse SDK response" in result.message
        assert result.prompt_name == "test_prompt"
        assert result.phase == "response"
        assert result.error_type == "CLIJSONDecodeError"

    def test_max_turns_exceeded_error(self) -> None:
        error = MockMaxTurnsExceededError("Exceeded 10 turns")
        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, ClaudeAgentSDKError)
        assert isinstance(result, PromptEvaluationError)
        assert "exceeded maximum turns" in result.message
        assert result.prompt_name == "test_prompt"
        assert result.phase == "response"
        assert result.error_type == "MaxTurnsExceededError"

    def test_generic_sdk_error(self) -> None:
        class MockSDKError(Exception):
            pass

        MockSDKError.__module__ = "claude_agent_sdk.errors"

        error = MockSDKError("Some SDK error")
        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, ClaudeAgentSDKError)
        assert isinstance(result, PromptEvaluationError)
        assert "Claude Agent SDK error" in result.message
        assert result.prompt_name == "test_prompt"
        assert result.phase == "request"
        assert result.error_type == "MockSDKError"

    def test_claude_code_module_error(self) -> None:
        class MockCodeError(Exception):
            pass

        MockCodeError.__module__ = "claude_code.client"

        error = MockCodeError("Code client error")
        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, ClaudeAgentSDKError)
        assert isinstance(result, PromptEvaluationError)
        assert "Claude Agent SDK error" in result.message
        assert result.error_type == "MockCodeError"

    def test_unknown_error(self) -> None:
        error = RuntimeError("Something went wrong")
        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, ClaudeAgentSDKError)
        assert isinstance(result, PromptEvaluationError)
        assert result.message == "Something went wrong"
        assert result.prompt_name == "test_prompt"
        assert result.phase == "request"
        assert result.error_type is None

    def test_process_error_without_attributes(self) -> None:
        class MinimalProcessError(Exception):
            pass

        MinimalProcessError.__name__ = "ProcessError"
        error = MinimalProcessError("Minimal error")
        result = normalize_sdk_error(error, "test_prompt")

        assert isinstance(result, ClaudeAgentSDKError)
        assert isinstance(result, PromptEvaluationError)
        assert "Claude Code process failed" in result.message
        assert result.error_type == "ProcessError"
        assert result.provider_payload is None

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

        class MockProcessErrorNoStderr(Exception):
            def __init__(self, message: str, exit_code: int) -> None:
                super().__init__(message)
                self.exit_code = exit_code

        MockProcessErrorNoStderr.__name__ = "ProcessError"
        error = MockProcessErrorNoStderr("Process failed", exit_code=1)
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
