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

"""Mock exception classes for Claude Agent SDK error testing.

These mocks simulate the exception types from the Claude Agent SDK
without requiring the actual SDK to be installed. They use __name__
manipulation to match the real exception class names that the error
normalization code checks.

Example::

    from tests.adapters.claude_agent_sdk.error_mocks import (
        MockCLINotFoundError,
        MockProcessError,
    )

    def test_cli_not_found_handling() -> None:
        error = MockCLINotFoundError("CLI not found")
        result = normalize_sdk_error(error, "test_prompt")
        assert "CLI not found" in result.message

"""

from __future__ import annotations


class MockCLINotFoundError(Exception):
    """Mock for CLINotFoundError from the Claude Agent SDK."""

    pass


class MockCLIConnectionError(Exception):
    """Mock for CLIConnectionError from the Claude Agent SDK."""

    pass


class MockProcessError(Exception):
    """Mock for ProcessError from the Claude Agent SDK.

    Includes exit_code and stderr attributes that the real error has.
    """

    def __init__(self, message: str, exit_code: int, stderr: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


class MockProcessErrorMinimal(Exception):
    """Mock for ProcessError without optional attributes.

    Used to test handling when stderr attribute is missing.
    """

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class MockCLIJSONDecodeError(Exception):
    """Mock for CLIJSONDecodeError from the Claude Agent SDK."""

    pass


class MockMaxTurnsExceededError(Exception):
    """Mock for MaxTurnsExceededError from the Claude Agent SDK."""

    pass


class MockSDKError(Exception):
    """Mock for generic SDK errors from claude_agent_sdk.errors module."""

    pass


class MockCodeError(Exception):
    """Mock for errors from claude_code.client module."""

    pass


# Set __name__ attributes to match what the error normalization code checks
MockCLINotFoundError.__name__ = "CLINotFoundError"
MockCLIConnectionError.__name__ = "CLIConnectionError"
MockProcessError.__name__ = "ProcessError"
MockProcessErrorMinimal.__name__ = "ProcessError"
MockCLIJSONDecodeError.__name__ = "CLIJSONDecodeError"
MockMaxTurnsExceededError.__name__ = "MaxTurnsExceededError"

# Set __module__ attributes for module-based error detection
MockSDKError.__module__ = "claude_agent_sdk.errors"
MockCodeError.__module__ = "claude_code.client"


__all__ = [
    "MockCLIConnectionError",
    "MockCLIJSONDecodeError",
    "MockCLINotFoundError",
    "MockCodeError",
    "MockMaxTurnsExceededError",
    "MockProcessError",
    "MockProcessErrorMinimal",
    "MockSDKError",
]
