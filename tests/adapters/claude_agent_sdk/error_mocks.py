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

These mocks inherit from the actual SDK exception types to work with
isinstance checks in the error normalization code.

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

from claude_agent_sdk import (
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
)


class MockCLINotFoundError(CLINotFoundError):
    """Mock for CLINotFoundError from the Claude Agent SDK."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message)


class MockCLIConnectionError(CLIConnectionError):
    """Mock for CLIConnectionError from the Claude Agent SDK."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class MockProcessError(ProcessError):
    """Mock for ProcessError from the Claude Agent SDK.

    Includes exit_code and stderr attributes that the real error has.
    """

    def __init__(self, message: str, exit_code: int, stderr: str) -> None:
        # ProcessError takes (message, exit_code, stderr) in constructor
        super().__init__(message, exit_code, stderr)


class MockProcessErrorMinimal(ProcessError):
    """Mock for ProcessError without optional stderr attribute.

    Used to test handling when stderr attribute is missing.
    """

    def __init__(self, message: str, exit_code: int) -> None:
        # ProcessError without stderr
        super().__init__(message, exit_code, None)
        # Remove stderr attribute to simulate minimal error
        if hasattr(self, "stderr"):
            del self.stderr


class MockCLIJSONDecodeError(CLIJSONDecodeError):
    """Mock for CLIJSONDecodeError from the Claude Agent SDK."""

    def __init__(self, message: str) -> None:
        # CLIJSONDecodeError takes (line, original_error)
        super().__init__(message, ValueError(message))


class MockMaxTurnsExceededError(Exception):
    """Mock for MaxTurnsExceededError from the Claude Agent SDK.

    Note: MaxTurnsExceededError is not exported by the SDK, so we use
    name-based matching for this error type.
    """

    pass


class MockSDKError(Exception):
    """Mock for generic SDK errors from claude_agent_sdk.errors module."""

    pass


class MockCodeError(Exception):
    """Mock for errors from claude_code.client module."""

    pass


# Set __name__ to match SDK class names (for error message formatting)
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
