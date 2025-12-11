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

"""Tests for Claude Agent SDK adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from weakincentives.adapters import CLAUDE_AGENT_SDK_ADAPTER_NAME, PromptEvaluationError


@dataclass(frozen=True, slots=True)
class _NoParams:
    """Empty dataclass for sections that don't need params."""


from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentSDKModelConfig,
)
from weakincentives.adapters.claude_agent_sdk._errors import normalize_sdk_error
from weakincentives.adapters.shared import ThrottleError
from weakincentives.deadlines import Deadline
from weakincentives.prompt import MarkdownSection, PromptTemplate
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session


class TestClaudeAgentSDKAdapterName:
    """Tests for adapter name constant."""

    def test_adapter_name_value(self) -> None:
        """Test adapter name has expected value."""
        assert CLAUDE_AGENT_SDK_ADAPTER_NAME == "claude_agent_sdk"


class TestClaudeAgentSDKAdapterInit:
    """Tests for ClaudeAgentSDKAdapter initialization."""

    def test_default_initialization(self) -> None:
        """Test adapter initializes with default values."""
        adapter = ClaudeAgentSDKAdapter()
        assert adapter._model == "claude-sonnet-4-5-20250929"
        assert adapter._client_config is not None
        assert adapter._allowed_tools is None
        assert adapter._disallowed_tools == ()

    def test_custom_model(self) -> None:
        """Test adapter with custom model."""
        adapter = ClaudeAgentSDKAdapter(model="claude-opus-4-20250514")
        assert adapter._model == "claude-opus-4-20250514"

    def test_custom_client_config(self) -> None:
        """Test adapter with custom client config."""
        config = ClaudeAgentSDKClientConfig(
            permission_mode="acceptEdits",
            cwd="/home/user/project",
        )
        adapter = ClaudeAgentSDKAdapter(client_config=config)
        assert adapter._client_config == config

    def test_custom_model_config(self) -> None:
        """Test adapter with custom model config."""
        config = ClaudeAgentSDKModelConfig(temperature=0.5)
        adapter = ClaudeAgentSDKAdapter(model_config=config)
        assert adapter._model_config == config

    def test_tool_filtering(self) -> None:
        """Test adapter with tool filtering."""
        adapter = ClaudeAgentSDKAdapter(
            allowed_tools=("Read", "Write"),
            disallowed_tools=("Bash",),
        )
        assert adapter._allowed_tools == ("Read", "Write")
        assert adapter._disallowed_tools == ("Bash",)


class TestClaudeAgentSDKAdapterEvaluate:
    """Tests for ClaudeAgentSDKAdapter.evaluate method."""

    def test_rejects_expired_deadline(self) -> None:
        """Test that evaluate rejects expired deadlines."""
        import time

        adapter = ClaudeAgentSDKAdapter()
        bus = InProcessEventBus()
        session = Session(bus=bus)

        # Create deadline that will expire in 1.1 seconds
        expiring_deadline = Deadline(
            expires_at=datetime.now(UTC) + timedelta(seconds=1.1)
        )

        @dataclass(slots=True, frozen=True)
        class DummyOutput:
            text: str

        template = PromptTemplate[DummyOutput](
            ns="test",
            key="expired-deadline",
            name="test_prompt",
            sections=[
                MarkdownSection[_NoParams](
                    title="Task",
                    key="task",
                    template="Do something",
                ),
            ],
        )

        from weakincentives.prompt import Prompt

        prompt = Prompt(template)

        # Wait for deadline to expire
        time.sleep(1.2)

        with pytest.raises(PromptEvaluationError) as exc_info:
            adapter.evaluate(prompt, session=session, deadline=expiring_deadline)

        assert "Deadline expired" in str(exc_info.value)

    def test_raises_import_error_when_sdk_not_installed(self) -> None:
        """Test that evaluate raises helpful error when SDK not installed."""
        import sys
        from unittest.mock import patch

        adapter = ClaudeAgentSDKAdapter()
        bus = InProcessEventBus()
        session = Session(bus=bus)

        @dataclass(slots=True, frozen=True)
        class DummyOutput:
            text: str

        template = PromptTemplate[DummyOutput](
            ns="test",
            key="import-error",
            name="test_prompt",
            sections=[
                MarkdownSection[_NoParams](
                    title="Task",
                    key="task",
                    template="Do something",
                ),
            ],
        )

        from weakincentives.prompt import Prompt

        prompt = Prompt(template)

        # Mock the import to raise ImportError
        with patch.dict(sys.modules, {"claude_agent_sdk": None}):
            # Force re-import by removing cached import
            original_import = __builtins__["__import__"]  # type: ignore[index]

            def mock_import(name: str, *args: object, **kwargs: object) -> object:
                if name == "claude_agent_sdk" or name.startswith("claude_agent_sdk."):
                    msg = "No module named 'claude_agent_sdk'"
                    raise ImportError(msg)
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises(PromptEvaluationError) as exc_info:
                    adapter.evaluate(prompt, session=session)

        # Should mention how to install
        error_msg = str(exc_info.value)
        assert "not installed" in error_msg or "not found" in error_msg.lower()


class TestErrorNormalization:
    """Tests for error normalization functions."""

    def test_normalize_cli_not_found_error(self) -> None:
        """Test normalizing CLINotFoundError."""

        class CLINotFoundError(Exception):
            pass

        error = CLINotFoundError("CLI not found")
        result = normalize_sdk_error(error, prompt_name="test")

        assert isinstance(result, PromptEvaluationError)
        assert "not found" in result.message.lower()
        assert result.prompt_name == "test"
        assert result.phase == "request"

    def test_normalize_cli_connection_error(self) -> None:
        """Test normalizing CLIConnectionError."""

        class CLIConnectionError(Exception):
            pass

        error = CLIConnectionError("Connection failed")
        result = normalize_sdk_error(error, prompt_name="test")

        assert isinstance(result, ThrottleError)
        assert result.kind == "timeout"
        assert result.retry_safe is True

    def test_normalize_process_error(self) -> None:
        """Test normalizing ProcessError."""

        class ProcessError(Exception):
            def __init__(self) -> None:
                super().__init__()
                self.exit_code = 1
                self.stderr = "Error occurred"

        error = ProcessError()
        result = normalize_sdk_error(error, prompt_name="test")

        assert isinstance(result, PromptEvaluationError)
        assert "process failed" in result.message.lower()
        assert result.provider_payload is not None
        assert result.provider_payload["exit_code"] == 1
        assert result.provider_payload["stderr"] == "Error occurred"

    def test_normalize_json_decode_error(self) -> None:
        """Test normalizing CLIJSONDecodeError."""

        class CLIJSONDecodeError(Exception):
            pass

        error = CLIJSONDecodeError("Invalid JSON")
        result = normalize_sdk_error(error, prompt_name="test")

        assert isinstance(result, PromptEvaluationError)
        assert "parse" in result.message.lower()
        assert result.phase == "response"

    def test_normalize_unknown_error(self) -> None:
        """Test normalizing unknown error types."""
        error = RuntimeError("Unknown error")
        result = normalize_sdk_error(error, prompt_name="test")

        assert isinstance(result, PromptEvaluationError)
        assert "Unknown error" in result.message


class TestClaudeAgentSDKAdapterIntegration:
    """Integration tests for ClaudeAgentSDKAdapter."""

    @pytest.mark.skip(reason="Requires claude-agent-sdk installation")
    def test_basic_evaluation(self) -> None:
        """Test basic prompt evaluation with SDK."""
        adapter = ClaudeAgentSDKAdapter(
            model="claude-sonnet-4-5-20250929",
            client_config=ClaudeAgentSDKClientConfig(
                permission_mode="bypassPermissions",
            ),
        )
        bus = InProcessEventBus()
        session = Session(bus=bus)

        @dataclass(slots=True, frozen=True)
        class Response:
            answer: str

        template = PromptTemplate[Response](
            ns="test",
            key="basic",
            name="test_prompt",
            sections=[
                MarkdownSection[_NoParams](
                    title="Task",
                    key="task",
                    template="Say hello world",
                ),
            ],
        )

        from weakincentives.prompt import Prompt

        prompt = Prompt(template)
        response = adapter.evaluate(prompt, session=session)

        assert response.prompt_name == "test_prompt"
        assert response.text is not None
