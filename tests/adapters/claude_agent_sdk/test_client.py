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

"""Tests for Claude Agent SDK client wrapper."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest

from tests.helpers import FrozenUtcNow
from weakincentives.adapters.claude_agent_sdk._client import (
    ClientConfig,
    ClientSession,
    QueryResult,
    _build_options_kwargs,
)
from weakincentives.adapters.claude_agent_sdk._hooks import HookContext
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session


@dataclass
class MockResultMessage:
    """Mock ResultMessage for testing."""

    result: str | None = None
    structured_output: dict[str, object] | None = None
    usage: dict[str, int] | None = None
    stop_reason: str | None = None


@dataclass
class MockAssistantMessage:
    """Mock AssistantMessage for testing."""

    content: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None


class MockClaudeAgentOptions:
    """Mock ClaudeAgentOptions for testing."""

    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockClaudeSDKClient:
    """Mock ClaudeSDKClient for testing."""

    _connected: bool = False
    _query_messages: ClassVar[list[Any]] = []
    _query_error: ClassVar[Exception | None] = None
    _interrupt_called: bool = False
    _disconnect_called: bool = False
    _query_prompt: str | None = None
    _query_session_id: str | None = None

    def __init__(self, options: MockClaudeAgentOptions | None = None) -> None:
        self.options = options
        self._connected = False
        self._interrupt_called = False
        self._disconnect_called = False

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._disconnect_called = True
        self._connected = False

    async def query(self, prompt: str, session_id: str = "default") -> None:
        self._query_prompt = prompt
        self._query_session_id = session_id

    async def receive_response(self) -> Any:  # noqa: ANN401
        if MockClaudeSDKClient._query_error:
            raise MockClaudeSDKClient._query_error

        for message in MockClaudeSDKClient._query_messages:
            yield message

    async def interrupt(self) -> None:
        self._interrupt_called = True

    @classmethod
    def reset(cls) -> None:
        """Reset class state between tests."""
        cls._query_messages = []
        cls._query_error = None

    @classmethod
    def set_messages(cls, messages: list[Any]) -> None:
        """Set messages to yield from receive_response."""
        cls._query_messages = messages

    @classmethod
    def set_error(cls, error: Exception) -> None:
        """Set an error to raise during receive_response."""
        cls._query_error = error


@dataclass
class MockHookMatcher:
    """Mock HookMatcher for testing."""

    matcher: str | None = None
    hooks: list[Any] | None = None


@pytest.fixture
def session() -> Session:
    bus = InProcessEventBus()
    return Session(bus=bus)


@pytest.fixture
def hook_context(session: Session) -> HookContext:
    return HookContext(
        session=session,
        adapter_name="claude_agent_sdk",
        prompt_name="test:prompt",
        deadline=None,
        budget_tracker=None,
    )


@pytest.fixture
def client_config() -> ClientConfig:
    return ClientConfig(
        model="claude-sonnet-4-5-20250929",
        cwd="/test/path",
        permission_mode="bypassPermissions",
    )


@contextmanager
def mock_sdk_imports() -> Generator[MagicMock, None, None]:
    """Mock all SDK imports needed for client tests."""
    mock_sdk = MagicMock()
    mock_sdk.ClaudeSDKClient = MockClaudeSDKClient
    mock_sdk.types = MagicMock()
    mock_sdk.types.ClaudeAgentOptions = MockClaudeAgentOptions
    mock_sdk.types.ResultMessage = MockResultMessage
    mock_sdk.types.HookMatcher = MockHookMatcher

    with patch(
        "weakincentives.adapters.claude_agent_sdk._client._import_sdk",
        return_value=mock_sdk,
    ):
        yield mock_sdk


class TestClientConfig:
    def test_default_values(self) -> None:
        config = ClientConfig(model="claude-sonnet-4-5-20250929")

        assert config.model == "claude-sonnet-4-5-20250929"
        assert config.cwd is None
        assert config.permission_mode == "bypassPermissions"
        assert config.max_turns is None
        assert config.output_format is None
        assert config.allowed_tools is None
        assert config.disallowed_tools == ()
        assert config.suppress_stderr is True

    def test_custom_values(self) -> None:
        config = ClientConfig(
            model="claude-opus-4-5-20250929",
            cwd="/custom/path",
            permission_mode="acceptEdits",
            max_turns=10,
            output_format={"type": "json_schema"},
            allowed_tools=("Read", "Write"),
            disallowed_tools=("Bash",),
            suppress_stderr=False,
        )

        assert config.model == "claude-opus-4-5-20250929"
        assert config.cwd == "/custom/path"
        assert config.permission_mode == "acceptEdits"
        assert config.max_turns == 10
        assert config.output_format == {"type": "json_schema"}
        assert config.allowed_tools == ("Read", "Write")
        assert config.disallowed_tools == ("Bash",)
        assert config.suppress_stderr is False


class TestQueryResult:
    def test_default_values(self) -> None:
        result = QueryResult(messages=[])

        assert result.messages == []
        assert result.result_text is None
        assert result.structured_output is None
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.stop_reason is None
        assert result.interrupted is False

    def test_custom_values(self) -> None:
        messages = [MockResultMessage(result="Hello")]
        result = QueryResult(
            messages=messages,
            result_text="Hello",
            structured_output={"message": "structured"},
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
            interrupted=False,
        )

        assert result.messages == messages
        assert result.result_text == "Hello"
        assert result.structured_output == {"message": "structured"}
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.stop_reason == "end_turn"
        assert result.interrupted is False


class TestClientSessionLifecycle:
    def test_connect_creates_client(
        self, client_config: ClientConfig, hook_context: HookContext
    ) -> None:
        async def run_test() -> None:
            MockClaudeSDKClient.reset()

            with mock_sdk_imports():
                client_session = ClientSession(client_config, hook_context)
                await client_session.connect()

                assert client_session._connected is True
                assert client_session._client is not None
                assert isinstance(client_session._client, MockClaudeSDKClient)
                assert client_session._client._connected is True

                await client_session.disconnect()

        asyncio.run(run_test())

    def test_disconnect_cleans_up(
        self, client_config: ClientConfig, hook_context: HookContext
    ) -> None:
        async def run_test() -> None:
            MockClaudeSDKClient.reset()

            with mock_sdk_imports():
                client_session = ClientSession(client_config, hook_context)
                await client_session.connect()
                await client_session.disconnect()

                assert client_session._connected is False
                assert client_session._client is None

        asyncio.run(run_test())

    def test_context_manager(
        self, client_config: ClientConfig, hook_context: HookContext
    ) -> None:
        async def run_test() -> ClientSession:
            MockClaudeSDKClient.reset()

            with mock_sdk_imports():
                async with ClientSession(client_config, hook_context) as client_session:
                    assert client_session._connected is True
                    return client_session

        session = asyncio.run(run_test())
        assert session._connected is False

    def test_multiple_connect_calls_idempotent(
        self, client_config: ClientConfig, hook_context: HookContext
    ) -> None:
        async def run_test() -> None:
            MockClaudeSDKClient.reset()

            with mock_sdk_imports():
                client_session = ClientSession(client_config, hook_context)
                await client_session.connect()
                original_client = client_session._client

                # Second connect should be idempotent
                await client_session.connect()
                assert client_session._client is original_client

                await client_session.disconnect()

        asyncio.run(run_test())


class TestClientSessionQuery:
    def test_basic_query(
        self, client_config: ClientConfig, hook_context: HookContext
    ) -> None:
        async def run_test() -> QueryResult:
            MockClaudeSDKClient.reset()
            MockClaudeSDKClient.set_messages(
                [
                    MockResultMessage(
                        result="Hello!",
                        usage={"input_tokens": 10, "output_tokens": 5},
                        stop_reason="end_turn",
                    )
                ]
            )

            with mock_sdk_imports():
                async with ClientSession(client_config, hook_context) as client:
                    return await client.query("Say hello")

        result = asyncio.run(run_test())
        assert result.result_text == "Hello!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.stop_reason == "end_turn"
        assert result.interrupted is False

    def test_query_without_connect_raises(
        self, client_config: ClientConfig, hook_context: HookContext
    ) -> None:
        async def run_test() -> None:
            MockClaudeSDKClient.reset()

            with mock_sdk_imports():
                client_session = ClientSession(client_config, hook_context)

                with pytest.raises(RuntimeError, match="Client not connected"):
                    await client_session.query("Hello")

        asyncio.run(run_test())

    def test_query_extracts_structured_output(
        self, client_config: ClientConfig, hook_context: HookContext
    ) -> None:
        async def run_test() -> QueryResult:
            MockClaudeSDKClient.reset()
            MockClaudeSDKClient.set_messages(
                [
                    MockResultMessage(
                        result="Response",
                        structured_output={"message": "structured"},
                        usage={"input_tokens": 10},
                    )
                ]
            )

            with mock_sdk_imports():
                async with ClientSession(client_config, hook_context) as client:
                    return await client.query("Get structured data")

        result = asyncio.run(run_test())
        assert result.structured_output == {"message": "structured"}

    def test_query_accumulates_usage(
        self, client_config: ClientConfig, hook_context: HookContext
    ) -> None:
        async def run_test() -> QueryResult:
            MockClaudeSDKClient.reset()
            MockClaudeSDKClient.set_messages(
                [
                    MockAssistantMessage(
                        usage={"input_tokens": 50, "output_tokens": 25}
                    ),
                    MockResultMessage(
                        result="Done",
                        usage={"input_tokens": 50, "output_tokens": 25},
                    ),
                ]
            )

            with mock_sdk_imports():
                async with ClientSession(client_config, hook_context) as client:
                    return await client.query("Multi-message query")

        result = asyncio.run(run_test())
        assert result.input_tokens == 100
        assert result.output_tokens == 50


class TestClientSessionInterrupt:
    def test_interrupt_sets_flag(
        self, client_config: ClientConfig, hook_context: HookContext
    ) -> None:
        async def run_test() -> bool:
            MockClaudeSDKClient.reset()
            MockClaudeSDKClient.set_messages([])

            with mock_sdk_imports():
                async with ClientSession(client_config, hook_context) as client:
                    await client.interrupt()
                    return client._interrupt_requested

        result = asyncio.run(run_test())
        assert result is True

    def test_interrupt_triggers_should_interrupt(
        self, client_config: ClientConfig, hook_context: HookContext
    ) -> None:
        """Test that _should_interrupt returns True when interrupt_requested is set."""
        MockClaudeSDKClient.reset()

        with mock_sdk_imports():
            client_session = ClientSession(client_config, hook_context)
            client_session._interrupt_requested = True

            # _should_interrupt should return True due to interrupt flag
            assert client_session._should_interrupt() is True

    def test_deadline_triggers_interrupt(
        self,
        client_config: ClientConfig,
        session: Session,
        frozen_utcnow: FrozenUtcNow,
    ) -> None:
        anchor = datetime.now(UTC)
        frozen_utcnow.set(anchor)
        deadline = Deadline(anchor + timedelta(seconds=5))
        frozen_utcnow.advance(timedelta(seconds=10))  # Past deadline

        hook_context = HookContext(
            session=session,
            adapter_name="claude_agent_sdk",
            prompt_name="test:prompt",
            deadline=deadline,
            budget_tracker=None,
        )

        MockClaudeSDKClient.reset()
        MockClaudeSDKClient.set_messages([MockResultMessage(result="Should not reach")])

        with mock_sdk_imports():
            client_session = ClientSession(
                client_config,
                hook_context,
                deadline=deadline,
            )

            # _should_interrupt should return True due to expired deadline
            assert client_session._should_interrupt() is True

    def test_budget_triggers_interrupt(
        self, client_config: ClientConfig, session: Session
    ) -> None:
        budget = Budget(max_total_tokens=100)
        budget_tracker = BudgetTracker(budget)
        # Consume the entire budget
        from weakincentives.runtime.events._types import TokenUsage

        budget_tracker.record_cumulative(
            "test",
            TokenUsage(input_tokens=60, output_tokens=50, cached_tokens=None),
        )

        hook_context = HookContext(
            session=session,
            adapter_name="claude_agent_sdk",
            prompt_name="test:prompt",
            deadline=None,
            budget_tracker=budget_tracker,
        )

        MockClaudeSDKClient.reset()

        with mock_sdk_imports():
            client_session = ClientSession(
                client_config,
                hook_context,
                budget_tracker=budget_tracker,
            )

            # _should_interrupt should return True due to exhausted budget
            assert client_session._should_interrupt() is True


class TestBuildOptionsKwargs:
    def test_builds_options_with_config(self) -> None:
        config = ClientConfig(
            model="claude-opus-4-5-20250929",
            cwd="/test/cwd",
            permission_mode="acceptEdits",
            max_turns=10,
            output_format={"type": "json_schema"},
            allowed_tools=("Read", "Write"),
            disallowed_tools=("Bash",),
            suppress_stderr=True,
            env={"HOME": "/tmp/test"},
            setting_sources=["user"],
        )

        options_kwargs = _build_options_kwargs(config)

        assert options_kwargs["model"] == "claude-opus-4-5-20250929"
        assert options_kwargs["cwd"] == "/test/cwd"
        assert options_kwargs["permission_mode"] == "acceptEdits"
        assert options_kwargs["max_turns"] == 10
        assert options_kwargs["output_format"] == {"type": "json_schema"}
        assert options_kwargs["allowed_tools"] == ["Read", "Write"]
        assert options_kwargs["disallowed_tools"] == ["Bash"]
        assert options_kwargs["env"] == {"HOME": "/tmp/test"}
        assert options_kwargs["setting_sources"] == ["user"]

    def test_builds_options_with_hooks(self) -> None:
        config = ClientConfig(
            model="claude-sonnet-4-5-20250929",
            hooks={"PreToolUse": [MockHookMatcher()]},
        )

        options_kwargs = _build_options_kwargs(config)

        assert options_kwargs["hooks"] is not None
        assert "PreToolUse" in options_kwargs["hooks"]

    def test_builds_options_with_mcp_servers(self) -> None:
        config = ClientConfig(
            model="claude-sonnet-4-5-20250929",
            mcp_servers={"wink": {"type": "sdk"}},
        )

        options_kwargs = _build_options_kwargs(config)

        assert options_kwargs["mcp_servers"] == {"wink": {"type": "sdk"}}
