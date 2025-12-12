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

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest

from tests.helpers import FrozenUtcNow
from weakincentives.adapters.claude_agent_sdk import (
    CLAUDE_AGENT_SDK_ADAPTER_NAME,
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentSDKModelConfig,
)
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.deadlines import Deadline
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime.events import (
    InProcessEventBus,
    PromptExecuted,
    PromptRendered,
)
from weakincentives.runtime.session import Session


# Create mock SDK types for testing
@dataclass
class MockResultMessage:
    """Mock ResultMessage for testing."""

    result: str | None = None
    structured_output: dict[str, object] | None = None
    usage: dict[str, int] | None = None


class MockClaudeAgentOptions:
    """Mock ClaudeAgentOptions for testing."""

    # Declare expected attributes for type checking
    model: str | None
    cwd: str | None
    permission_mode: str | None
    max_turns: int | None
    output_format: dict[str, object] | None
    allowed_tools: list[str] | None
    disallowed_tools: list[str] | None
    max_thinking_tokens: int | None
    mcp_servers: dict[str, object] | None
    hooks: dict[str, list[object]] | None

    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class MockHookMatcher:
    """Mock HookMatcher for testing."""

    matcher: str | None = None
    hooks: list[object] | None = None


class MockClaudeSDKClient:
    """Mock ClaudeSDKClient for testing."""

    captured_options: ClassVar[list[MockClaudeAgentOptions]] = []
    _results: ClassVar[list[object]] = []

    def __init__(self, options: MockClaudeAgentOptions | None = None) -> None:
        self.options = options
        if options is not None:
            MockClaudeSDKClient.captured_options.append(options)

    async def __aenter__(self) -> MockClaudeSDKClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        pass

    async def query(self, prompt: str) -> None:
        """Send a query - the client pattern sends and receives separately."""
        pass

    async def receive_messages(self) -> AsyncGenerator[object, None]:
        """Yield the configured results."""
        for result in MockClaudeSDKClient._results:
            yield result

    @classmethod
    def reset(cls) -> None:
        """Reset captured options and results between tests."""
        cls.captured_options = []
        cls._results = []

    @classmethod
    def set_results(cls, results: list[object]) -> None:
        """Set the results to return from receive_messages."""
        cls._results = results


@dataclass(slots=True, frozen=True)
class SimpleOutput:
    message: str


@pytest.fixture
def session() -> Session:
    bus = InProcessEventBus()
    return Session(bus=bus)


@pytest.fixture
def simple_prompt() -> Prompt[SimpleOutput]:
    template = PromptTemplate[SimpleOutput](
        ns="test",
        key="simple",
        sections=[
            MarkdownSection(
                title="Task",
                key="task",
                template="Say hello",
            ),
        ],
    )
    return Prompt(template)


@pytest.fixture
def untyped_prompt() -> Prompt[None]:
    template = PromptTemplate[None](
        ns="test",
        key="untyped",
        sections=[
            MarkdownSection(
                title="Task",
                key="task",
                template="Do something",
            ),
        ],
    )
    return Prompt(template)


class TestClaudeAgentSDKAdapterInit:
    def test_default_values(self) -> None:
        adapter = ClaudeAgentSDKAdapter()

        assert adapter._model == "claude-sonnet-4-5-20250929"
        assert adapter._client_config.permission_mode == "bypassPermissions"
        assert adapter._allowed_tools is None
        assert adapter._disallowed_tools == ()

    def test_custom_model(self) -> None:
        adapter = ClaudeAgentSDKAdapter(model="claude-opus-4-5-20250929")

        assert adapter._model == "claude-opus-4-5-20250929"

    def test_custom_client_config(self) -> None:
        config = ClaudeAgentSDKClientConfig(
            permission_mode="acceptEdits",
            cwd="/home/user",
            max_turns=5,
        )
        adapter = ClaudeAgentSDKAdapter(client_config=config)

        assert adapter._client_config.permission_mode == "acceptEdits"
        assert adapter._client_config.cwd == "/home/user"
        assert adapter._client_config.max_turns == 5

    def test_custom_model_config(self) -> None:
        config = ClaudeAgentSDKModelConfig(
            model="claude-opus-4-5-20250929",
            temperature=0.5,
            max_tokens=1000,
        )
        adapter = ClaudeAgentSDKAdapter(model_config=config)

        assert adapter._model_config.temperature == 0.5
        assert adapter._model_config.max_tokens == 1000

    def test_tool_filtering(self) -> None:
        adapter = ClaudeAgentSDKAdapter(
            allowed_tools=("Read", "Write"),
            disallowed_tools=("Bash",),
        )

        assert adapter._allowed_tools == ("Read", "Write")
        assert adapter._disallowed_tools == ("Bash",)


def _create_mock_client_class(
    results: list[object],
) -> type[MockClaudeSDKClient]:
    """Create a MockClaudeSDKClient class that yields specific results."""
    # Configure the class-level results
    MockClaudeSDKClient.set_results(results)
    return MockClaudeSDKClient


class TestClaudeAgentSDKAdapterEvaluate:
    def test_raises_when_deadline_expired(
        self,
        session: Session,
        simple_prompt: Prompt[SimpleOutput],
        frozen_utcnow: FrozenUtcNow,
    ) -> None:
        anchor = datetime.now(UTC)
        frozen_utcnow.set(anchor)
        deadline = Deadline(anchor + timedelta(seconds=5))
        frozen_utcnow.advance(timedelta(seconds=10))

        adapter = ClaudeAgentSDKAdapter()

        with pytest.raises(PromptEvaluationError, match="Deadline expired"):
            adapter.evaluate(simple_prompt, session=session, deadline=deadline)

    def test_raises_when_sdk_not_installed(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        adapter = ClaudeAgentSDKAdapter()

        with patch.dict("sys.modules", {"claude_agent_sdk": None}):
            with patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                side_effect=ImportError("claude-agent-sdk is not installed"),
            ):
                with pytest.raises(ImportError, match="claude-agent-sdk"):
                    adapter.evaluate(simple_prompt, session=session)

    def test_publishes_prompt_rendered_event(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        events: list[PromptRendered] = []
        session.event_bus.subscribe(PromptRendered, lambda e: events.append(e))

        MockClaudeSDKClient.reset()
        mock_client_class = _create_mock_client_class(
            [MockResultMessage(result="Hello!", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=MagicMock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                mock_client_class,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                MockClaudeAgentOptions,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
        ):
            adapter.evaluate(simple_prompt, session=session)

        assert len(events) == 1
        event = events[0]
        assert event.prompt_ns == "test"
        assert event.prompt_key == "simple"
        assert event.adapter == CLAUDE_AGENT_SDK_ADAPTER_NAME

    def test_returns_prompt_response(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        MockClaudeSDKClient.reset()
        mock_client_class = _create_mock_client_class(
            [
                MockResultMessage(
                    result="Hello, world!",
                    usage={"input_tokens": 10, "output_tokens": 5},
                    structured_output=None,
                )
            ]
        )

        adapter = ClaudeAgentSDKAdapter()

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=MagicMock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                mock_client_class,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                MockClaudeAgentOptions,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
        ):
            response = adapter.evaluate(simple_prompt, session=session)

        assert response.prompt_name == "test:simple"
        assert response.text == "Hello, world!"

    def test_extracts_structured_output(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        MockClaudeSDKClient.reset()
        mock_client_class = _create_mock_client_class(
            [
                MockResultMessage(
                    result="Hello!",
                    usage=None,
                    structured_output={"message": "structured hello"},
                )
            ]
        )

        adapter = ClaudeAgentSDKAdapter()

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=MagicMock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                mock_client_class,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                MockClaudeAgentOptions,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
        ):
            response = adapter.evaluate(simple_prompt, session=session)

        assert response.output is not None
        assert response.output.message == "structured hello"

    def test_handles_invalid_structured_output(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        MockClaudeSDKClient.reset()
        mock_client_class = _create_mock_client_class(
            [
                MockResultMessage(
                    result="Hello!",
                    usage=None,
                    # Invalid - not a dict at all
                    structured_output="not a dict",  # type: ignore[arg-type]
                )
            ]
        )

        adapter = ClaudeAgentSDKAdapter()

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=MagicMock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                mock_client_class,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                MockClaudeAgentOptions,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
        ):
            response = adapter.evaluate(simple_prompt, session=session)

        # Should gracefully handle parse error and return None output
        assert response.output is None
        assert response.text == "Hello!"

    def test_handles_no_structured_output(
        self, session: Session, untyped_prompt: Prompt[None]
    ) -> None:
        MockClaudeSDKClient.reset()
        mock_client_class = _create_mock_client_class(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=MagicMock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                mock_client_class,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                MockClaudeAgentOptions,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
        ):
            response = adapter.evaluate(untyped_prompt, session=session)

        assert response.output is None

    def test_accumulates_token_usage(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        # First message with usage (non-result)
        msg1 = MagicMock()
        msg1.usage = {"input_tokens": 100, "output_tokens": 50}

        MockClaudeSDKClient.reset()
        mock_client_class = _create_mock_client_class(
            [
                msg1,
                MockResultMessage(
                    result="Done",
                    usage={"input_tokens": 50, "output_tokens": 25},
                    structured_output=None,
                ),
            ]
        )

        events: list[PromptExecuted] = []
        session.event_bus.subscribe(PromptExecuted, lambda e: events.append(e))

        adapter = ClaudeAgentSDKAdapter()

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=MagicMock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                mock_client_class,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                MockClaudeAgentOptions,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
        ):
            adapter.evaluate(simple_prompt, session=session)

        assert len(events) == 1
        usage = events[0].usage
        assert usage is not None
        assert usage.input_tokens == 150
        assert usage.output_tokens == 75


class TestAdapterName:
    def test_adapter_name_constant(self) -> None:
        assert CLAUDE_AGENT_SDK_ADAPTER_NAME == "claude_agent_sdk"


class TestBuildOutputFormat:
    def test_none_output_type_returns_none(self, untyped_prompt: Prompt[None]) -> None:
        adapter = ClaudeAgentSDKAdapter()
        rendered = untyped_prompt.render()
        result = adapter._build_output_format(rendered)
        assert result is None

    def test_dataclass_output_type_returns_schema(
        self, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        adapter = ClaudeAgentSDKAdapter()
        rendered = simple_prompt.render()
        result = adapter._build_output_format(rendered)

        assert result is not None
        assert result["type"] == "json_schema"
        assert "schema" in result
        assert "properties" in result["schema"]
        assert "message" in result["schema"]["properties"]


class TestSDKConfigOptions:
    def test_passes_cwd_option(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        MockClaudeSDKClient.reset()
        mock_client_class = _create_mock_client_class(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(cwd="/home/user/project"),
        )

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=MagicMock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                mock_client_class,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                MockClaudeAgentOptions,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
        ):
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockClaudeSDKClient.captured_options) == 1
        assert MockClaudeSDKClient.captured_options[0].cwd == "/home/user/project"

    def test_passes_max_turns_option(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        MockClaudeSDKClient.reset()
        mock_client_class = _create_mock_client_class(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(max_turns=10),
        )

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=MagicMock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                mock_client_class,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                MockClaudeAgentOptions,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
        ):
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockClaudeSDKClient.captured_options) == 1
        assert MockClaudeSDKClient.captured_options[0].max_turns == 10

    def test_passes_allowed_tools_option(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        MockClaudeSDKClient.reset()
        mock_client_class = _create_mock_client_class(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(allowed_tools=("Read", "Write"))

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=MagicMock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                mock_client_class,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                MockClaudeAgentOptions,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
        ):
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockClaudeSDKClient.captured_options) == 1
        assert MockClaudeSDKClient.captured_options[0].allowed_tools == [
            "Read",
            "Write",
        ]

    def test_passes_disallowed_tools_option(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        MockClaudeSDKClient.reset()
        mock_client_class = _create_mock_client_class(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(disallowed_tools=("Bash",))

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=MagicMock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                mock_client_class,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                MockClaudeAgentOptions,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
        ):
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockClaudeSDKClient.captured_options) == 1
        assert MockClaudeSDKClient.captured_options[0].disallowed_tools == ["Bash"]

    def test_model_config_does_not_pass_unsupported_params(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Verify model config params that aren't supported by SDK are ignored.

        The Claude Agent SDK does not expose max_tokens or temperature parameters
        directly - it manages token budgets internally. This test verifies that
        the adapter handles a model config gracefully without passing these
        unsupported parameters to ClaudeAgentOptions.
        """
        MockClaudeSDKClient.reset()
        mock_client_class = _create_mock_client_class(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            model_config=ClaudeAgentSDKModelConfig(
                model="claude-sonnet-4-5-20250929",
                temperature=0.7,  # Not supported by SDK
                max_tokens=2000,  # Not supported by SDK
            ),
        )

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=MagicMock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                mock_client_class,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                MockClaudeAgentOptions,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
        ):
            adapter.evaluate(simple_prompt, session=session)

        # Verify SDK client was created but max_thinking_tokens was not set
        # (since max_tokens doesn't map cleanly to SDK options)
        assert len(MockClaudeSDKClient.captured_options) == 1
        # max_thinking_tokens should not be set (attribute shouldn't exist
        # or should be None if it was explicitly set)
        assert not hasattr(
            MockClaudeSDKClient.captured_options[0], "max_thinking_tokens"
        ) or (MockClaudeSDKClient.captured_options[0].max_thinking_tokens is None)

    def test_creates_mcp_server_for_prompt_tools(self, session: Session) -> None:
        """Test that prompts with tools create an MCP server."""
        from tests.adapters.claude_agent_sdk.test_bridge import search_tool

        template_with_tools = PromptTemplate[SimpleOutput](
            ns="test",
            key="with_tools",
            sections=[
                MarkdownSection(
                    title="Task",
                    key="task",
                    template="Use the tool",
                    tools=(search_tool,),
                ),
            ],
        )
        prompt_with_tools = Prompt(template_with_tools)

        MockClaudeSDKClient.reset()
        mock_client_class = _create_mock_client_class(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()

        mock_mcp_server = MagicMock(return_value={"type": "sdk"})

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=MagicMock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                mock_client_class,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                MockClaudeAgentOptions,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter.create_mcp_server",
                mock_mcp_server,
            ),
        ):
            adapter.evaluate(prompt_with_tools, session=session)

        assert len(MockClaudeSDKClient.captured_options) == 1
        # Verify mcp_servers was set in options
        assert MockClaudeSDKClient.captured_options[0].mcp_servers is not None
        assert "weakincentives" in MockClaudeSDKClient.captured_options[0].mcp_servers
        mock_mcp_server.assert_called_once()

    def test_passes_hooks_to_options(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that hooks are passed to ClaudeAgentOptions."""
        MockClaudeSDKClient.reset()
        mock_client_class = _create_mock_client_class(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=MagicMock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                mock_client_class,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                MockClaudeAgentOptions,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
        ):
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockClaudeSDKClient.captured_options) == 1
        options = MockClaudeSDKClient.captured_options[0]
        # Verify hooks dict is present with expected keys
        assert hasattr(options, "hooks")
        assert options.hooks is not None
        assert "PreToolUse" in options.hooks
        assert "PostToolUse" in options.hooks
        assert "Stop" in options.hooks
        assert "UserPromptSubmit" in options.hooks


class TestSDKErrorHandling:
    def test_normalizes_sdk_error(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        class ErrorMockClient(MockClaudeSDKClient):
            async def receive_messages(self) -> AsyncGenerator[Any, None]:
                raise RuntimeError("SDK crashed")
                yield  # Make it a generator

        MockClaudeSDKClient.reset()

        adapter = ClaudeAgentSDKAdapter()

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=MagicMock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                ErrorMockClient,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                MockClaudeAgentOptions,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
        ):
            with pytest.raises(PromptEvaluationError, match="SDK crashed"):
                adapter.evaluate(simple_prompt, session=session)


class TestBudgetTracking:
    def test_records_usage_to_budget_tracker(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        from weakincentives.budget import Budget, BudgetTracker

        MockClaudeSDKClient.reset()
        mock_client_class = _create_mock_client_class(
            [
                MockResultMessage(
                    result="Done",
                    usage={"input_tokens": 100, "output_tokens": 50},
                    structured_output=None,
                )
            ]
        )

        budget = Budget(max_total_tokens=1000)
        budget_tracker = BudgetTracker(budget)

        adapter = ClaudeAgentSDKAdapter()

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=MagicMock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                mock_client_class,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                MockClaudeAgentOptions,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
        ):
            adapter.evaluate(
                simple_prompt, session=session, budget_tracker=budget_tracker
            )

        assert budget_tracker.consumed.input_tokens == 100
        assert budget_tracker.consumed.output_tokens == 50

    def test_creates_budget_tracker_from_budget(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that passing budget without budget_tracker creates one."""
        from weakincentives.budget import Budget

        MockClaudeSDKClient.reset()
        mock_client_class = _create_mock_client_class(
            [
                MockResultMessage(
                    result="Done",
                    usage={"input_tokens": 100, "output_tokens": 50},
                    structured_output=None,
                )
            ]
        )

        budget = Budget(max_total_tokens=1000)

        adapter = ClaudeAgentSDKAdapter()

        with (
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
                return_value=MagicMock(),
            ),
            patch(
                "claude_agent_sdk.ClaudeSDKClient",
                mock_client_class,
            ),
            patch(
                "claude_agent_sdk.types.ClaudeAgentOptions",
                MockClaudeAgentOptions,
            ),
            patch(
                "claude_agent_sdk.types.HookMatcher",
                MockHookMatcher,
            ),
            patch(
                "claude_agent_sdk.types.ResultMessage",
                MockResultMessage,
            ),
        ):
            # Pass budget but not budget_tracker - should create one internally
            response = adapter.evaluate(simple_prompt, session=session, budget=budget)

        assert response.text == "Done"
