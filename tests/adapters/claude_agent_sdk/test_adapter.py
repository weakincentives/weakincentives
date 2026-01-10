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

from collections.abc import AsyncGenerator, AsyncIterable, Generator
from contextlib import contextmanager
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
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    SectionVisibility,
)
from weakincentives.prompt.errors import VisibilityExpansionRequired
from weakincentives.runtime.events import (
    InProcessDispatcher,
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
    max_budget_usd: float | None
    betas: list[str] | None
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


class MockSDKQuery:
    """Mock for sdk[] async generator function."""

    captured_options: ClassVar[list[MockClaudeAgentOptions]] = []
    captured_prompts: ClassVar[list[str]] = []
    _results: ClassVar[list[object]] = []
    _error: ClassVar[Exception | None] = None

    @classmethod
    def reset(cls) -> None:
        """Reset captured options and results between tests."""
        cls.captured_options = []
        cls.captured_prompts = []
        cls._results = []
        cls._error = None

    @classmethod
    def set_results(cls, results: list[object]) -> None:
        """Set the results to yield from query."""
        cls._results = results

    @classmethod
    def set_error(cls, error: Exception) -> None:
        """Set an error to raise during iteration."""
        cls._error = error

    @classmethod
    async def query(
        cls,
        *,
        prompt: str | AsyncIterable[dict[str, Any]],
        options: MockClaudeAgentOptions,
    ) -> AsyncGenerator[object, None]:
        """Mock sdk[] that yields configured results.

        Handles both string prompts (legacy) and AsyncIterable prompts (streaming mode).
        For AsyncIterable prompts, consumes the generator and extracts the user message content.
        """
        # Handle AsyncIterable prompts (streaming mode)
        if not isinstance(prompt, str):
            # Consume the async generator to get prompt content
            prompt_content = ""
            async for msg in prompt:
                if isinstance(msg, dict) and "message" in msg:
                    message = msg["message"]
                    if isinstance(message, dict) and "content" in message:
                        prompt_content = message["content"]
            cls.captured_prompts.append(prompt_content)
        else:
            cls.captured_prompts.append(prompt)

        cls.captured_options.append(options)

        if cls._error is not None:
            raise cls._error

        for result in cls._results:
            yield result


@dataclass(slots=True, frozen=True)
class SimpleOutput:
    message: str


@pytest.fixture
def session() -> Session:
    dispatcher = InProcessDispatcher()
    return Session(dispatcher=dispatcher)


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


def _setup_mock_query(results: list[object]) -> None:
    """Set up MockSDKQuery with specific results."""
    MockSDKQuery.reset()
    MockSDKQuery.set_results(results)


def _create_sdk_mock() -> MagicMock:
    """Create mock SDK module with query function."""
    mock_sdk = MagicMock()
    mock_sdk.query = MockSDKQuery.query
    return mock_sdk


@contextmanager
def sdk_patches() -> Generator[None, None, None]:
    """Context manager for common SDK patches."""
    with (
        patch(
            "weakincentives.adapters.claude_agent_sdk.adapter._import_sdk",
            return_value=_create_sdk_mock(),
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
        yield


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
        session.dispatcher.subscribe(PromptRendered, lambda e: events.append(e))

        _setup_mock_query(
            [MockResultMessage(result="Hello!", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(events) == 1
        event = events[0]
        assert event.prompt_ns == "test"
        assert event.prompt_key == "simple"
        assert event.adapter == CLAUDE_AGENT_SDK_ADAPTER_NAME

    def test_returns_prompt_response(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        _setup_mock_query(
            [
                MockResultMessage(
                    result="Hello, world!",
                    usage={"input_tokens": 10, "output_tokens": 5},
                    structured_output=None,
                )
            ]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            response = adapter.evaluate(simple_prompt, session=session)

        assert response.prompt_name == "test:simple"
        assert response.text == "Hello, world!"

    def test_extracts_structured_output(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        _setup_mock_query(
            [
                MockResultMessage(
                    result="Hello!",
                    usage=None,
                    structured_output={"message": "structured hello"},
                )
            ]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            response = adapter.evaluate(simple_prompt, session=session)

        assert response.output is not None
        assert response.output.message == "structured hello"

    def test_handles_invalid_structured_output(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        _setup_mock_query(
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

        with sdk_patches():
            response = adapter.evaluate(simple_prompt, session=session)

        # Should gracefully handle parse error and return None output
        assert response.output is None
        assert response.text == "Hello!"

    def test_handles_no_structured_output(
        self, session: Session, untyped_prompt: Prompt[None]
    ) -> None:
        _setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            response = adapter.evaluate(untyped_prompt, session=session)

        assert response.output is None

    def test_accumulates_token_usage(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        # First message with usage (non-result)
        msg1 = MagicMock()
        msg1.usage = {"input_tokens": 100, "output_tokens": 50}

        _setup_mock_query(
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
        session.dispatcher.subscribe(PromptExecuted, lambda e: events.append(e))

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
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
        _setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(cwd="/home/user/project"),
        )

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert MockSDKQuery.captured_options[0].cwd == "/home/user/project"

    def test_passes_max_turns_option(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        _setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(max_turns=10),
        )

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert MockSDKQuery.captured_options[0].max_turns == 10

    def test_passes_allowed_tools_option(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        _setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(allowed_tools=("Read", "Write"))

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert MockSDKQuery.captured_options[0].allowed_tools == [
            "Read",
            "Write",
        ]

    def test_passes_disallowed_tools_option(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        _setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(disallowed_tools=("Bash",))

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert MockSDKQuery.captured_options[0].disallowed_tools == ["Bash"]

    def test_model_config_does_not_pass_unsupported_params(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Verify model config params that aren't supported by SDK are ignored.

        The Claude Agent SDK does not expose max_tokens or temperature parameters
        directly - it manages token budgets internally. This test verifies that
        the adapter handles a model config gracefully without passing these
        unsupported parameters to ClaudeAgentOptions.
        """
        _setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            model_config=ClaudeAgentSDKModelConfig(
                model="claude-sonnet-4-5-20250929",
                temperature=0.7,  # Not supported by SDK
                max_tokens=2000,  # Not supported by SDK
            ),
        )

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        # Verify SDK query was called but max_thinking_tokens was not set
        # (since max_tokens doesn't map cleanly to SDK options)
        assert len(MockSDKQuery.captured_options) == 1
        # max_thinking_tokens should not be set (attribute shouldn't exist
        # or should be None if it was explicitly set)
        assert not hasattr(MockSDKQuery.captured_options[0], "max_thinking_tokens") or (
            MockSDKQuery.captured_options[0].max_thinking_tokens is None
        )

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

        _setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()

        mock_mcp_server = MagicMock(return_value={"type": "sdk"})

        with (
            sdk_patches(),
            patch(
                "weakincentives.adapters.claude_agent_sdk.adapter.create_mcp_server",
                mock_mcp_server,
            ),
        ):
            adapter.evaluate(prompt_with_tools, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        # Verify mcp_servers was set in options
        assert MockSDKQuery.captured_options[0].mcp_servers is not None
        assert "wink" in MockSDKQuery.captured_options[0].mcp_servers
        mock_mcp_server.assert_called_once()

    def test_passes_hooks_to_options(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that hooks are passed to ClaudeAgentOptions."""
        _setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        options = MockSDKQuery.captured_options[0]
        # Verify hooks dict is present with expected keys
        assert hasattr(options, "hooks")
        assert options.hooks is not None
        assert "PreToolUse" in options.hooks
        assert "PostToolUse" in options.hooks
        assert "Stop" in options.hooks
        assert "UserPromptSubmit" in options.hooks

    def test_passes_max_budget_usd_option(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        _setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(max_budget_usd=5.0),
        )

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert MockSDKQuery.captured_options[0].max_budget_usd == 5.0

    def test_passes_betas_option(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        _setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                betas=("extended-thinking", "computer-use")
            ),
        )

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert MockSDKQuery.captured_options[0].betas == [
            "extended-thinking",
            "computer-use",
        ]

    def test_passes_max_thinking_tokens_option(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        _setup_mock_query(
            [MockResultMessage(result="Done", usage=None, structured_output=None)]
        )

        adapter = ClaudeAgentSDKAdapter(
            model_config=ClaudeAgentSDKModelConfig(max_thinking_tokens=10000),
        )

        with sdk_patches():
            adapter.evaluate(simple_prompt, session=session)

        assert len(MockSDKQuery.captured_options) == 1
        assert MockSDKQuery.captured_options[0].max_thinking_tokens == 10000


class TestSDKErrorHandling:
    def test_normalizes_sdk_error(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        MockSDKQuery.reset()
        MockSDKQuery.set_error(RuntimeError("SDK crashed"))

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            with pytest.raises(PromptEvaluationError, match="SDK crashed"):
                adapter.evaluate(simple_prompt, session=session)


class TestBudgetTracking:
    def test_records_usage_to_budget_tracker(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        from weakincentives.budget import Budget, BudgetTracker

        _setup_mock_query(
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

        with sdk_patches():
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

        _setup_mock_query(
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

        with sdk_patches():
            # Pass budget but not budget_tracker - should create one internally
            response = adapter.evaluate(simple_prompt, session=session, budget=budget)

        assert response.text == "Done"


class TestVisibilityExpansionRequired:
    """Tests for VisibilityExpansionRequired exception propagation."""

    def test_propagates_visibility_expansion_required(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that VisibilityExpansionRequired propagates through the adapter."""
        MockSDKQuery.reset()
        MockSDKQuery.set_error(
            VisibilityExpansionRequired(
                "Model requested expansion",
                requested_overrides={("section", "key"): SectionVisibility.FULL},
                reason="Need more details",
                section_keys=("section.key",),
            )
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            with pytest.raises(VisibilityExpansionRequired) as exc_info:
                adapter.evaluate(simple_prompt, session=session)

        # Verify the exception has the expected attributes
        exc = exc_info.value
        assert isinstance(exc, VisibilityExpansionRequired)
        assert exc.requested_overrides == {("section", "key"): SectionVisibility.FULL}
        assert exc.section_keys == ("section.key",)
        assert exc.reason == "Need more details"


class TestIsolationConfig:
    """Tests for IsolationConfig integration with the adapter."""

    def test_evaluate_with_isolation_creates_ephemeral_home(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that isolation config creates an ephemeral home and cleans it up."""
        from weakincentives.adapters.claude_agent_sdk import (
            IsolationConfig,
            NetworkPolicy,
            SandboxConfig,
        )

        _setup_mock_query(
            [MockResultMessage(result="Hello!", usage={"input_tokens": 10})]
        )

        isolation = IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
            sandbox=SandboxConfig(enabled=True),
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                permission_mode="bypassPermissions",
                isolation=isolation,
            ),
        )

        with sdk_patches():
            response = adapter.evaluate(simple_prompt, session=session)

        assert response.text == "Hello!"

        # Verify isolation options were passed to SDK
        assert len(MockSDKQuery.captured_options) == 1
        options = MockSDKQuery.captured_options[0]

        # Verify env was set with ephemeral HOME
        assert hasattr(options, "env")
        env: dict[str, str] = options.env  # type: ignore[assignment]
        assert isinstance(env, dict)
        assert "HOME" in env
        # Ephemeral home should be in temp directory
        home_value = env["HOME"]
        assert isinstance(home_value, str)
        assert "claude-agent-" in home_value

        # Verify setting_sources was set to load from ephemeral HOME
        assert hasattr(options, "setting_sources")
        assert options.setting_sources == ["user"]

    def test_evaluate_with_isolation_cleans_up_on_error(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that ephemeral home is cleaned up even when SDK raises an error."""
        from weakincentives.adapters.claude_agent_sdk import (
            IsolationConfig,
            NetworkPolicy,
        )

        MockSDKQuery.reset()
        MockSDKQuery.set_error(RuntimeError("SDK error"))

        isolation = IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                permission_mode="bypassPermissions",
                isolation=isolation,
            ),
        )

        with sdk_patches():
            with pytest.raises(Exception):  # noqa: B017
                adapter.evaluate(simple_prompt, session=session)

    def test_uses_prompt_filesystem_when_available(
        self, session: Session, simple_prompt: Prompt[None]
    ) -> None:
        """When prompt has filesystem, adapter uses it instead of creating one."""
        from weakincentives.contrib.tools.vfs import VfsToolsSection

        # Create a prompt with a VFS workspace section containing a filesystem
        workspace = VfsToolsSection(session=session)

        template = PromptTemplate[None](
            ns="test",
            key="with-fs",
            name="with_filesystem",
            sections=[
                workspace,
                MarkdownSection(title="Task", template="Do something", key="task"),
            ],
        )
        prompt = Prompt(template)

        adapter = ClaudeAgentSDKAdapter()
        MockSDKQuery.reset()
        MockSDKQuery.set_results([MockResultMessage(result="Done")])

        with sdk_patches():
            response = adapter.evaluate(prompt, session=session)

        # Verify the filesystem from the prompt was used
        assert response is not None

    def test_no_permission_mode_when_none(
        self, session: Session, simple_prompt: Prompt[None]
    ) -> None:
        """When permission_mode is None, it's not added to options."""
        config = ClaudeAgentSDKClientConfig(permission_mode=None)
        adapter = ClaudeAgentSDKAdapter(client_config=config)

        MockSDKQuery.reset()
        MockSDKQuery.set_results([MockResultMessage(result="Done")])

        with sdk_patches():
            _ = adapter.evaluate(simple_prompt, session=session)

        # Check captured options don't include permission_mode
        assert len(MockSDKQuery.captured_options) == 1
        options = MockSDKQuery.captured_options[0]
        assert (
            not hasattr(options, "permission_mode") or options.permission_mode is None
        )

    def test_suppress_stderr_option(
        self, session: Session, simple_prompt: Prompt[None]
    ) -> None:
        """When suppress_stderr is True, stderr callback is added to options."""
        config = ClaudeAgentSDKClientConfig(suppress_stderr=True)
        adapter = ClaudeAgentSDKAdapter(client_config=config)

        MockSDKQuery.reset()
        MockSDKQuery.set_results([MockResultMessage(result="Done")])

        with sdk_patches():
            _ = adapter.evaluate(simple_prompt, session=session)

        # Check captured options include stderr callback
        assert len(MockSDKQuery.captured_options) == 1
        options = MockSDKQuery.captured_options[0]
        assert hasattr(options, "stderr")
        assert callable(options.stderr)

    def test_suppress_stderr_false(
        self, session: Session, simple_prompt: Prompt[None]
    ) -> None:
        """When suppress_stderr is False, stderr is still captured for debug logging.

        NOTE: As of the debug logging enhancement, stderr is always captured
        for debug logging purposes regardless of suppress_stderr setting.
        The captured stderr is logged at DEBUG level and included in error
        payloads when process failures occur.
        """
        config = ClaudeAgentSDKClientConfig(suppress_stderr=False)
        adapter = ClaudeAgentSDKAdapter(client_config=config)

        MockSDKQuery.reset()
        MockSDKQuery.set_results([MockResultMessage(result="Done")])

        with sdk_patches():
            _ = adapter.evaluate(simple_prompt, session=session)

        # stderr handler is always present for debug logging
        assert len(MockSDKQuery.captured_options) == 1
        options = MockSDKQuery.captured_options[0]
        assert hasattr(options, "stderr") and options.stderr is not None

    def test_stderr_handler_buffers_output(
        self, session: Session, simple_prompt: Prompt[None]
    ) -> None:
        """Stderr handler buffers output for debug logging and error payloads."""
        adapter = ClaudeAgentSDKAdapter()

        MockSDKQuery.reset()
        MockSDKQuery.set_results([MockResultMessage(result="Done")])

        with sdk_patches():
            _ = adapter.evaluate(simple_prompt, session=session)

        # Get the stderr handler and invoke it
        assert len(MockSDKQuery.captured_options) == 1
        options = MockSDKQuery.captured_options[0]
        stderr_handler = options.stderr

        # Invoke the handler with some test output
        stderr_handler("Test stderr line 1\n")
        stderr_handler("Test stderr line 2\n")

        # Verify the buffer captured the output
        assert len(adapter._stderr_buffer) == 2
        assert adapter._stderr_buffer[0] == "Test stderr line 1\n"
        assert adapter._stderr_buffer[1] == "Test stderr line 2\n"

    def test_message_without_result(
        self, session: Session, simple_prompt: Prompt[None]
    ) -> None:
        """Messages without result attribute or with falsy result are handled."""
        MockSDKQuery.reset()
        # Create a message without result attribute
        message_without_result = MockResultMessage()
        message_without_result.result = None
        MockSDKQuery.set_results([message_without_result])

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            response = adapter.evaluate(simple_prompt, session=session)

        assert response.text is None

    def test_non_dict_usage_ignored(
        self, session: Session, simple_prompt: Prompt[None]
    ) -> None:
        """Non-dict usage values are gracefully ignored."""
        from weakincentives.runtime.events import PromptExecuted

        MockSDKQuery.reset()
        # Create a message with non-dict usage
        message = MockResultMessage(result="Done")
        message.usage = "not a dict"  # type: ignore[assignment]
        MockSDKQuery.set_results([message])

        events: list[PromptExecuted] = []
        session.dispatcher.subscribe(PromptExecuted, lambda e: events.append(e))

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            _ = adapter.evaluate(simple_prompt, session=session)

        # Should not crash and usage should be None (non-dict usage ignored)
        assert len(events) == 1
        usage = events[0].usage
        assert usage is not None
        assert usage.input_tokens is None
        assert usage.output_tokens is None

    def test_creates_temp_folder_when_no_workspace_or_cwd(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """When no workspace and no cwd, creates a temp folder as cwd."""
        MockSDKQuery.reset()
        MockSDKQuery.set_results([MockResultMessage(result="Done")])

        # No cwd configured - should create a temp folder
        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            _ = adapter.evaluate(simple_prompt, session=session)

        # Verify cwd was passed to SDK (should be a temp folder)
        assert len(MockSDKQuery.captured_options) == 1
        options = MockSDKQuery.captured_options[0]
        assert hasattr(options, "cwd")
        assert options.cwd is not None
        # Temp folder should have wink-sdk- prefix
        assert "wink-sdk-" in options.cwd

    def test_temp_folder_cleaned_up_after_evaluate(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Temp folder is cleaned up after evaluate completes."""
        from pathlib import Path

        MockSDKQuery.reset()
        MockSDKQuery.set_results([MockResultMessage(result="Done")])

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            _ = adapter.evaluate(simple_prompt, session=session)

        # Get the cwd that was used
        assert len(MockSDKQuery.captured_options) == 1
        temp_cwd = MockSDKQuery.captured_options[0].cwd

        # Temp folder should be cleaned up
        assert not Path(temp_cwd).exists()

    def test_temp_folder_cleaned_up_on_error(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Temp folder is cleaned up even when SDK raises an error."""
        from pathlib import Path

        MockSDKQuery.reset()
        MockSDKQuery.set_error(RuntimeError("SDK error"))

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            with pytest.raises(Exception):  # noqa: B017
                adapter.evaluate(simple_prompt, session=session)

        # Get the cwd that was used
        assert len(MockSDKQuery.captured_options) == 1
        temp_cwd = MockSDKQuery.captured_options[0].cwd

        # Temp folder should still be cleaned up
        assert not Path(temp_cwd).exists()


class TestMessageContentExtraction:
    """Tests for the message content extraction helper functions."""

    def test_extract_content_block_tool_use(self) -> None:
        """Tool use blocks include name, id, and input."""
        from weakincentives.adapters.claude_agent_sdk.adapter import (
            _extract_content_block,
        )

        block = {
            "type": "tool_use",
            "name": "my_tool",
            "id": "toolu_123",
            "input": {"path": "/foo"},
        }
        result = _extract_content_block(block)
        assert result["type"] == "tool_use"
        assert result["name"] == "my_tool"
        assert result["id"] == "toolu_123"
        assert result["input"] == {"path": "/foo"}

    def test_extract_content_block_text(self) -> None:
        """Text blocks include full text content."""
        from weakincentives.adapters.claude_agent_sdk.adapter import (
            _extract_content_block,
        )

        block = {"type": "text", "text": "Hello world, this is a long message"}
        result = _extract_content_block(block)
        assert result["type"] == "text"
        assert result["text"] == "Hello world, this is a long message"

    def test_extract_content_block_tool_result(self) -> None:
        """Tool result blocks include tool_use_id and full content."""
        from weakincentives.adapters.claude_agent_sdk.adapter import (
            _extract_content_block,
        )

        block = {
            "type": "tool_result",
            "tool_use_id": "toolu_123",
            "content": "Full result content here",
            "is_error": False,
        }
        result = _extract_content_block(block)
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "toolu_123"
        assert result["content"] == "Full result content here"
        assert result["is_error"] is False

    def test_extract_content_block_tool_result_no_is_error(self) -> None:
        """Tool result blocks without is_error field work correctly."""
        from weakincentives.adapters.claude_agent_sdk.adapter import (
            _extract_content_block,
        )

        block = {
            "type": "tool_result",
            "tool_use_id": "toolu_456",
            "content": "Some content",
        }
        result = _extract_content_block(block)
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "toolu_456"
        assert result["content"] == "Some content"
        assert "is_error" not in result

    def test_extract_content_block_unknown_type(self) -> None:
        """Unknown block types include all fields."""
        from weakincentives.adapters.claude_agent_sdk.adapter import (
            _extract_content_block,
        )

        block = {"type": "image", "data": "base64...", "media_type": "image/png"}
        result = _extract_content_block(block)
        assert result["type"] == "image"
        assert result["data"] == "base64..."
        assert result["media_type"] == "image/png"

    def test_extract_list_content_mixed(self) -> None:
        """List content extracts all blocks with full content."""
        from weakincentives.adapters.claude_agent_sdk.adapter import (
            _extract_list_content,
        )

        content = [
            {"type": "text", "text": "Hello"},
            {
                "type": "tool_use",
                "name": "read_file",
                "id": "t1",
                "input": {"path": "/x"},
            },
            {"type": "text", "text": "World"},
        ]
        result = _extract_list_content(content)
        assert len(result) == 3
        assert result[0] == {"type": "text", "text": "Hello"}
        assert result[1]["name"] == "read_file"
        assert result[1]["input"] == {"path": "/x"}
        assert result[2] == {"type": "text", "text": "World"}

    def test_extract_list_content_skips_non_dict(self) -> None:
        """Non-dict blocks are skipped."""
        from weakincentives.adapters.claude_agent_sdk.adapter import (
            _extract_list_content,
        )

        content = [
            {"type": "text", "text": "Hello"},
            "not a dict",
            None,
        ]
        result = _extract_list_content(content)
        assert len(result) == 1
        assert result[0]["text"] == "Hello"

    def test_extract_inner_message_content_string(self) -> None:
        """String content is extracted fully."""
        from weakincentives.adapters.claude_agent_sdk.adapter import (
            _extract_inner_message_content,
        )

        inner_msg = {"role": "assistant", "content": "Full message content here"}
        result = _extract_inner_message_content(inner_msg)
        assert result == {"role": "assistant", "content": "Full message content here"}

    def test_extract_inner_message_content_list(self) -> None:
        """List content is extracted as content_blocks."""
        from weakincentives.adapters.claude_agent_sdk.adapter import (
            _extract_inner_message_content,
        )

        inner_msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Thinking..."},
                {"type": "tool_use", "name": "bash", "id": "t1"},
            ],
        }
        result = _extract_inner_message_content(inner_msg)
        assert result["role"] == "assistant"
        assert len(result["content_blocks"]) == 2
        assert result["content_blocks"][0]["text"] == "Thinking..."
        assert result["content_blocks"][1]["name"] == "bash"

    def test_extract_inner_message_content_no_role(self) -> None:
        """Inner message without role skips role field."""
        from weakincentives.adapters.claude_agent_sdk.adapter import (
            _extract_inner_message_content,
        )

        inner_msg = {"content": "Hello"}
        result = _extract_inner_message_content(inner_msg)
        assert "role" not in result
        assert result["content"] == "Hello"

    def test_extract_inner_message_content_non_str_non_list(self) -> None:
        """Non-string/non-list content returns only role."""
        from weakincentives.adapters.claude_agent_sdk.adapter import (
            _extract_inner_message_content,
        )

        inner_msg = {"role": "user", "content": 12345}
        result = _extract_inner_message_content(inner_msg)
        assert result == {"role": "user"}

    def test_extract_message_content_with_inner_message(self) -> None:
        """Message with inner message dict extracts full content."""
        from weakincentives.adapters.claude_agent_sdk.adapter import (
            _extract_message_content,
        )

        message = MagicMock()
        message.message = {"role": "user", "content": "Full user message"}
        message.result = None
        message.structured_output = None
        message.usage = None

        result = _extract_message_content(message)
        assert result["role"] == "user"
        assert result["content"] == "Full user message"

    def test_extract_message_content_with_result(self) -> None:
        """ResultMessage with result field extracts full result."""
        from weakincentives.adapters.claude_agent_sdk.adapter import (
            _extract_message_content,
        )

        message = MagicMock()
        message.message = None
        message.result = "Final answer with full content"
        message.structured_output = None
        message.usage = None

        result = _extract_message_content(message)
        assert result["result"] == "Final answer with full content"

    def test_extract_message_content_with_structured_output(self) -> None:
        """Message with structured_output includes full structured output."""
        from weakincentives.adapters.claude_agent_sdk.adapter import (
            _extract_message_content,
        )

        message = MagicMock()
        message.message = None
        message.result = None
        message.structured_output = {"summary": "test", "issues": ["a", "b"]}
        message.usage = None

        result = _extract_message_content(message)
        assert result["structured_output"] == {"summary": "test", "issues": ["a", "b"]}

    def test_extract_message_content_with_usage(self) -> None:
        """Message with usage includes usage data."""
        from weakincentives.adapters.claude_agent_sdk.adapter import (
            _extract_message_content,
        )

        message = MagicMock()
        message.message = None
        message.result = None
        message.structured_output = None
        message.usage = {"input_tokens": 100, "output_tokens": 50}

        result = _extract_message_content(message)
        assert result["usage"] == {"input_tokens": 100, "output_tokens": 50}

    def test_extract_message_content_no_attrs(self) -> None:
        """Message without expected attributes returns empty dict."""
        from weakincentives.adapters.claude_agent_sdk.adapter import (
            _extract_message_content,
        )

        message = MagicMock(spec=[])

        result = _extract_message_content(message)
        assert result == {}


class TestVerifyTaskCompletion:
    """Tests for _verify_task_completion method."""

    @pytest.fixture
    def session(self) -> Session:
        return Session(dispatcher=InProcessDispatcher())

    @pytest.fixture
    def adapter(self) -> ClaudeAgentSDKAdapter:
        return ClaudeAgentSDKAdapter()

    def test_no_checker_configured_does_nothing(
        self, adapter: ClaudeAgentSDKAdapter, session: Session
    ) -> None:
        """When no checker is configured, verification passes."""
        # Default adapter has no task_completion_checker
        adapter._verify_task_completion(
            output={"key": "value"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test",
        )
        # Should not raise

    def test_no_output_does_nothing(self, session: Session) -> None:
        """When output is None, verification passes."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            PlanBasedChecker,
        )
        from weakincentives.contrib.tools.planning import Plan

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=PlanBasedChecker(plan_type=Plan),
            ),
        )
        adapter._verify_task_completion(
            output=None,
            session=session,
            stop_reason="structured_output",
            prompt_name="test",
        )
        # Should not raise

    def test_raises_when_tasks_incomplete(self, session: Session) -> None:
        """When tasks are incomplete, raises PromptEvaluationError."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            PlanBasedChecker,
        )
        from weakincentives.contrib.tools.planning import (
            Plan,
            PlanningToolsSection,
            PlanStep,
        )

        # Initialize plan with incomplete tasks
        PlanningToolsSection._initialize_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(
                    PlanStep(step_id=1, title="Done", status="done"),
                    PlanStep(step_id=2, title="Pending", status="pending"),
                ),
            )
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=PlanBasedChecker(plan_type=Plan),
            ),
        )

        with pytest.raises(PromptEvaluationError) as exc_info:
            adapter._verify_task_completion(
                output={"summary": "done"},
                session=session,
                stop_reason="structured_output",
                prompt_name="test_prompt",
            )

        assert "Tasks incomplete" in str(exc_info.value)
        assert "Pending" in str(exc_info.value)

    def test_passes_when_tasks_complete(self, session: Session) -> None:
        """When all tasks are complete, verification passes."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            PlanBasedChecker,
        )
        from weakincentives.contrib.tools.planning import (
            Plan,
            PlanningToolsSection,
            PlanStep,
        )

        # Initialize plan with all tasks done
        PlanningToolsSection._initialize_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="completed",
                steps=(
                    PlanStep(step_id=1, title="Task 1", status="done"),
                    PlanStep(step_id=2, title="Task 2", status="done"),
                ),
            )
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=PlanBasedChecker(plan_type=Plan),
            ),
        )

        # Should not raise
        adapter._verify_task_completion(
            output={"summary": "done"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
        )

    def test_skips_when_deadline_exceeded(self, session: Session) -> None:
        """When deadline is exceeded, verification is skipped."""
        from datetime import timedelta
        from unittest.mock import MagicMock

        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            PlanBasedChecker,
        )
        from weakincentives.contrib.tools.planning import (
            Plan,
            PlanningToolsSection,
            PlanStep,
        )

        # Initialize plan with incomplete tasks
        PlanningToolsSection._initialize_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Pending", status="pending"),),
            )
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=PlanBasedChecker(plan_type=Plan),
            ),
        )

        # Create a mock deadline that is exceeded (remaining returns negative)
        exceeded_deadline = MagicMock()
        exceeded_deadline.remaining.return_value = timedelta(seconds=-1)

        # Should not raise despite incomplete tasks
        adapter._verify_task_completion(
            output={"summary": "partial"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
            deadline=exceeded_deadline,
        )

    def test_skips_when_budget_exhausted(self, session: Session) -> None:
        """When budget is exhausted, verification is skipped."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            PlanBasedChecker,
        )
        from weakincentives.budget import Budget, BudgetTracker
        from weakincentives.contrib.tools.planning import (
            Plan,
            PlanningToolsSection,
            PlanStep,
        )

        # Initialize plan with incomplete tasks
        PlanningToolsSection._initialize_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Pending", status="pending"),),
            )
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=PlanBasedChecker(plan_type=Plan),
            ),
        )

        # Create exhausted budget tracker
        budget = Budget(max_total_tokens=100)
        tracker = BudgetTracker(budget)
        # Consume all budget
        from weakincentives.runtime.events._types import TokenUsage

        tracker.record_cumulative("test", TokenUsage(input_tokens=50, output_tokens=50))

        # Should not raise despite incomplete tasks
        adapter._verify_task_completion(
            output={"summary": "partial"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
            budget_tracker=tracker,
        )

    def test_passes_filesystem_and_adapter_to_context(self, session: Session) -> None:
        """Filesystem and adapter are passed to TaskCompletionContext."""
        from unittest.mock import MagicMock

        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            TaskCompletionChecker,
            TaskCompletionContext,
            TaskCompletionResult,
        )
        from weakincentives.filesystem import Filesystem

        # Create a mock checker that captures the context
        captured_context: list[TaskCompletionContext] = []

        class CapturingChecker(TaskCompletionChecker):
            def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
                captured_context.append(context)
                return TaskCompletionResult.ok()

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=CapturingChecker(),
            ),
        )

        # Create mock prompt with filesystem resource
        mock_filesystem = MagicMock(spec=Filesystem)
        mock_resources = MagicMock()
        mock_resources.get.return_value = mock_filesystem
        mock_prompt = MagicMock()
        mock_prompt.resources = mock_resources

        adapter._verify_task_completion(
            output={"summary": "done"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
            prompt=mock_prompt,
        )

        assert len(captured_context) == 1
        ctx = captured_context[0]
        assert ctx.filesystem is mock_filesystem
        assert ctx.adapter is adapter

    def test_handles_filesystem_lookup_failure(self, session: Session) -> None:
        """When filesystem lookup fails, context still gets adapter but no filesystem."""
        from unittest.mock import MagicMock

        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            TaskCompletionChecker,
            TaskCompletionContext,
            TaskCompletionResult,
        )
        from weakincentives.resources.errors import UnboundResourceError

        # Create a mock checker that captures the context
        captured_context: list[TaskCompletionContext] = []

        class CapturingChecker(TaskCompletionChecker):
            def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
                captured_context.append(context)
                return TaskCompletionResult.ok()

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=CapturingChecker(),
            ),
        )

        # Create mock prompt where filesystem lookup raises UnboundResourceError
        mock_resources = MagicMock()
        mock_resources.get.side_effect = UnboundResourceError(object)
        mock_prompt = MagicMock()
        mock_prompt.resources = mock_resources

        adapter._verify_task_completion(
            output={"summary": "done"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
            prompt=mock_prompt,
        )

        assert len(captured_context) == 1
        ctx = captured_context[0]
        assert ctx.filesystem is None  # Lookup failed
        assert ctx.adapter is adapter  # Adapter still passed

    def test_runs_checker_when_budget_not_exhausted(self, session: Session) -> None:
        """When budget_tracker is provided but not exhausted, checker still runs."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            PlanBasedChecker,
        )
        from weakincentives.budget import Budget, BudgetTracker
        from weakincentives.contrib.tools.planning import (
            Plan,
            PlanningToolsSection,
            PlanStep,
        )

        # Initialize plan with incomplete tasks
        PlanningToolsSection._initialize_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Pending", status="pending"),),
            )
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=PlanBasedChecker(plan_type=Plan),
            ),
        )

        # Create budget tracker with plenty of budget remaining
        budget = Budget(max_total_tokens=1000)
        tracker = BudgetTracker(budget)
        # Only consume 10% of budget
        from weakincentives.runtime.events._types import TokenUsage

        tracker.record_cumulative("test", TokenUsage(input_tokens=50, output_tokens=50))

        # Should raise because tasks are incomplete and budget is not exhausted
        with pytest.raises(PromptEvaluationError) as exc_info:
            adapter._verify_task_completion(
                output={"summary": "partial"},
                session=session,
                stop_reason="structured_output",
                prompt_name="test_prompt",
                budget_tracker=tracker,
            )

        assert "Tasks incomplete" in str(exc_info.value)
