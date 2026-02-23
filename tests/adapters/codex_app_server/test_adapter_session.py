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

"""Tests for the Codex App Server adapter — resolve_cwd, structured output, and
session-level evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weakincentives.adapters.codex_app_server._response import (
    parse_structured_output_or_raise,
)
from weakincentives.adapters.codex_app_server.adapter import CodexAppServerAdapter
from weakincentives.adapters.codex_app_server.client import (
    CodexAppServerClient,
    CodexClientError,
)
from weakincentives.adapters.codex_app_server.config import (
    CodexAppServerClientConfig,
)
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.budget import Budget
from weakincentives.filesystem import Filesystem
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate, Tool
from weakincentives.prompt.tool import ToolContext, ToolResult
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session

# ---- Helpers ----


def _make_session() -> tuple[Session, InProcessDispatcher]:
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher, tags={"suite": "tests"})
    return session, dispatcher


def _make_simple_prompt(name: str = "test-prompt") -> Prompt[object]:
    template: PromptTemplate[object] = PromptTemplate(
        ns="test",
        key="basic",
        sections=(),
        name=name,
    )
    return Prompt(template)


@dataclass(slots=True, frozen=True)
class _AddParams:
    x: int
    y: int


@dataclass(slots=True, frozen=True)
class _AddResult:
    sum: int


def _add_handler(params: _AddParams, *, context: ToolContext) -> ToolResult[_AddResult]:
    return ToolResult.ok(
        _AddResult(sum=params.x + params.y), message=str(params.x + params.y)
    )


_ADD_TOOL = Tool[_AddParams, _AddResult](
    name="add",
    description="Add two numbers",
    handler=_add_handler,
)


def _make_prompt_with_tool(name: str = "tool-prompt") -> Prompt[object]:
    section = MarkdownSection(
        title="Tools",
        template="Use the tools below.",
        key="tools",
        tools=[_ADD_TOOL],
    )
    template: PromptTemplate[object] = PromptTemplate(
        ns="test",
        key="with-tool",
        sections=(section,),
        name=name,
    )
    return Prompt(template)


def _make_mock_client() -> AsyncMock:
    """Create a mock CodexAppServerClient."""
    client = AsyncMock(spec=CodexAppServerClient)
    client.stderr_output = ""
    client.start = AsyncMock()
    client.stop = AsyncMock()
    client.send_request = AsyncMock(return_value={})
    client.send_notification = AsyncMock()
    client.send_response = AsyncMock()
    return client


def _messages_iterator(
    messages: list[dict[str, Any]],
) -> Any:
    """Create an async iterator from a list of messages."""

    async def _iter() -> Any:
        for msg in messages:
            yield msg

    return _iter()


# ---- Tests ----


class TestResolveCwd:
    def test_no_filesystem_no_cwd_creates_temp(self) -> None:
        import shutil as _shutil

        adapter = CodexAppServerAdapter()
        prompt = _make_simple_prompt()

        cwd, temp_dir, _new_prompt = adapter._resolve_cwd(prompt)
        try:
            assert cwd is not None
            assert temp_dir is not None
            assert cwd == temp_dir
        finally:
            if temp_dir:
                _shutil.rmtree(temp_dir, ignore_errors=True)

    def test_no_filesystem_with_cwd_uses_configured(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/configured")
        )
        prompt = _make_simple_prompt()

        cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
        assert cwd == "/tmp/configured"
        assert temp_dir is None

    def test_workspace_section_extracts_root(self) -> None:
        """When prompt has a workspace section with HostFilesystem and no cwd."""
        from weakincentives.prompt import WorkspaceSection

        adapter = CodexAppServerAdapter()
        session, _ = _make_session()

        workspace = WorkspaceSection(session=session)
        workspace_root = str(workspace.temp_dir)
        try:
            template: PromptTemplate[object] = PromptTemplate(
                ns="test",
                key="with-ws",
                sections=(workspace,),
                name="ws-prompt",
            )
            prompt = Prompt(template)

            cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
            assert cwd == workspace_root
            assert temp_dir is None
        finally:
            workspace.cleanup()

    def test_non_host_filesystem_falls_back_to_cwd(self) -> None:
        """When filesystem is not HostFilesystem, falls back to Path.cwd()."""
        adapter = CodexAppServerAdapter()
        prompt = _make_simple_prompt()

        # Mock prompt.filesystem() to return a non-HostFilesystem
        mock_fs = MagicMock(spec=Filesystem)
        with patch.object(type(prompt), "filesystem", return_value=mock_fs):
            cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
            assert cwd is not None
            assert temp_dir is None

    def test_workspace_with_configured_cwd(self) -> None:
        """When prompt has workspace section AND cwd is configured, cwd wins."""
        from weakincentives.prompt import WorkspaceSection

        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/configured")
        )
        session, _ = _make_session()

        workspace = WorkspaceSection(session=session)
        try:
            template: PromptTemplate[object] = PromptTemplate(
                ns="test",
                key="with-ws2",
                sections=(workspace,),
                name="ws-prompt2",
            )
            prompt = Prompt(template)

            cwd, temp_dir, _ = adapter._resolve_cwd(prompt)
            # Configured cwd should win over workspace root
            assert cwd == "/tmp/configured"
            assert temp_dir is None
        finally:
            workspace.cleanup()


class TestParseStructuredOutput:
    def test_valid_json(self) -> None:
        from weakincentives.prompt.rendering import RenderedPrompt as RP
        from weakincentives.prompt.structured_output import StructuredOutputConfig

        @dataclass(slots=True, frozen=True)
        class Result:
            answer: int

        rendered = RP(
            text="",
            structured_output=StructuredOutputConfig(
                dataclass_type=Result,
                container="object",
                allow_extra_keys=False,
            ),
        )

        result = parse_structured_output_or_raise('{"answer": 42}', rendered, "test")
        assert result is not None
        assert result.answer == 42

    def test_invalid_json_raises(self) -> None:
        from weakincentives.prompt.rendering import RenderedPrompt as RP
        from weakincentives.prompt.structured_output import StructuredOutputConfig

        @dataclass(slots=True, frozen=True)
        class Dummy:
            x: int

        rendered = RP(
            text="",
            structured_output=StructuredOutputConfig(
                dataclass_type=Dummy,
                container="object",
                allow_extra_keys=False,
            ),
        )

        with pytest.raises(PromptEvaluationError, match="parse structured"):
            parse_structured_output_or_raise("not json", rendered, "test")

    def test_array_container_parsed(self) -> None:
        from weakincentives.prompt.rendering import RenderedPrompt as RP
        from weakincentives.prompt.structured_output import StructuredOutputConfig

        @dataclass(slots=True, frozen=True)
        class Item:
            value: int

        rendered = RP(
            text="",
            structured_output=StructuredOutputConfig(
                dataclass_type=Item,
                container="array",
                allow_extra_keys=False,
            ),
        )

        # Array wrapper format: {"items": [...]}
        text = '{"items": [{"value": 1}, {"value": 2}]}'
        result = parse_structured_output_or_raise(text, rendered, "test")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].value == 1
        assert result[1].value == 2


class TestArraySchemaWrapping:
    def test_array_container_wraps_schema(self) -> None:
        """When container='array', the output schema wraps element in items."""
        from weakincentives.prompt.rendering import RenderedPrompt as RP
        from weakincentives.prompt.structured_output import StructuredOutputConfig

        @dataclass(slots=True, frozen=True)
        class Item:
            value: int

        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            {
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": "agentMessage",
                        "text": '{"items": [{"value": 1}]}',
                    }
                },
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        original_render = prompt.render

        def patched_render(**kwargs: Any) -> RP[Any]:
            rendered = original_render(**kwargs)
            return RP(
                text=rendered.text,
                structured_output=StructuredOutputConfig(
                    dataclass_type=Item,
                    container="array",
                    allow_extra_keys=False,
                ),
                _tools=rendered.tools,
            )

        with (
            patch(
                "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
            ) as MockClient,
            patch.object(prompt, "render", side_effect=patched_render),
        ):
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.output is not None
        assert isinstance(result.output, list)
        assert result.output[0].value == 1

        # Verify outputSchema was wrapped with items array
        turn_call = mock_client.send_request.call_args_list[2]
        output_schema = turn_call[0][1]["outputSchema"]
        assert output_schema["type"] == "object"
        assert "items" in output_schema["properties"]
        assert output_schema["properties"]["items"]["type"] == "array"


class TestEvaluateWithOutputSchema:
    """Test evaluate with output schema/structured output."""

    def test_output_schema_sent_and_parsed(self) -> None:
        from weakincentives.prompt.rendering import RenderedPrompt
        from weakincentives.prompt.structured_output import StructuredOutputConfig

        @dataclass(slots=True, frozen=True)
        class Answer:
            value: int

        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": '{"value": 42}'}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        # Patch prompt.render to return a RenderedPrompt with output_type
        original_render = prompt.render

        def patched_render(**kwargs: Any) -> RenderedPrompt[Any]:
            rendered = original_render(**kwargs)
            # Replace structured_output to set output_type
            return RenderedPrompt(
                text=rendered.text,
                structured_output=StructuredOutputConfig(
                    dataclass_type=Answer,
                    container="object",
                    allow_extra_keys=False,
                ),
                _tools=rendered.tools,
            )

        with (
            patch(
                "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
            ) as MockClient,
            patch.object(prompt, "render", side_effect=patched_render),
        ):
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        assert result.output is not None
        assert result.output.value == 42

        # Verify outputSchema was sent in turn/start
        turn_call = mock_client.send_request.call_args_list[2]
        assert "outputSchema" in turn_call[0][1]


class TestEvaluateWithDeltaAccumulation:
    """Test that delta messages are accumulated properly."""

    def test_delta_then_text_uses_text(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            {"method": "item/agentMessage/delta", "params": {"delta": "Hel"}},
            {"method": "item/agentMessage/delta", "params": {"delta": "lo"}},
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "Hello world"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)

        # The "text" kind from item/completed replaces accumulated delta
        assert result.text == "Hello world"


class TestEvaluateWithBudgetTracking:
    """Test budget tracking integration."""

    def test_budget_tracker_records_usage(self) -> None:
        from weakincentives.budget import BudgetTracker

        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()
        budget = Budget(max_input_tokens=1000, max_output_tokens=500)
        budget_tracker = BudgetTracker(budget)

        messages = [
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "tokenUsage": {"last": {"inputTokens": 100, "outputTokens": 50}}
                },
            },
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "ok"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(
                prompt, session=session, budget_tracker=budget_tracker
            )
            assert result.text == "ok"


class TestEvaluateWithServerRequestDuringStream:
    """Test that server requests during streaming are handled."""

    def test_tool_call_during_stream(self) -> None:
        adapter = CodexAppServerAdapter(
            client_config=CodexAppServerClientConfig(cwd="/tmp/test"),
        )
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        messages = [
            # Server request: tool call
            {
                "id": 100,
                "method": "item/tool/call",
                "params": {"tool": "unknown_tool", "arguments": {}},
            },
            {
                "method": "item/completed",
                "params": {"item": {"type": "agentMessage", "text": "done"}},
            },
            {
                "method": "turn/completed",
                "params": {"turn": {"status": "completed"}},
            },
        ]

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = [
                {"capabilities": {}},
                {"thread": {"id": "t-1"}},
                {"turn": {"id": 1}},
            ]
            mock_client.read_messages.return_value = _messages_iterator(messages)
            MockClient.return_value = mock_client

            result = adapter.evaluate(prompt, session=session)
            assert result.text == "done"
            # The unknown tool call should have sent a response
            mock_client.send_response.assert_called()


class TestEvaluateTempWorkspaceCleanup:
    """Test that temporary workspace is cleaned up on failure."""

    def test_temp_dir_cleaned_up_on_error(self) -> None:
        adapter = CodexAppServerAdapter()
        session, _ = _make_session()
        prompt = _make_simple_prompt()

        with patch(
            "weakincentives.adapters.codex_app_server.adapter.CodexAppServerClient"
        ) as MockClient:
            mock_client = _make_mock_client()
            mock_client.send_request.side_effect = CodexClientError("fail")
            MockClient.return_value = mock_client

            with pytest.raises(PromptEvaluationError):
                adapter.evaluate(prompt, session=session)
