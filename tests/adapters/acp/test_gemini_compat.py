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

"""Gemini CLI ACP compatibility tests.

Gemini CLI supports ACP via ``--experimental-acp``.  These tests verify that
WINK's ACP adapter handles Gemini-specific protocol behavior correctly at
the unit level (no real binary required).
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

from weakincentives.adapters.acp._events import extract_token_usage
from weakincentives.adapters.acp._mcp_http import MCPHttpServer
from weakincentives.adapters.acp.adapter import _extract_chunk_text
from weakincentives.adapters.acp.config import ACPClientConfig

from .conftest import (
    MockAgentCapabilities,
    MockAgentMessageChunk,
    MockHttpHeader,
    MockHttpMcpServer,
    MockInitializeResponse,
    MockNewSessionResponse,
    MockRequestError,
    MockSessionNotification,
    MockToolCallStart,
    make_mock_connection,
)


def _mock_acp_modules() -> dict[str, Any]:
    """Build mock ``acp`` / ``acp.schema`` modules for patching ``sys.modules``."""
    mock_acp = MagicMock()
    mock_acp.RequestError = MockRequestError
    mock_acp.PROTOCOL_VERSION = 1
    mock_schema = MagicMock()
    mock_schema.SessionNotification = MockSessionNotification
    mock_schema.HttpMcpServer = MockHttpMcpServer
    mock_schema.HttpHeader = MockHttpHeader
    return {"acp": mock_acp, "acp.schema": mock_schema}


# ---------------------------------------------------------------------------
# Mock content block types that mirror real ACP pydantic models
# ---------------------------------------------------------------------------


@dataclass
class TextContentBlock:
    """Mock of ``acp.schema.TextContentBlock``."""

    type: str
    text: str


@dataclass
class ImageContentBlock:
    """Mock of ``acp.schema.ImageContentBlock`` (no ``.text`` attribute)."""

    type: str
    url: str


# ---------------------------------------------------------------------------
# Mock update types that Gemini may send
# ---------------------------------------------------------------------------


@dataclass
class UsageUpdate:
    """Mock of an ACP UsageUpdate notification."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AgentPlanUpdate:
    """Mock of an ACP AgentPlanUpdate notification."""

    plan: str = ""


@dataclass
class AvailableCommandsUpdate:
    """Mock of an ACP AvailableCommandsUpdate notification."""

    commands: list[str] | None = None


@dataclass
class CurrentModeUpdate:
    """Mock of an ACP CurrentModeUpdate notification."""

    mode_id: str = ""


@dataclass
class ConfigOptionUpdate:
    """Mock of an ACP ConfigOptionUpdate notification."""

    key: str = ""
    value: str = ""


@dataclass
class SessionInfoUpdate:
    """Mock of an ACP SessionInfoUpdate notification."""

    info: str = ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGeminiHandshakeWithAuthMethods:
    """Verify _handshake() succeeds when InitializeResponse includes auth_methods."""

    def test_auth_methods_present_does_not_block(self) -> None:
        """Gemini is pre-authenticated; auth_methods in InitializeResponse is ignored."""
        conn = make_mock_connection()
        # Gemini returns auth_methods in the initialize response
        conn.initialize.return_value = MockInitializeResponse(
            protocol_version=1,
            auth_methods=["google_oauth"],
            agent_capabilities=MockAgentCapabilities(load_session=True),
        )
        conn.new_session.return_value = MockNewSessionResponse(
            session_id="gemini-sess-1"
        )

        async def _run() -> str:
            from weakincentives.adapters.acp.adapter import ACPAdapter

            adapter = ACPAdapter()
            return await adapter._handshake(conn, "/tmp/cwd", [])

        modules = _mock_acp_modules()
        # Wire up schema types needed by _handshake
        modules["acp.schema"].ClientCapabilities = MagicMock()
        modules["acp.schema"].FileSystemCapability = MagicMock()
        modules["acp.schema"].Implementation = MagicMock()
        with patch.dict(sys.modules, modules):
            session_id = asyncio.run(_run())

        assert session_id == "gemini-sess-1"


class TestTextContentBlockExtraction:
    """_extract_chunk_text() handles TextContentBlock pydantic models."""

    def test_text_content_block_with_text_attr(self) -> None:
        """Gemini sends TextContentBlock objects with a ``.text`` attribute."""

        @dataclass
        class Chunk:
            content: Any

        block = TextContentBlock(type="text", text="Hello from Gemini")
        chunk = Chunk(content=block)
        assert _extract_chunk_text(chunk) == "Hello from Gemini"

    def test_plain_string_content(self) -> None:
        """Plain string content still works (backward compat)."""
        chunk = MockAgentMessageChunk(content="plain text")
        assert _extract_chunk_text(chunk) == "plain text"

    def test_list_of_content_blocks(self) -> None:
        """A list of TextContentBlock objects is concatenated."""

        @dataclass
        class Chunk:
            content: Any

        blocks = [
            TextContentBlock(type="text", text="Hello "),
            TextContentBlock(type="text", text="world"),
        ]
        chunk = Chunk(content=blocks)
        assert _extract_chunk_text(chunk) == "Hello world"


class TestImageContentBlockFallback:
    """_extract_chunk_text() does not crash on ImageContentBlock."""

    def test_image_block_no_crash(self) -> None:
        """ImageContentBlock has no ``.text``; should fall back to str()."""

        @dataclass
        class Chunk:
            content: Any

        block = ImageContentBlock(type="image", url="https://example.com/img.png")
        chunk = Chunk(content=block)
        result = _extract_chunk_text(chunk)
        # Should not crash; produces some string representation
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_content(self) -> None:
        """Empty string content returns empty string."""

        @dataclass
        class Chunk:
            content: str

        chunk = Chunk(content="")
        assert _extract_chunk_text(chunk) == ""

    def test_none_content(self) -> None:
        """Missing content attribute returns empty string."""

        class Chunk:
            pass

        chunk = Chunk()
        assert _extract_chunk_text(chunk) == ""


class TestUnhandledUpdateTypesDontCrash:
    """Session updates of unrecognized types are silently ignored."""

    def test_gemini_specific_updates_ignored(self) -> None:
        """UsageUpdate, AgentPlanUpdate, etc. don't crash _track_update."""

        async def _run() -> None:
            from weakincentives.adapters.acp.client import ACPClient

            config = ACPClientConfig()
            client = ACPClient(config)

            updates = [
                UsageUpdate(input_tokens=100, output_tokens=50),
                AgentPlanUpdate(plan="Step 1: think"),
                AvailableCommandsUpdate(commands=["/help", "/clear"]),
                CurrentModeUpdate(mode_id="code"),
                ConfigOptionUpdate(key="temperature", value="0.7"),
                SessionInfoUpdate(info="Gemini 2.0"),
            ]
            for update in updates:
                await client.session_update("gemini-sess", update)

            # None of these should appear in message/thought/tool trackers
            assert len(client.message_chunks) == 0
            assert len(client.thought_chunks) == 0
            assert len(client.tool_call_tracker) == 0

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())


class TestUsageUpdateTimestampsLastUpdate:
    """Non-message updates still bump last_update_time."""

    def test_unrecognized_update_bumps_timestamp(self) -> None:
        async def _run() -> None:
            from weakincentives.adapters.acp.client import ACPClient

            config = ACPClientConfig()
            client = ACPClient(config)

            assert client.last_update_time is None
            await client.session_update("gemini-sess", UsageUpdate())
            assert client.last_update_time is not None
            first_time = client.last_update_time

            await client.session_update("gemini-sess", AgentPlanUpdate())
            assert client.last_update_time is not None
            assert client.last_update_time >= first_time

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())


class TestMcpHttpServerTypeField:
    """to_http_mcp_server() produces type='http' matching ACP schema."""

    def test_type_field_is_http(self) -> None:
        server = MCPHttpServer(MagicMock(), server_name="wink-tools")
        server._port = 8080

        mock_acp_schema = MagicMock()
        mock_acp_schema.HttpMcpServer = MockHttpMcpServer
        mock_acp_schema.HttpHeader = MockHttpHeader
        with patch.dict(
            sys.modules, {"acp.schema": mock_acp_schema, "acp": MagicMock()}
        ):
            result = server.to_http_mcp_server()

        assert result.type == "http"
        assert result.url == "http://127.0.0.1:8080/mcp"
        assert result.name == "wink-tools"


class TestGeminiConfigDefaults:
    """ACPClientConfig accepts Gemini-specific arguments."""

    def test_gemini_binary_config(self) -> None:
        cfg = ACPClientConfig(
            agent_bin="gemini",
            agent_args=("--experimental-acp",),
        )
        assert cfg.agent_bin == "gemini"
        assert cfg.agent_args == ("--experimental-acp",)

    def test_gemini_config_frozen(self) -> None:
        cfg = ACPClientConfig(
            agent_bin="gemini",
            agent_args=("--experimental-acp",),
        )
        try:
            cfg.agent_bin = "other"  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised


class TestRealToolCallIdsTrackedCorrectly:
    """Gemini sends proper (non-empty) tool_call_id values."""

    def test_real_ids_tracked_without_synthetic(self) -> None:
        async def _run() -> None:
            from weakincentives.adapters.acp.client import ACPClient

            config = ACPClientConfig()
            client = ACPClient(config)

            t1 = MockToolCallStart(
                tool_call_id="gemini-tc-abc123", title="code_execution"
            )
            t2 = MockToolCallStart(tool_call_id="gemini-tc-def456", title="file_search")
            await client.session_update("gemini-sess", t1)
            await client.session_update("gemini-sess", t2)

            assert "gemini-tc-abc123" in client.tool_call_tracker
            assert "gemini-tc-def456" in client.tool_call_tracker
            # No synthetic IDs should be present
            for key in client.tool_call_tracker:
                assert not key.startswith("_tc_")

        with patch.dict(sys.modules, _mock_acp_modules()):
            asyncio.run(_run())


class TestCachedReadTokensMapped:
    """extract_token_usage() maps cached_read_tokens to cached_tokens."""

    def test_cached_read_tokens_mapped(self) -> None:
        @dataclass
        class GeminiUsage:
            input_tokens: int = 500
            output_tokens: int = 200
            cached_read_tokens: int = 150

        usage = extract_token_usage(GeminiUsage())
        assert usage is not None
        assert usage.input_tokens == 500
        assert usage.output_tokens == 200
        assert usage.cached_tokens == 150

    def test_no_cached_tokens(self) -> None:
        @dataclass
        class GeminiUsage:
            input_tokens: int = 100
            output_tokens: int = 50
            cached_read_tokens: None = None

        usage = extract_token_usage(GeminiUsage())
        assert usage is not None
        assert usage.cached_tokens is None


class TestPromptSendsTextContentBlock:
    """_send_prompt() sends TextContentBlock(type='text', text=...)."""

    def test_prompt_uses_text_content_block(self) -> None:
        """Verify the prompt payload shape sent to conn.prompt()."""
        conn = make_mock_connection()

        async def _run() -> None:
            from weakincentives.adapters.acp.adapter import ACPAdapter

            adapter = ACPAdapter()
            await adapter._send_prompt(
                conn=conn,
                acp_session_id="gemini-sess",
                text="What is 2+2?",
                prompt_name="test",
                deadline=None,
                text_content_block_cls=TextContentBlock,
            )

        asyncio.run(_run())

        # Verify conn.prompt was called with a list of TextContentBlock
        conn.prompt.assert_called_once()
        args = conn.prompt.call_args
        prompt_blocks = args[0][0]  # First positional arg is the list
        assert len(prompt_blocks) == 1
        block = prompt_blocks[0]
        assert isinstance(block, TextContentBlock)
        assert block.type == "text"
        assert block.text == "What is 2+2?"
        assert args[1]["session_id"] == "gemini-sess"


class TestAgentCapabilitiesMcpHttpCheck:
    """Parse InitializeResponse with Gemini-like mcp_capabilities."""

    def test_mcp_capabilities_readable(self) -> None:
        """Gemini advertises mcp_capabilities: {http: true, sse: true}."""

        @dataclass
        class McpCapabilities:
            http: bool = True
            sse: bool = True

        caps = MockAgentCapabilities(
            mcp_capabilities=McpCapabilities(http=True, sse=True),
        )
        init_resp = MockInitializeResponse(
            protocol_version=1,
            agent_capabilities=caps,
        )

        # Verify fields are accessible
        assert init_resp.agent_capabilities is not None
        mcp_caps = init_resp.agent_capabilities.mcp_capabilities
        assert mcp_caps.http is True
        assert mcp_caps.sse is True

    def test_no_mcp_capabilities(self) -> None:
        """Agent without MCP capabilities still produces valid InitializeResponse."""
        init_resp = MockInitializeResponse(
            protocol_version=1,
            agent_capabilities=MockAgentCapabilities(mcp_capabilities=None),
        )
        assert init_resp.agent_capabilities.mcp_capabilities is None
