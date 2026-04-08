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

"""Integration tests for Codex App Server transports (stdio and WebSocket).

These tests require:
- ``codex`` CLI on PATH
- Valid Codex authentication (ChatGPT or API key)

Tests are skipped automatically when ``codex`` is not available.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from typing import Any

import pytest

from weakincentives.adapters.codex_app_server.client import (
    CodexAppServerClient,
    CodexClientError,
)

# Skip entire module if codex is not installed.
_HAS_CODEX = shutil.which("codex") is not None
pytestmark = pytest.mark.skipif(not _HAS_CODEX, reason="codex CLI not on PATH")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


async def _run_handshake(client: CodexAppServerClient) -> dict[str, Any]:
    """Perform initialize + initialized handshake, return init result."""
    await client.start()
    result = await client.send_request(
        "initialize",
        {
            "clientInfo": {
                "name": "wink-integ",
                "title": "Integration",
                "version": "0.0.1",
            },
            "capabilities": {"experimentalApi": True},
        },
        timeout=15.0,
    )
    await client.send_notification("initialized")
    return result


async def _run_simple_prompt(
    client: CodexAppServerClient,
    cwd: str,
    prompt_text: str = "Reply with exactly: HELLO. Nothing else.",
) -> str:
    """Run a full init → thread → turn → stream cycle, return accumulated text."""
    await _run_handshake(client)

    thread_result = await client.send_request(
        "thread/start",
        {
            "model": "gpt-5.3-codex",
            "cwd": cwd,
            "approvalPolicy": "never",
            "ephemeral": True,
            "sandbox": "danger-full-access",
        },
        timeout=15.0,
    )
    thread_id = thread_result["thread"]["id"]

    turn_result = await client.send_request(
        "turn/start",
        {
            "threadId": thread_id,
            "input": [{"type": "text", "text": prompt_text}],
        },
        timeout=15.0,
    )
    _ = turn_result  # turn_id not needed for simple streaming

    # Drain messages until turn/completed.
    accumulated = ""
    async for msg in client.read_messages():
        method = msg.get("method", "")

        # Handle server requests (approvals, tool calls).
        if "id" in msg and "method" in msg:
            await client.send_response(msg["id"], {"decision": "accept"})
            continue

        if method == "item/agentMessage/delta":
            accumulated += msg.get("params", {}).get("delta", "")
        elif method == "item/completed":
            item = msg.get("params", {}).get("item", {})
            if item.get("type") == "agentMessage" and item.get("text"):
                accumulated = item["text"]
        elif method == "turn/completed":
            break

    return accumulated


# ---------------------------------------------------------------------------
# stdio transport tests
# ---------------------------------------------------------------------------


class TestStdioTransport:
    """Integration tests for the stdio (NDJSON) transport."""

    @pytest.mark.timeout(60)
    def test_handshake(self) -> None:
        async def _run() -> None:
            client = CodexAppServerClient()
            try:
                result = await _run_handshake(client)
                assert "codexHome" in result
            finally:
                await client.stop()

        asyncio.run(_run())

    @pytest.mark.timeout(120)
    def test_simple_prompt(self) -> None:
        async def _run() -> None:
            with tempfile.TemporaryDirectory(prefix="codex_integ_stdio_") as cwd:
                client = CodexAppServerClient()
                try:
                    text = await _run_simple_prompt(client, cwd)
                    assert "HELLO" in text
                finally:
                    await client.stop()

        asyncio.run(_run())

    @pytest.mark.timeout(60)
    def test_double_initialize_rejected(self) -> None:
        async def _run() -> None:
            client = CodexAppServerClient()
            try:
                await _run_handshake(client)
                with pytest.raises(CodexClientError, match="Already initialized"):
                    await client.send_request(
                        "initialize",
                        {"clientInfo": {"name": "x", "title": "X", "version": "0"}},
                        timeout=10.0,
                    )
            finally:
                await client.stop()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Managed WebSocket transport tests
# ---------------------------------------------------------------------------


class TestManagedWebSocketTransport:
    """Integration tests for the managed WebSocket transport.

    The client spawns ``codex app-server --listen ws://…`` and connects.
    """

    @pytest.mark.timeout(60)
    def test_handshake(self) -> None:
        async def _run() -> None:
            client = CodexAppServerClient(transport="websocket")
            try:
                result = await _run_handshake(client)
                assert "codexHome" in result
                assert client._transport == "websocket"
                assert client._remote_url is None
            finally:
                await client.stop()

        asyncio.run(_run())

    @pytest.mark.timeout(120)
    def test_simple_prompt(self) -> None:
        async def _run() -> None:
            with tempfile.TemporaryDirectory(prefix="codex_integ_ws_") as cwd:
                client = CodexAppServerClient(transport="websocket")
                try:
                    text = await _run_simple_prompt(client, cwd)
                    assert "HELLO" in text
                finally:
                    await client.stop()

        asyncio.run(_run())

    @pytest.mark.timeout(120)
    def test_dynamic_tool_call(self) -> None:
        """Verify bidirectional tool calls work over WebSocket."""

        async def _run() -> None:
            with tempfile.TemporaryDirectory(prefix="codex_integ_ws_tool_") as cwd:
                client = CodexAppServerClient(transport="websocket")
                try:
                    await _run_handshake(client)

                    thread_result = await client.send_request(
                        "thread/start",
                        {
                            "model": "gpt-5.3-codex",
                            "cwd": cwd,
                            "approvalPolicy": "never",
                            "ephemeral": True,
                            "sandbox": "danger-full-access",
                            "dynamicTools": [
                                {
                                    "name": "echo_tool",
                                    "description": "Echoes the input back. Always use when asked to echo.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "description": "Text to echo",
                                            }
                                        },
                                        "required": ["text"],
                                        "additionalProperties": False,
                                    },
                                }
                            ],
                        },
                        timeout=15.0,
                    )
                    thread_id = thread_result["thread"]["id"]

                    await client.send_request(
                        "turn/start",
                        {
                            "threadId": thread_id,
                            "input": [
                                {
                                    "type": "text",
                                    "text": "Use the echo_tool to echo 'WS_TOOL_OK', then report the result.",
                                }
                            ],
                        },
                        timeout=15.0,
                    )

                    tool_calls = []
                    accumulated = ""
                    async for msg in client.read_messages():
                        method = msg.get("method", "")

                        if "id" in msg and "method" in msg:
                            if method == "item/tool/call":
                                params = msg.get("params", {})
                                tool_calls.append(params.get("tool"))
                                args = params.get("arguments", {})
                                if isinstance(args, str):
                                    args = json.loads(args)
                                text = args.get("text", "")
                                await client.send_response(
                                    msg["id"],
                                    {
                                        "success": True,
                                        "contentItems": [
                                            {"type": "inputText", "text": text}
                                        ],
                                    },
                                )
                            else:
                                await client.send_response(
                                    msg["id"], {"decision": "accept"}
                                )
                            continue

                        if method == "item/agentMessage/delta":
                            accumulated += msg.get("params", {}).get("delta", "")
                        elif method == "turn/completed":
                            break

                    assert len(tool_calls) >= 1
                    assert "echo_tool" in tool_calls
                    assert "WS_TOOL_OK" in accumulated

                finally:
                    await client.stop()

        asyncio.run(_run())

    @pytest.mark.timeout(60)
    def test_double_initialize_rejected(self) -> None:
        async def _run() -> None:
            client = CodexAppServerClient(transport="websocket")
            try:
                await _run_handshake(client)
                with pytest.raises(CodexClientError, match="Already initialized"):
                    await client.send_request(
                        "initialize",
                        {"clientInfo": {"name": "x", "title": "X", "version": "0"}},
                        timeout=10.0,
                    )
            finally:
                await client.stop()

        asyncio.run(_run())
