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

"""Shared test fixtures for ACP adapter tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


class MockRequestError(Exception):
    """Mock of acp.RequestError."""

    def __init__(self, message: str = "", code: int = -1) -> None:
        super().__init__(message)
        self.code = code

    @classmethod
    def method_not_found(cls, method: str) -> MockRequestError:
        return cls(f"Method not found: {method}", code=-32601)


@dataclass
class MockSessionNotification:
    """Mock of acp.schema.SessionNotification."""

    sessionId: str
    update: Any


@dataclass
class AgentMessageChunk:
    """Mock of an AgentMessageChunk update.

    Named to match real ACP class so ``type(update).__name__`` works in client.
    """

    content: str


# Alias for test imports
MockAgentMessageChunk = AgentMessageChunk


@dataclass
class AgentThoughtChunk:
    """Mock of an AgentThoughtChunk update."""

    content: str


MockAgentThoughtChunk = AgentThoughtChunk


@dataclass
class ToolCallStart:
    """Mock of a ToolCallStart update.

    Field names match the real ACP ``ToolCall`` pydantic model.
    """

    tool_call_id: str
    title: str
    status: str = "in_progress"
    raw_input: Any = None
    raw_output: Any = None


MockToolCallStart = ToolCallStart


@dataclass
class ToolCallProgress:
    """Mock of a ToolCallProgress update.

    Named to match real ACP class so ``type(update).__name__`` works in client.
    Field names match the real ACP ``ToolCallUpdate`` pydantic model.
    """

    tool_call_id: str
    title: str
    status: str  # "completed" | "failed"
    raw_input: Any = None
    raw_output: Any = None


MockToolCallProgress = ToolCallProgress


@dataclass
class MockUsage:
    """Mock of ACP Usage object."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_read_tokens: int | None = None
    thought_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class MockPromptResponse:
    """Mock of ACP PromptResponse."""

    stop_reason: str = "end_turn"
    usage: MockUsage | None = None


@dataclass
class MockModelInfo:
    """Mock of ACP ModelInfo."""

    model_id: str
    name: str
    description: str | None = None


@dataclass
class MockSessionModelState:
    """Mock of ACP SessionModelState."""

    available_models: list[MockModelInfo] = field(default_factory=list)
    current_model_id: str = ""


@dataclass
class MockSessionMode:
    """Mock of ACP SessionMode."""

    id: str
    name: str
    description: str | None = None


@dataclass
class MockSessionModeState:
    """Mock of ACP SessionModeState."""

    available_modes: list[MockSessionMode] = field(default_factory=list)
    current_mode_id: str = ""


@dataclass
class MockNewSessionResponse:
    """Mock of ACP NewSessionResponse."""

    session_id: str = "test-session-123"
    models: MockSessionModelState | None = None
    modes: MockSessionModeState | None = None
    config_options: list[Any] | None = None


@dataclass
class MockInitializeResponse:
    """Mock of ACP InitializeResponse."""

    protocol_version: int = 1
    agent_info: Any = None
    agent_capabilities: Any = None
    auth_methods: list[Any] | None = None


@dataclass
class MockAgentCapabilities:
    """Mock of ACP AgentCapabilities."""

    load_session: bool | None = None
    mcp_capabilities: Any = None


@dataclass
class MockPermissionRequestResponse:
    """Mock of acp.schema.PermissionRequestResponse."""

    approved: bool = True
    reason: str | None = None


@dataclass
class MockReadTextFileResponse:
    """Mock of acp.schema.ReadTextFileResponse."""

    content: str = ""


@dataclass
class MockWriteTextFileResponse:
    """Mock of acp.schema.WriteTextFileResponse."""


@dataclass
class MockSessionSnapshot:
    """Mock of SessionAccumulator snapshot."""

    session_id: str = "test-session-123"
    tool_calls: dict[str, Any] = field(default_factory=dict)
    agent_messages: tuple[Any, ...] = ()
    agent_thoughts: tuple[Any, ...] = ()


class MockSessionAccumulator:
    """Mock of acp.contrib.session_state.SessionAccumulator."""

    def __init__(self) -> None:
        self._notifications: list[Any] = []
        self._snapshot = MockSessionSnapshot()

    def apply(self, notification: Any) -> None:
        self._notifications.append(notification)

    def snapshot(self) -> MockSessionSnapshot:
        return self._snapshot


@dataclass
class MockHttpHeader:
    """Mock of acp.schema.HttpHeader."""

    name: str
    value: str


@dataclass
class MockHttpMcpServer:
    """Mock of acp.schema.HttpMcpServer."""

    url: str
    name: str
    headers: list[Any] = field(default_factory=list)
    type: str = "http"


def make_mock_connection() -> AsyncMock:
    """Create a mock ACP ClientSideConnection."""
    conn = AsyncMock()
    conn.initialize = AsyncMock(return_value=MockInitializeResponse())
    conn.new_session = AsyncMock(return_value=MockNewSessionResponse())
    conn.load_session = AsyncMock()
    conn.prompt = AsyncMock(return_value=MockPromptResponse())
    conn.cancel = AsyncMock()
    conn.set_session_model = AsyncMock()
    conn.set_session_mode = AsyncMock()
    return conn


def make_mock_process() -> MagicMock:
    """Create a mock asyncio.subprocess.Process."""
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdout = MagicMock()
    proc.stderr = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    proc.kill = MagicMock()
    proc.returncode = 0
    return proc


@asynccontextmanager
async def mock_spawn_agent_process(
    client: Any,
    *args: Any,
    conn: AsyncMock | None = None,
    proc: MagicMock | None = None,
    **kwargs: Any,
) -> AsyncIterator[tuple[AsyncMock, MagicMock]]:
    """Mock async context manager for spawn_agent_process."""
    yield (conn or make_mock_connection(), proc or make_mock_process())


@pytest.fixture
def mock_conn() -> AsyncMock:
    return make_mock_connection()


@pytest.fixture
def mock_proc() -> MagicMock:
    return make_mock_process()


@pytest.fixture
def mock_accumulator() -> MockSessionAccumulator:
    return MockSessionAccumulator()
