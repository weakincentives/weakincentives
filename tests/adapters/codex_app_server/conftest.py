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

"""Shared fixtures and helpers for Codex App Server adapter tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

from weakincentives.adapters.codex_app_server.client import CodexAppServerClient
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate, Tool
from weakincentives.prompt.tool import ToolContext, ToolResult
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session

# =============================================================================
# Fake Process / Stream helpers for client tests
# =============================================================================


class FakeStreamReader:
    """Simulates an asyncio.StreamReader for testing."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self._index = 0

    async def readline(self) -> bytes:
        if self._index >= len(self._lines):
            return b""  # EOF
        line = self._lines[self._index]
        self._index += 1
        return (line + "\n").encode()


class FakeStreamWriter:
    """Simulates an asyncio.StreamWriter for testing."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    """Simulates an asyncio.subprocess.Process for testing."""

    def __init__(
        self,
        stdout_lines: list[str] | None = None,
        stderr_lines: list[str] | None = None,
    ) -> None:
        self.stdin = FakeStreamWriter()
        self.stdout = FakeStreamReader(stdout_lines or [])
        self.stderr = FakeStreamReader(stderr_lines or [])
        self._killed = False
        self._waited = False

    async def wait(self) -> int:
        self._waited = True
        return 0

    def kill(self) -> None:
        self._killed = True


class _StubReadTask:
    """Minimal task-like object exposing ``done()`` for send_request tests."""

    def __init__(self, done_values: list[bool]) -> None:
        self._done_values = done_values
        self._index = 0

    def done(self) -> bool:
        if self._index >= len(self._done_values):
            return self._done_values[-1]
        value = self._done_values[self._index]
        self._index += 1
        return value


def _response_line(req_id: int, result: dict[str, Any] | None = None) -> str:
    msg: dict[str, Any] = {"id": req_id}
    if result is not None:
        msg["result"] = result
    return json.dumps(msg)


def _notification_line(method: str, params: dict[str, Any] | None = None) -> str:
    msg: dict[str, Any] = {"method": method}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg)


def _server_request_line(
    req_id: int, method: str, params: dict[str, Any] | None = None
) -> str:
    msg: dict[str, Any] = {"id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg)


def make_session() -> tuple[Session, InProcessDispatcher]:
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher, tags={"suite": "tests"})
    return session, dispatcher


def make_simple_prompt(name: str = "test-prompt") -> Prompt[object]:
    template: PromptTemplate[object] = PromptTemplate(
        ns="test",
        key="basic",
        sections=(),
        name=name,
    )
    return Prompt(template)


@dataclass(slots=True, frozen=True)
class AddParams:
    x: int
    y: int


@dataclass(slots=True, frozen=True)
class AddResult:
    sum: int


def _add_handler(params: AddParams, *, context: ToolContext) -> ToolResult[AddResult]:
    return ToolResult.ok(
        AddResult(sum=params.x + params.y), message=str(params.x + params.y)
    )


ADD_TOOL = Tool[AddParams, AddResult](
    name="add",
    description="Add two numbers",
    handler=_add_handler,
)


def make_prompt_with_tool(name: str = "tool-prompt") -> Prompt[object]:
    section = MarkdownSection(
        title="Tools",
        template="Use the tools below.",
        key="tools",
        tools=[ADD_TOOL],
    )
    template: PromptTemplate[object] = PromptTemplate(
        ns="test",
        key="with-tool",
        sections=(section,),
        name=name,
    )
    return Prompt(template)


def make_mock_client() -> AsyncMock:
    """Create a mock CodexAppServerClient."""
    client = AsyncMock(spec=CodexAppServerClient)
    client.stderr_output = ""
    client.start = AsyncMock()
    client.stop = AsyncMock()
    client.send_request = AsyncMock(return_value={})
    client.send_notification = AsyncMock()
    client.send_response = AsyncMock()
    return client


def messages_iterator(
    messages: list[dict[str, Any]],
) -> Any:
    """Create an async iterator from a list of messages."""

    async def _iter() -> Any:
        for msg in messages:
            yield msg

    return _iter()
