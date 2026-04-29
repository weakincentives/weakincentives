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

"""Tests for Tool, ToolResult, and ToolContext."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.core.errors import ToolError
from weakincentives.core.prompt import MarkdownSection, Prompt, RenderedPrompt
from weakincentives.core.session import Session
from weakincentives.core.tool import Tool, ToolContext, ToolResult


@dataclass(frozen=True)
class Echo:
    text: str


def _handler(params: Echo, context: ToolContext) -> ToolResult[str]:
    del context
    return ToolResult.ok(params.text)


def test_tool_result_ok_factory() -> None:
    result = ToolResult.ok(42, message="done")
    assert result.success is True
    assert result.value == 42
    assert result.message == "done"


def test_tool_result_error_factory() -> None:
    result = ToolResult.error("boom")
    assert result.success is False
    assert result.value is None
    assert result.message == "boom"


def test_tool_invalid_name_raises() -> None:
    with pytest.raises(ToolError, match="invalid tool name"):
        Tool(name="BAD NAME", description="d", handler=_handler)


def test_tool_empty_description_raises() -> None:
    with pytest.raises(ToolError, match="must be 1-200"):
        Tool(name="t", description="", handler=_handler)


def test_tool_overlong_description_raises() -> None:
    with pytest.raises(ToolError):
        Tool(name="t", description="x" * 201, handler=_handler)


def test_tool_context_get_resource_without_provider_raises() -> None:
    section = MarkdownSection[None](title="T", key="t")
    prompt = Prompt(ns="ns", key="k", sections=(section,))
    rendered = RenderedPrompt(text="", tools=())
    context = ToolContext(session=Session(), prompt=prompt, rendered=rendered)
    with pytest.raises(LookupError, match="no ResourceProvider"):
        context.get_resource(int)


def test_tool_context_uses_resource_provider() -> None:
    section = MarkdownSection[None](title="T", key="t")
    prompt = Prompt(ns="ns", key="k", sections=(section,))
    rendered = RenderedPrompt(text="", tools=())

    class Provider:
        def get[T](self, protocol_type: type[T]) -> T:
            del protocol_type
            from typing import cast

            return cast("T", "resource")

    context = ToolContext(
        session=Session(),
        prompt=prompt,
        rendered=rendered,
        resources=Provider(),
    )
    assert context.get_resource(str) == "resource"
