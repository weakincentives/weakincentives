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

"""Tests for typed sections and prompt rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from weakincentives.core.errors import PromptError
from weakincentives.core.prompt import (
    MarkdownSection,
    Prompt,
    Section,
)
from weakincentives.core.tool import Tool, ToolContext, ToolResult


def _noop(params: object, context: ToolContext) -> ToolResult[None]:
    del params, context
    return ToolResult.ok(None)


@dataclass(frozen=True)
class Greeting:
    name: str


def test_invalid_section_key_raises() -> None:
    with pytest.raises(PromptError, match="invalid section key"):
        MarkdownSection[None](title="t", key="UPPER")


def test_invalid_namespace_raises() -> None:
    with pytest.raises(PromptError, match="invalid namespace"):
        Prompt(ns="bad ns", key="k")


def test_invalid_key_raises() -> None:
    with pytest.raises(PromptError, match="invalid key"):
        Prompt(ns="ns", key="bad key")


def test_duplicate_tool_name_raises() -> None:
    tool = Tool(name="dup", description="d", handler=_noop)
    section_a = MarkdownSection[None](title="A", key="a", tools=(tool,))
    section_b = MarkdownSection[None](title="B", key="b", tools=(tool,))
    with pytest.raises(PromptError, match="duplicate tool name"):
        Prompt(ns="ns", key="k", sections=(section_a, section_b))


def test_template_must_use_known_fields() -> None:
    with pytest.raises(PromptError, match="unknown fields"):
        MarkdownSection[Greeting](
            title="G",
            key="g",
            template="hello $unknown",
            params_type=Greeting,
        )


def test_non_dataclass_params_type_rejected() -> None:
    class NotADataclass:
        pass

    with pytest.raises(PromptError, match="must be a dataclass"):
        MarkdownSection[Any](
            title="T",
            key="t",
            template="hi",
            params_type=NotADataclass,
        )


def test_render_with_typed_params() -> None:
    section = MarkdownSection[Greeting](
        title="Greet",
        key="greet",
        template="hello $name",
        params_type=Greeting,
    )
    prompt = Prompt(
        ns="demo",
        key="say",
        sections=(section,),
        params={"greet": Greeting(name="world")},
    )
    rendered = prompt.render()
    assert "## 1. Greet" in rendered.text
    assert "hello world" in rendered.text


def test_render_with_default_params() -> None:
    section = MarkdownSection[Greeting](
        title="Greet",
        key="greet",
        template="hello $name",
        params_type=Greeting,
        default_params=Greeting(name="default"),
    )
    prompt = Prompt(ns="demo", key="say", sections=(section,))
    assert "hello default" in prompt.render().text


def test_render_with_no_template_emits_heading_only() -> None:
    section = MarkdownSection[None](title="Heading", key="heading")
    prompt = Prompt(ns="ns", key="k", sections=(section,))
    assert prompt.render().text == "## 1. Heading"


def test_render_walks_children_with_numbered_headings() -> None:
    leaf = MarkdownSection[None](title="Leaf", key="leaf", template="leaf body")
    root = MarkdownSection[Greeting](
        title="Root",
        key="root",
        template="hi $name",
        params_type=Greeting,
        children=(leaf,),
    )
    prompt = Prompt(
        ns="demo",
        key="k",
        sections=(root,),
        params={"root": Greeting(name="world")},
    )
    text = prompt.render().text
    assert "## 1. Root" in text
    assert "hi world" in text
    assert "### 1.1. Leaf" in text
    assert "leaf body" in text


def test_render_rejects_non_dataclass_params_at_runtime() -> None:
    section = MarkdownSection[Greeting](
        title="G",
        key="g",
        template="hello $name",
        params_type=Greeting,
    )
    prompt = Prompt(
        ns="ns",
        key="k",
        sections=(section,),
        params={"g": "not a dataclass"},
    )
    with pytest.raises(PromptError, match="dataclass instance"):
        prompt.render()


def test_render_reports_missing_placeholder() -> None:
    section = MarkdownSection[None](
        title="T",
        key="t",
        template="hi $name",
    )
    prompt = Prompt(ns="ns", key="k", sections=(section,))
    with pytest.raises(PromptError, match="missing placeholder"):
        prompt.render()


def test_disabled_section_omitted() -> None:
    visible = MarkdownSection[None](title="V", key="v", template="visible")
    hidden = MarkdownSection[None](
        title="H",
        key="h",
        template="hidden",
        enabled=lambda params: False,
    )
    prompt = Prompt(ns="ns", key="k", sections=(visible, hidden))
    text = prompt.render().text
    assert "visible" in text
    assert "hidden" not in text


def test_bind_returns_new_prompt_with_merged_params() -> None:
    section = MarkdownSection[Greeting](
        title="G",
        key="g",
        template="hi $name",
        params_type=Greeting,
    )
    base = Prompt(
        ns="ns",
        key="k",
        sections=(section,),
        params={"g": Greeting(name="alice")},
    )
    bound = base.bind(g=Greeting(name="bob"))
    assert "alice" in base.render().text
    assert "bob" in bound.render().text


def test_section_subclass_renders_via_default_body() -> None:
    section = Section[None](title="Plain", key="plain")
    assert section.render_body(None) == ""


def test_section_expects_returns_field_names() -> None:
    section = MarkdownSection[Greeting](
        title="G",
        key="g",
        template="hi $name",
        params_type=Greeting,
    )
    assert section.expects() == frozenset({"name"})


def test_reachable_tools_walks_children() -> None:
    root_tool = Tool(name="root", description="d", handler=_noop)
    leaf_tool = Tool(name="leaf", description="d", handler=_noop)
    leaf = MarkdownSection[None](title="L", key="leaf", tools=(leaf_tool,))
    root = MarkdownSection[None](
        title="R",
        key="root",
        tools=(root_tool,),
        children=(leaf,),
    )
    names = [tool.name for tool in root.reachable_tools()]
    assert names == ["root", "leaf"]


def test_render_collects_tools() -> None:
    tool = Tool(name="t", description="d", handler=_noop)
    section = MarkdownSection[None](title="S", key="s", tools=(tool,))
    prompt = Prompt(ns="ns", key="k", sections=(section,))
    assert prompt.render().tools == (tool,)


def test_template_with_escaped_dollar_is_allowed() -> None:
    section = MarkdownSection[None](
        title="T",
        key="t",
        template="literal $$ no placeholder",
    )
    prompt = Prompt(ns="ns", key="k", sections=(section,))
    assert "literal $ no placeholder" in prompt.render().text
