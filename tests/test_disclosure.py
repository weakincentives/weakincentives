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

"""Tests for ``weakincentives.disclosure`` and section visibility."""

from __future__ import annotations

from weakincentives.core import (
    MarkdownSection,
    Prompt,
    RenderedPrompt,
    Section,
    SectionVisibility,
    Session,
    ToolContext,
)
from weakincentives.disclosure import (
    SectionExpansions,
    apply_visibility_overrides,
    open_sections_tool,
    read_section_tool,
)


def _make_prompt() -> Prompt:
    short = MarkdownSection[None](
        title="Notes",
        key="notes",
        template="The full notes go here with detail.",
        summary="Notes summary.",
        visibility=SectionVisibility.SUMMARY,
    )
    return Prompt(ns="ns", key="k", sections=(short,))


def _context(prompt: Prompt, session: Session) -> ToolContext:
    rendered = RenderedPrompt(text="", tools=())
    return ToolContext(session=session, prompt=prompt, rendered=rendered)


def test_section_renders_summary_when_visibility_is_summary() -> None:
    text = _make_prompt().render().text
    assert "Notes summary" in text
    assert "full notes" not in text


def test_apply_visibility_overrides_keeps_full_when_no_state() -> None:
    prompt = _make_prompt()
    rebuilt = apply_visibility_overrides(prompt, Session())
    assert rebuilt is prompt


def test_open_sections_tool_expands_named_sections() -> None:
    prompt = _make_prompt()
    session = Session()
    session.install(SectionExpansions, initial=SectionExpansions)
    result = open_sections_tool.handler({"keys": ["notes"]}, _context(prompt, session))
    assert result.success is True
    assert result.value == ("notes",)
    rebuilt = apply_visibility_overrides(prompt, session)
    text = rebuilt.render().text
    assert "full notes" in text


def test_open_sections_tool_validates_input() -> None:
    prompt = _make_prompt()
    session = Session()
    ctx = _context(prompt, session)
    assert open_sections_tool.handler("not a dict", ctx).success is False
    assert open_sections_tool.handler({"keys": "no"}, ctx).success is False
    assert open_sections_tool.handler({"keys": []}, ctx).success is False


def test_read_section_tool_returns_full_template() -> None:
    prompt = _make_prompt()
    result = read_section_tool.handler({"key": "notes"}, _context(prompt, Session()))
    assert result.success is True
    assert "full notes" in str(result.value)


def test_read_section_tool_unknown_key() -> None:
    result = read_section_tool.handler(
        {"key": "missing"}, _context(_make_prompt(), Session())
    )
    assert result.success is False


def test_read_section_tool_validates_input() -> None:
    ctx = _context(_make_prompt(), Session())
    assert read_section_tool.handler("no", ctx).success is False
    assert read_section_tool.handler({"key": 1}, ctx).success is False


def test_visibility_override_walks_into_children() -> None:
    leaf = MarkdownSection[None](
        title="Leaf",
        key="leaf",
        template="leaf detail",
        summary="leaf summary",
        visibility=SectionVisibility.SUMMARY,
    )
    root = MarkdownSection[None](
        title="Root", key="root", template="root", children=(leaf,)
    )
    prompt = Prompt(ns="ns", key="k", sections=(root,))
    session = Session()
    session.install(SectionExpansions, initial=SectionExpansions)
    open_sections_tool.handler({"keys": ["leaf"]}, _context(prompt, session))
    rebuilt = apply_visibility_overrides(prompt, session)
    text = rebuilt.render().text
    assert "leaf detail" in text


def test_visibility_override_returns_same_section_when_no_changes() -> None:
    section = MarkdownSection[None](
        title="Plain",
        key="plain",
        template="body",
    )
    prompt = Prompt(ns="ns", key="k", sections=(section,))
    session = Session()
    session.install(SectionExpansions, initial=SectionExpansions)
    open_sections_tool.handler({"keys": ["other"]}, _context(prompt, session))
    rebuilt = apply_visibility_overrides(prompt, session)
    assert rebuilt.sections[0] is section


def test_section_subclass_render_body_returns_empty() -> None:
    section: Section[None] = Section[None](title="T", key="t")
    assert section.render_body(None) == ""


def test_read_section_finds_nested_section() -> None:
    """Cover the recursion path in _find_in_subtree."""
    deep = MarkdownSection[None](
        title="Deep",
        key="deep",
        template="deep body",
    )
    middle = MarkdownSection[None](
        title="Mid",
        key="mid",
        template="mid body",
        children=(deep,),
    )
    root = MarkdownSection[None](
        title="Root",
        key="root",
        template="root body",
        children=(middle,),
    )
    prompt = Prompt(ns="ns", key="k", sections=(root,))
    result = read_section_tool.handler({"key": "deep"}, _context(prompt, Session()))
    assert result.success is True
    assert "deep body" in str(result.value)


def test_read_section_for_non_markdown_section() -> None:
    """Cover the non-MarkdownSection fallback in _full_body_for."""
    plain = Section[None](title="Plain", key="plain")
    prompt = Prompt(ns="ns", key="k", sections=(plain,))
    result = read_section_tool.handler({"key": "plain"}, _context(prompt, Session()))
    assert result.success is True
    assert result.value == ""


def test_find_section_iterates_past_non_matching_children() -> None:
    """Cover the loop branch in _find_in_subtree where child[0] doesn't match."""
    a = MarkdownSection[None](title="A", key="a", template="a")
    b = MarkdownSection[None](title="B", key="b", template="b")
    root = MarkdownSection[None](
        title="Root", key="root", template="root", children=(a, b)
    )
    prompt = Prompt(ns="ns", key="k", sections=(root,))
    result = read_section_tool.handler({"key": "b"}, _context(prompt, Session()))
    assert result.success is True
    assert "b" in str(result.value)
