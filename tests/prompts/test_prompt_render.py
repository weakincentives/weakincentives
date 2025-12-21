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

"""Unit tests covering prompt rendering, validation, and error handling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptRenderError,
    PromptTemplate,
    PromptValidationError,
    SectionVisibility,
)
from weakincentives.prompt.prompt import (
    RenderedPrompt,
    _format_specialization_argument,
)
from weakincentives.types import SupportsDataclass


@dataclass
class IntroParams:
    title: str


@dataclass
class DetailsParams:
    body: str


@dataclass
class OutroParams:
    footer: str


def build_prompt() -> PromptTemplate:
    intro = MarkdownSection[IntroParams](
        title="Intro",
        template="Intro: ${title}",
        key="intro",
    )
    details = MarkdownSection[DetailsParams](
        title="Details",
        template="Details: ${body}",
        key="details",
    )
    outro = MarkdownSection[OutroParams](
        title="Outro",
        template="Outro: ${footer}",
        key="outro",
        default_params=OutroParams(footer="bye"),
    )
    return PromptTemplate(
        ns="tests/prompts",
        key="render-basic",
        sections=[intro, details, outro],
    )


@dataclass
class ParentToggleParams:
    heading: str
    include_children: bool


@dataclass
class ChildNestedParams:
    detail: str


@dataclass
class LeafParams:
    note: str


@dataclass
class SummaryParams:
    summary: str


def test_prompt_renders_section_without_params() -> None:
    static_section = MarkdownSection(
        title="Static", key="static", template="Static content."
    )
    template = PromptTemplate(
        ns="tests.prompts",
        key="paramless-section",
        sections=(static_section,),
    )

    rendered = Prompt(template).render()

    assert rendered.text.strip().endswith("Static content.")


def test_prompt_rejects_placeholders_for_paramless_section() -> None:
    section = MarkdownSection(title="Bad", key="bad", template="${value}")

    with pytest.raises(PromptValidationError):
        PromptTemplate(ns="tests.prompts", key="bad-section", sections=(section,))


def build_nested_prompt() -> PromptTemplate:
    leaf = MarkdownSection[LeafParams](
        title="Leaf",
        template="Leaf: ${note}",
        key="leaf",
    )
    child = MarkdownSection[ChildNestedParams](
        title="Child",
        template="Child detail: ${detail}",
        key="child",
        children=[leaf],
    )
    parent = MarkdownSection[ParentToggleParams](
        title="Parent",
        template="Parent: ${heading}",
        key="parent",
        children=[child],
        enabled=lambda params: params.include_children,
    )
    summary = MarkdownSection[SummaryParams](
        title="Summary",
        template="Summary: ${summary}",
        key="summary",
    )
    return PromptTemplate(
        ns="tests/prompts",
        key="render-nested",
        sections=[parent, summary],
    )


def test_prompt_render_merges_defaults_and_overrides() -> None:
    template = build_prompt()

    rendered = (
        Prompt(template)
        .bind(
            IntroParams(title="hello"),
            DetailsParams(body="world"),
        )
        .render()
    )

    assert rendered.text == "\n\n".join(
        [
            "## 1. Intro (intro)\n\nIntro: hello",
            "## 2. Details (details)\n\nDetails: world",
            "## 3. Outro (outro)\n\nOutro: bye",
        ]
    )


def test_prompt_render_accepts_unordered_inputs() -> None:
    template = build_prompt()

    rendered = (
        Prompt(template)
        .bind(
            DetailsParams(body="unordered"),
            IntroParams(title="still works"),
        )
        .render()
    )

    assert "still works" in rendered.text
    assert "unordered" in rendered.text


def test_prompt_render_requires_parameter_values() -> None:
    template = build_prompt()

    with pytest.raises(PromptRenderError) as exc:
        Prompt(template).bind(IntroParams(title="missing detail")).render()

    assert isinstance(exc.value, PromptRenderError)
    assert exc.value.dataclass_type is DetailsParams


def test_prompt_render_iterates_through_siblings() -> None:
    parent1 = MarkdownSection[IntroParams](
        title="Parent1",
        template="Parent1: ${title}",
        key="parent1",
    )
    child1 = MarkdownSection[DetailsParams](
        title="Child1",
        template="Child1: ${body}",
        key="child1",
    )
    parent2 = MarkdownSection[OutroParams](
        title="Parent2",
        template="Parent2: ${footer}",
        key="parent2",
    )
    child2 = MarkdownSection[SummaryParams](
        title="Child2",
        template="Child2: ${summary}",
        key="child2",
    )

    template = PromptTemplate(
        ns="tests/prompts",
        key="siblings",
        sections=[
            MarkdownSection(
                title="Root",
                key="root",
                template="Root",
                children=[parent1, child1, parent2, child2],
            )
        ],
    )

    rendered = (
        Prompt(template)
        .bind(
            IntroParams(title="a"),
            DetailsParams(body="b"),
            OutroParams(footer="c"),
            SummaryParams(summary="d"),
        )
        .render()
    )

    assert "Parent1: a" in rendered.text
    assert "Child1: b" in rendered.text
    assert "Parent2: c" in rendered.text
    assert "Child2: d" in rendered.text


def test_prompt_render_requires_dataclass_instances() -> None:
    template = build_prompt()

    with pytest.raises(PromptValidationError) as exc:
        Prompt(template).bind(cast(SupportsDataclass, IntroParams)).render()

    assert isinstance(exc.value, PromptValidationError)
    assert exc.value.dataclass_type is IntroParams


def test_prompt_render_rejects_duplicate_param_instances() -> None:
    template = build_prompt()

    with pytest.raises(PromptValidationError) as exc:
        Prompt(template).bind(
            IntroParams(title="first"), IntroParams(title="second")
        ).render()

    assert isinstance(exc.value, PromptValidationError)
    assert exc.value.dataclass_type is IntroParams


def test_prompt_render_renders_nested_sections_and_depth() -> None:
    template = build_nested_prompt()

    rendered = (
        Prompt(template)
        .bind(
            ParentToggleParams(heading="Main Heading", include_children=True),
            ChildNestedParams(detail="Child detail"),
            LeafParams(note="Deep note"),
            SummaryParams(summary="All done"),
        )
        .render()
    )

    assert rendered.text == "\n\n".join(
        [
            "## 1. Parent (parent)\n\nParent: Main Heading",
            "### 1.1. Child (parent.child)\n\nChild detail: Child detail",
            "#### 1.1.1. Leaf (parent.child.leaf)\n\nLeaf: Deep note",
            "## 2. Summary (summary)\n\nSummary: All done",
        ]
    )


def test_prompt_render_skips_disabled_parent_and_children() -> None:
    template = build_nested_prompt()

    rendered = (
        Prompt(template)
        .bind(
            ParentToggleParams(heading="Unused", include_children=False),
            SummaryParams(summary="Visible"),
        )
        .render()
    )

    assert rendered.text == "## 2. Summary (summary)\n\nSummary: Visible"
    assert "Parent" not in rendered.text
    assert "Child" not in rendered.text
    assert "Leaf" not in rendered.text


def test_prompt_render_wraps_template_errors_with_context() -> None:
    @dataclass
    class ErrorParams:
        value: str

    class ExplodingSection(MarkdownSection[ErrorParams]):
        def render(
            self,
            params: ErrorParams,
            depth: int,
            number: str,
            *,
            path: tuple[str, ...] = (),
            visibility: SectionVisibility | None = None,
        ) -> str:
            del params, depth, number, path, visibility
            raise ValueError(f"boom:{self.title}")

    section = ExplodingSection(
        title="Explode",
        template="unused",
        key="explode",
    )
    template = PromptTemplate(
        ns="tests/prompts", key="render-error", sections=[section]
    )

    with pytest.raises(PromptRenderError) as exc:
        Prompt(template).bind(ErrorParams(value="x")).render()

    assert isinstance(exc.value, PromptRenderError)
    assert exc.value.section_path == ("explode",)
    assert exc.value.dataclass_type is ErrorParams


def test_prompt_render_propagates_enabled_errors() -> None:
    @dataclass
    class ToggleParams:
        flag: bool

    def raising_enabled(params: ToggleParams) -> bool:
        raise RuntimeError("enabled failure")

    section = MarkdownSection[ToggleParams](
        title="Guard",
        template="Guard: ${flag}",
        key="guard",
        enabled=cast(Callable[[SupportsDataclass], bool], raising_enabled),
    )
    template = PromptTemplate(
        ns="tests/prompts",
        key="render-enabled-error",
        sections=[section],
    )

    with pytest.raises(PromptRenderError) as exc:
        Prompt(template).bind(ToggleParams(flag=True)).render()

    assert isinstance(exc.value, PromptRenderError)
    assert exc.value.section_path == ("guard",)
    assert exc.value.dataclass_type is ToggleParams


def test_rendered_prompt_str_returns_text() -> None:
    rendered = RenderedPrompt(text="Rendered output")

    assert str(rendered) == "Rendered output"


def test_prompt_bind_mutates_and_replaces_params() -> None:
    prompt = PromptTemplate(
        ns="tests/prompts",
        key="bind-mutation",
        sections=[
            MarkdownSection[IntroParams](title="Intro", template="", key="intro")
        ],
    )
    bound = Prompt(prompt)

    assert bound.bind() is bound  # no-op, identity
    assert bound.params == ()

    first = IntroParams(title="v1")
    second = IntroParams(title="v2")

    assert bound.bind(first) is bound
    assert bound.params == (first,)

    assert bound.bind(second) is bound
    assert bound.params == (second,)


def test_format_specialization_argument_variants() -> None:
    assert _format_specialization_argument(None) == "?"
    assert _format_specialization_argument(int) == "int"
    assert _format_specialization_argument({"id": 1}) == "{'id': 1}"


def test_markdown_section_missing_placeholder_raises_prompt_error() -> None:
    section = MarkdownSection[IntroParams](
        title="Intro",
        template="Hello ${name}",
        key="intro",
    )

    with pytest.raises(PromptRenderError) as exc:
        section.render(IntroParams(title="ignored"), depth=0, number="1")

    assert isinstance(exc.value, PromptRenderError)
    assert exc.value.placeholder == "name"


def test_section_rendering_empty_string_is_skipped() -> None:
    """Test that sections rendering to empty strings are not included in output."""

    class EmptySection(MarkdownSection[IntroParams]):
        def render(
            self,
            params: IntroParams,
            depth: int,
            number: str,
            *,
            path: tuple[str, ...] = (),
            visibility: SectionVisibility | None = None,
        ) -> str:
            del params, depth, number, path, visibility
            return ""  # Empty render

    empty_section = EmptySection(
        title="Empty",
        template="unused",
        key="empty",
    )
    non_empty = MarkdownSection[DetailsParams](
        title="Content",
        template="Content: ${body}",
        key="content",
    )

    template = PromptTemplate(
        ns="tests/prompts",
        key="empty-section",
        sections=[empty_section, non_empty],
    )

    rendered = (
        Prompt(template)
        .bind(
            IntroParams(title="ignored"),
            DetailsParams(body="visible"),
        )
        .render()
    )

    # Empty section should not appear in output
    assert "Empty" not in rendered.text
    assert "Content: visible" in rendered.text


def test_collect_child_keys_exits_on_sibling() -> None:
    """Test _collect_child_keys stops when encountering siblings."""

    parent = MarkdownSection[IntroParams](
        title="Parent",
        template="Parent: ${title}",
        key="parent",
        children=[
            MarkdownSection[DetailsParams](
                title="Child1",
                template="Child1: ${body}",
                key="child1",
            ),
            MarkdownSection[OutroParams](
                title="Child2",
                template="Child2: ${footer}",
                key="child2",
            ),
        ],
    )
    sibling = MarkdownSection[SummaryParams](
        title="Sibling",
        template="Sibling: ${summary}",
        key="sibling",
    )

    template = PromptTemplate(
        ns="tests/prompts",
        key="collect-children",
        sections=[parent, sibling],
    )

    rendered = (
        Prompt(template)
        .bind(
            IntroParams(title="p"),
            DetailsParams(body="c1"),
            OutroParams(footer="c2"),
            SummaryParams(summary="s"),
        )
        .render()
    )

    # All sections should render
    assert "Parent: p" in rendered.text
    assert "Child1: c1" in rendered.text
    assert "Child2: c2" in rendered.text
    assert "Sibling: s" in rendered.text


def test_collect_child_keys_skips_grandchildren() -> None:
    """Test _collect_child_keys skips grandchildren and continues to next child.

    This covers the branch 339->329 in rendering.py where we iterate past
    grandchildren to find subsequent direct children.
    """

    grandchild = MarkdownSection[OutroParams](
        title="Grandchild",
        template="Grandchild: ${footer}",
        key="grandchild",
    )
    child1 = MarkdownSection[DetailsParams](
        title="Child1",
        template="Child1: ${body}",
        key="child1",
        children=[grandchild],
    )
    child2 = MarkdownSection[SummaryParams](
        title="Child2",
        template="Child2: ${summary}",
        key="child2",
    )
    parent = MarkdownSection[IntroParams](
        title="Parent",
        template="Parent: ${title}",
        key="parent",
        summary="Parent summary for SUMMARY visibility",
        visibility=SectionVisibility.SUMMARY,
        children=[child1, child2],
    )

    template = PromptTemplate(
        ns="tests/prompts",
        key="grandchildren-skip",
        sections=[parent],
    )

    rendered = (
        Prompt(template)
        .bind(
            IntroParams(title="p"),
            DetailsParams(body="c1"),
            OutroParams(footer="gc"),
            SummaryParams(summary="c2"),
        )
        .render()
    )

    # Parent should render with SUMMARY visibility (shows summary suffix)
    assert "Parent summary" in rendered.text
    # Children and grandchild are hidden in SUMMARY mode
    # The _collect_child_keys method should skip grandchild and find child2


def test_markdown_section_placeholder_extraction_with_braced_syntax() -> None:
    """Test that MarkdownSection extracts placeholders from ${var} syntax."""
    section = MarkdownSection[IntroParams](
        title="Test",
        template="Title: ${title}",
        key="test",
    )

    # This should extract 'title' placeholder using braced syntax
    params = IntroParams(title="Hello")
    prompt = Prompt(PromptTemplate(ns="test", key="braced", sections=[section])).bind(
        params
    )

    rendered = prompt.render()
    assert "Title: Hello" in rendered.text
