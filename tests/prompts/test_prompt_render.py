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
from pathlib import Path
from typing import cast

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptRenderError,
    PromptTemplate,
    PromptValidationError,
    SectionVisibility,
    SupportsDataclass,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.prompt.prompt import (
    RenderedPrompt,
    _format_specialization_argument,
)


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


# Tests for render_section


def test_render_section_returns_single_section() -> None:
    template = build_prompt()

    rendered = (
        Prompt(template)
        .bind(IntroParams(title="hello"), DetailsParams(body="world"))
        .render_section(("intro",))
    )

    assert rendered.path == ("intro",)
    assert rendered.text == "## 1. Intro (intro)\n\nIntro: hello"


def test_render_section_matches_full_render_extraction() -> None:
    template = build_prompt()
    prompt = Prompt(template).bind(
        IntroParams(title="hello"),
        DetailsParams(body="world"),
    )

    full_rendered = prompt.render()
    section_rendered = prompt.render_section(("details",))

    # Extract the section text from full render
    full_sections = full_rendered.text.split("\n\n## ")
    # Find the details section
    details_from_full = None
    for s in full_sections:
        if s.startswith("2. Details"):
            details_from_full = "## " + s
            break

    assert section_rendered.text == details_from_full


def test_render_section_nested_includes_children() -> None:
    template = build_nested_prompt()

    rendered = (
        Prompt(template)
        .bind(
            ParentToggleParams(heading="Main", include_children=True),
            ChildNestedParams(detail="Child detail"),
            LeafParams(note="Leaf note"),
            SummaryParams(summary="Summary"),
        )
        .render_section(("parent",))
    )

    assert rendered.path == ("parent",)
    assert "Parent: Main" in rendered.text
    assert "Child detail: Child detail" in rendered.text
    assert "Leaf: Leaf note" in rendered.text
    assert "Summary" not in rendered.text


def test_render_section_nested_child_only() -> None:
    template = build_nested_prompt()

    rendered = (
        Prompt(template)
        .bind(
            ParentToggleParams(heading="Main", include_children=True),
            ChildNestedParams(detail="Child detail"),
            LeafParams(note="Leaf note"),
            SummaryParams(summary="Summary"),
        )
        .render_section(("parent", "child"))
    )

    assert rendered.path == ("parent", "child")
    assert "Parent" not in rendered.text
    assert "Child detail: Child detail" in rendered.text
    assert "Leaf: Leaf note" in rendered.text


def test_render_section_not_found_raises_error() -> None:
    template = build_prompt()

    with pytest.raises(PromptRenderError) as exc:
        Prompt(template).bind(IntroParams(title="hello")).render_section(
            ("nonexistent",)
        )

    error = exc.value
    assert isinstance(error, PromptRenderError)
    assert error.section_path == ("nonexistent",)


def test_render_section_with_visibility_override() -> None:
    @dataclass
    class SummaryTestParams:
        content: str

    section = MarkdownSection[SummaryTestParams](
        title="Content",
        key="content",
        template="Full content: ${content}",
        summary="Content summary available.",
    )

    template = PromptTemplate(
        ns="tests/prompts",
        key="render-section-visibility",
        sections=[section],
    )

    # Render with SUMMARY visibility
    rendered = (
        Prompt(template)
        .bind(SummaryTestParams(content="details"))
        .render_section(
            ("content",), visibility_overrides={("content",): SectionVisibility.SUMMARY}
        )
    )

    assert "Content summary available." in rendered.text
    assert "Full content:" not in rendered.text


def test_render_section_visibility_override_full() -> None:
    @dataclass
    class SummaryTestParams:
        content: str

    section = MarkdownSection[SummaryTestParams](
        title="Content",
        key="content",
        template="Full content: ${content}",
        summary="Content summary available.",
        visibility=SectionVisibility.SUMMARY,  # Default to summary
    )

    template = PromptTemplate(
        ns="tests/prompts",
        key="render-section-visibility-full",
        sections=[section],
    )

    # Render with FULL visibility override
    rendered = (
        Prompt(template)
        .bind(SummaryTestParams(content="details"))
        .render_section(
            ("content",), visibility_overrides={("content",): SectionVisibility.FULL}
        )
    )

    assert "Full content: details" in rendered.text
    assert "Content summary available." not in rendered.text


@dataclass
class _RenderSectionToolParams:
    value: str


@dataclass
class _RenderSectionToolResult:
    answer: str


def _render_section_tool_handler(
    params: _RenderSectionToolParams, *, context: ToolContext
) -> ToolResult[_RenderSectionToolResult]:
    del context
    return ToolResult(
        message="ok",
        value=_RenderSectionToolResult(answer=params.value),
        success=True,
    )


def test_render_section_collects_tools() -> None:
    section = MarkdownSection[_RenderSectionToolParams](
        title="With Tool",
        key="with-tool",
        template="Value: ${value}",
        tools=[
            Tool[_RenderSectionToolParams, _RenderSectionToolResult](
                name="my_tool",
                description="A test tool",
                handler=_render_section_tool_handler,
            )
        ],
    )

    template = PromptTemplate(
        ns="tests/prompts",
        key="render-section-tools",
        sections=[section],
    )

    rendered = (
        Prompt(template)
        .bind(_RenderSectionToolParams(value="test"))
        .render_section(("with-tool",))
    )

    assert len(rendered.tools) == 1
    assert rendered.tools[0].name == "my_tool"


def test_render_section_excludes_tools_with_summary_visibility() -> None:
    section = MarkdownSection[_RenderSectionToolParams](
        title="With Tool",
        key="with-tool",
        template="Full: ${value}",
        summary="Summary available.",
        tools=[
            Tool[_RenderSectionToolParams, _RenderSectionToolResult](
                name="my_tool_summary",
                description="A test tool",
                handler=_render_section_tool_handler,
            )
        ],
    )

    template = PromptTemplate(
        ns="tests/prompts",
        key="render-section-tools-summary",
        sections=[section],
    )

    # Render with SUMMARY visibility - tools should be excluded
    rendered = (
        Prompt(template)
        .bind(_RenderSectionToolParams(value="test"))
        .render_section(
            ("with-tool",),
            visibility_overrides={("with-tool",): SectionVisibility.SUMMARY},
        )
    )

    assert len(rendered.tools) == 0


def test_rendered_section_str_returns_text() -> None:
    from weakincentives.prompt import RenderedSection

    rendered = RenderedSection(text="Section output", path=("test",))

    assert str(rendered) == "Section output"


def test_rendered_section_tool_param_descriptions() -> None:
    from weakincentives.prompt import RenderedSection

    descriptions = {"my_tool": {"param1": "Description for param1"}}
    rendered = RenderedSection(
        text="Section output",
        path=("test",),
        _tool_param_descriptions=descriptions,
    )

    assert rendered.tool_param_descriptions == descriptions


def test_render_section_skips_children_when_parent_has_summary() -> None:
    """Test that children are skipped when parent section renders with SUMMARY."""

    @dataclass
    class ParentParams:
        title: str

    @dataclass
    class ChildParams:
        detail: str

    child = MarkdownSection[ChildParams](
        title="Child",
        key="child",
        template="Child detail: ${detail}",
    )

    parent = MarkdownSection[ParentParams](
        title="Parent",
        key="parent",
        template="Full parent: ${title}",
        summary="Parent summary.",
        children=[child],
    )

    template = PromptTemplate(
        ns="tests/prompts",
        key="render-section-skip-children",
        sections=[parent],
    )

    # Render parent with SUMMARY visibility - children should be skipped
    rendered = (
        Prompt(template)
        .bind(ParentParams(title="Test"), ChildParams(detail="Skipped"))
        .render_section(
            ("parent",),
            visibility_overrides={("parent",): SectionVisibility.SUMMARY},
        )
    )

    # Parent summary should be present, but not the child detail
    assert "Parent summary." in rendered.text
    assert "Child detail" not in rendered.text


def test_render_section_summary_followed_by_sibling() -> None:
    """Test that sibling sections after a SUMMARY section are rendered."""

    @dataclass
    class FirstChildParams:
        first: str

    @dataclass
    class GrandchildParams:
        grandchild: str

    @dataclass
    class SecondChildParams:
        second: str

    @dataclass
    class ContainerParams:
        container: str

    grandchild = MarkdownSection[GrandchildParams](
        title="Grandchild",
        key="grandchild",
        template="Grandchild: ${grandchild}",
    )

    first_child = MarkdownSection[FirstChildParams](
        title="First",
        key="first",
        template="Full first: ${first}",
        summary="First summary.",
        children=[grandchild],
    )

    second_child = MarkdownSection[SecondChildParams](
        title="Second",
        key="second",
        template="Second: ${second}",
    )

    container = MarkdownSection[ContainerParams](
        title="Container",
        key="container",
        template="Container: ${container}",
        children=[first_child, second_child],
    )

    template = PromptTemplate(
        ns="tests/prompts",
        key="render-section-summary-sibling",
        sections=[container],
    )

    # Render container with first child as SUMMARY - second child should still render
    rendered = (
        Prompt(template)
        .bind(
            ContainerParams(container="main"),
            FirstChildParams(first="first"),
            GrandchildParams(grandchild="grand"),
            SecondChildParams(second="second"),
        )
        .render_section(
            ("container",),
            visibility_overrides={("container", "first"): SectionVisibility.SUMMARY},
        )
    )

    # First child summary should be present
    assert "First summary." in rendered.text
    # Grandchild should be skipped
    assert "Grandchild" not in rendered.text
    # Second child should be rendered
    assert "Second: second" in rendered.text


def test_render_section_with_overrides_store(tmp_path: Path) -> None:
    """Test render_section respects overrides from an overrides store."""
    from weakincentives.prompt import LocalPromptOverridesStore

    @dataclass
    class OverrideTestParams:
        content: str

    section = MarkdownSection[OverrideTestParams](
        title="Overridable",
        key="overridable",
        template="Original: ${content}",
    )

    template = PromptTemplate(
        ns="tests/prompts",
        key="render-section-override-store",
        sections=[section],
    )

    # Create overrides store with overridden body
    store = LocalPromptOverridesStore(root_path=tmp_path)
    prompt = Prompt(template, overrides_store=store).bind(
        OverrideTestParams(content="test")
    )

    # First render without overrides
    rendered_original = prompt.render_section(("overridable",))
    assert "Original: test" in rendered_original.text

    # Set an override for the section
    store.set_section_override(
        prompt,
        tag="latest",
        path=("overridable",),
        body="Overridden: ${content}",
    )

    # Render with overrides
    rendered_overridden = prompt.render_section(("overridable",))
    assert "Overridden: test" in rendered_overridden.text
    assert "Original" not in rendered_overridden.text
