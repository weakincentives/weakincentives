from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.prompts import (
    Prompt,
    PromptRenderError,
    PromptValidationError,
    TextSection,
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


def build_prompt() -> Prompt:
    intro = TextSection[IntroParams](
        title="Intro",
        body="Intro: ${title}",
    )
    details = TextSection[DetailsParams](
        title="Details",
        body="Details: ${body}",
    )
    outro = TextSection[OutroParams](
        title="Outro",
        body="Outro: ${footer}",
        defaults=OutroParams(footer="bye"),
    )
    return Prompt(sections=[intro, details, outro])


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


def build_nested_prompt() -> Prompt:
    leaf = TextSection[LeafParams](
        title="Leaf",
        body="Leaf: ${note}",
    )
    child = TextSection[ChildNestedParams](
        title="Child",
        body="Child detail: ${detail}",
        children=[leaf],
    )
    parent = TextSection[ParentToggleParams](
        title="Parent",
        body="Parent: ${heading}",
        children=[child],
        enabled=lambda params: params.include_children,
    )
    summary = TextSection[SummaryParams](
        title="Summary",
        body="Summary: ${summary}",
    )
    return Prompt(sections=[parent, summary])


def test_prompt_render_merges_defaults_and_overrides():
    prompt = build_prompt()

    rendered = prompt.render(
        IntroParams(title="hello"),
        DetailsParams(body="world"),
    )

    assert rendered.text == "\n\n".join(
        [
            "## Intro\n\nIntro: hello",
            "## Details\n\nDetails: world",
            "## Outro\n\nOutro: bye",
        ]
    )


def test_prompt_render_accepts_unordered_inputs():
    prompt = build_prompt()

    rendered = prompt.render(
        DetailsParams(body="unordered"),
        IntroParams(title="still works"),
    )

    assert "still works" in rendered.text
    assert "unordered" in rendered.text


def test_prompt_render_requires_parameter_values():
    prompt = build_prompt()

    with pytest.raises(PromptRenderError) as exc:
        prompt.render(IntroParams(title="missing detail"))

    assert isinstance(exc.value, PromptRenderError)
    assert exc.value.dataclass_type is DetailsParams


def test_prompt_render_requires_dataclass_instances():
    prompt = build_prompt()

    with pytest.raises(PromptValidationError) as exc:
        prompt.render(IntroParams)

    assert isinstance(exc.value, PromptValidationError)
    assert exc.value.dataclass_type is IntroParams


def test_prompt_render_rejects_duplicate_param_instances():
    prompt = build_prompt()

    with pytest.raises(PromptValidationError) as exc:
        prompt.render(IntroParams(title="first"), IntroParams(title="second"))

    assert isinstance(exc.value, PromptValidationError)
    assert exc.value.dataclass_type is IntroParams


def test_prompt_render_renders_nested_sections_and_depth():
    prompt = build_nested_prompt()

    rendered = prompt.render(
        ParentToggleParams(heading="Main Heading", include_children=True),
        ChildNestedParams(detail="Child detail"),
        LeafParams(note="Deep note"),
        SummaryParams(summary="All done"),
    )

    assert rendered.text == "\n\n".join(
        [
            "## Parent\n\nParent: Main Heading",
            "### Child\n\nChild detail: Child detail",
            "#### Leaf\n\nLeaf: Deep note",
            "## Summary\n\nSummary: All done",
        ]
    )


def test_prompt_render_skips_disabled_parent_and_children():
    prompt = build_nested_prompt()

    rendered = prompt.render(
        ParentToggleParams(heading="Unused", include_children=False),
        SummaryParams(summary="Visible"),
    )

    assert rendered.text == "## Summary\n\nSummary: Visible"
    assert "Parent" not in rendered.text
    assert "Child" not in rendered.text
    assert "Leaf" not in rendered.text


def test_prompt_render_wraps_template_errors_with_context():
    @dataclass
    class ErrorParams:
        value: str

    class ExplodingSection(TextSection[ErrorParams]):
        def render(self, params: ErrorParams, depth: int) -> str:
            raise ValueError("boom")

    section = ExplodingSection(
        title="Explode",
        body="unused",
    )
    prompt = Prompt(sections=[section])

    with pytest.raises(PromptRenderError) as exc:
        prompt.render(ErrorParams(value="x"))

    assert isinstance(exc.value, PromptRenderError)
    assert exc.value.section_path == ("Explode",)
    assert exc.value.dataclass_type is ErrorParams


def test_prompt_render_propagates_enabled_errors():
    @dataclass
    class ToggleParams:
        flag: bool

    def raising_enabled(params: ToggleParams) -> bool:
        raise RuntimeError("enabled failure")

    section = TextSection[ToggleParams](
        title="Guard",
        body="Guard: ${flag}",
        enabled=raising_enabled,
    )
    prompt = Prompt(sections=[section])

    with pytest.raises(PromptRenderError) as exc:
        prompt.render(ToggleParams(flag=True))

    assert isinstance(exc.value, PromptRenderError)
    assert exc.value.section_path == ("Guard",)
    assert exc.value.dataclass_type is ToggleParams
