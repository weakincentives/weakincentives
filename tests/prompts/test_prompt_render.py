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
    intro = TextSection(
        title="Intro",
        body="Intro: ${title}",
        params=IntroParams,
    )
    details = TextSection(
        title="Details",
        body="Details: ${body}",
        params=DetailsParams,
    )
    outro = TextSection(
        title="Outro",
        body="Outro: ${footer}",
        params=OutroParams,
        defaults=OutroParams(footer="bye"),
    )
    return Prompt(root_sections=[intro, details, outro])


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
    leaf = TextSection(
        title="Leaf",
        body="Leaf: ${note}",
        params=LeafParams,
    )
    child = TextSection(
        title="Child",
        body="Child detail: ${detail}",
        params=ChildNestedParams,
        children=[leaf],
    )
    parent = TextSection(
        title="Parent",
        body="Parent: ${heading}",
        params=ParentToggleParams,
        children=[child],
        enabled=lambda params: params.include_children,
    )
    summary = TextSection(
        title="Summary",
        body="Summary: ${summary}",
        params=SummaryParams,
    )
    return Prompt(root_sections=[parent, summary])


def test_prompt_render_merges_defaults_and_overrides():
    prompt = build_prompt()

    output = prompt.render(
        params=[
            IntroParams(title="hello"),
            DetailsParams(body="world"),
        ]
    )

    assert output == "\n\n".join(
        [
            "## Intro\n\nIntro: hello",
            "## Details\n\nDetails: world",
            "## Outro\n\nOutro: bye",
        ]
    )


def test_prompt_render_accepts_unordered_inputs():
    prompt = build_prompt()

    output = prompt.render(
        params=[
            DetailsParams(body="unordered"),
            IntroParams(title="still works"),
        ]
    )

    assert "still works" in output
    assert "unordered" in output


def test_prompt_render_requires_parameter_values():
    prompt = build_prompt()

    with pytest.raises(PromptRenderError) as exc:
        prompt.render(params=[IntroParams(title="missing detail")])

    assert isinstance(exc.value, PromptRenderError)
    assert exc.value.dataclass_type is DetailsParams


def test_prompt_render_requires_dataclass_instances():
    prompt = build_prompt()

    with pytest.raises(PromptValidationError) as exc:
        prompt.render(params=[IntroParams])

    assert isinstance(exc.value, PromptValidationError)
    assert exc.value.dataclass_type is IntroParams


def test_prompt_render_renders_nested_sections_and_depth():
    prompt = build_nested_prompt()

    output = prompt.render(
        params=[
            ParentToggleParams(heading="Main Heading", include_children=True),
            ChildNestedParams(detail="Child detail"),
            LeafParams(note="Deep note"),
            SummaryParams(summary="All done"),
        ]
    )

    assert output == "\n\n".join(
        [
            "## Parent\n\nParent: Main Heading",
            "### Child\n\nChild detail: Child detail",
            "#### Leaf\n\nLeaf: Deep note",
            "## Summary\n\nSummary: All done",
        ]
    )


def test_prompt_render_skips_disabled_parent_and_children():
    prompt = build_nested_prompt()

    output = prompt.render(
        params=[
            ParentToggleParams(heading="Unused", include_children=False),
            SummaryParams(summary="Visible"),
        ]
    )

    assert output == "## Summary\n\nSummary: Visible"
    assert "Parent" not in output
    assert "Child" not in output
    assert "Leaf" not in output
