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
