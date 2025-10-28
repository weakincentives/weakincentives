from __future__ import annotations

from dataclasses import dataclass

import pytest

from typing import cast

from weakincentives import prompts
from weakincentives.prompts import (
    Prompt,
    PromptRenderError,
    PromptValidationError,
    Section,
    SupportsDataclass,
    TextSection,
)


@dataclass
class RoutingParams:
    recipient: str = ""
    subject: str | None = None


@dataclass
class ToneParams:
    tone: str = "neutral"


@dataclass
class ContentParams:
    summary: str


def build_compose_prompt() -> Prompt:
    tone_section = TextSection[ToneParams](
        title="Tone",
        body="""
        Target tone: ${tone}
        """,
    )

    content_section = TextSection[ContentParams](
        title="Content Guidance",
        body="""
        Include the following summary:
        ${summary}
        """,
        enabled=lambda params: bool(params.summary.strip()),
    )

    sections = [
        TextSection[RoutingParams](
            title="Message Routing",
            body="""
            To: ${recipient}
            Subject: ${subject}
            """,
            defaults=RoutingParams(subject="(optional subject)"),
            children=cast(
                list[Section[SupportsDataclass]],
                [tone_section, content_section],
            ),
        ),
    ]

    return Prompt(sections=cast(list[Section[SupportsDataclass]], sections))


def test_prompt_integration_renders_expected_markdown():
    prompt = build_compose_prompt()

    output = prompt.render(
        RoutingParams(recipient="Jordan", subject="Q2 sync"),
        ToneParams(tone="warm"),
        ContentParams(summary="Top takeaways from yesterday's meeting."),
    )

    assert output == "\n\n".join(
        [
            "## Message Routing\n\nTo: Jordan\nSubject: Q2 sync",
            "### Tone\n\nTarget tone: warm",
            "### Content Guidance\n\nInclude the following summary:\nTop takeaways from yesterday's meeting.",
        ]
    )


def test_prompt_integration_handles_disabled_sections():
    prompt = build_compose_prompt()

    output = prompt.render(
        RoutingParams(recipient="Avery"),
        ToneParams(tone="direct"),
        ContentParams(summary="   \n"),
    )

    assert "Content Guidance" not in output
    assert "Target tone: direct" in output


def test_prompt_integration_rejects_mismatched_types():
    prompt = build_compose_prompt()

    with pytest.raises(PromptValidationError):
        prompt.render(RoutingParams)


def test_prompt_integration_propagates_render_errors():
    prompt = build_compose_prompt()

    with pytest.raises(PromptRenderError):
        prompt.render(RoutingParams(recipient="Kim"))


def test_prompt_module_public_exports():
    for symbol in ("Prompt", "Section", "TextSection"):
        assert hasattr(prompts, symbol), f"prompts module missing export: {symbol}"
    assert "Prompt" in prompts.__all__
    assert "Section" in prompts.__all__
    assert "TextSection" in prompts.__all__
