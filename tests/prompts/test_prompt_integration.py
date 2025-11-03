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

"""Integration tests for cross-section prompt composition.

Validation and error-path coverage resides in ``tests/prompts/test_prompt_render.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from weakincentives import prompt
from weakincentives.prompt import MarkdownSection, Prompt


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
    tone_section = MarkdownSection[ToneParams](
        title="Tone",
        template="""
        Target tone: ${tone}
        """,
        key="tone",
    )

    content_section = MarkdownSection[ContentParams](
        title="Content Guidance",
        template="""
        Include the following summary:
        ${summary}
        """,
        key="content-guidance",
        enabled=lambda params: bool(params.summary.strip()),
    )

    sections = [
        MarkdownSection[RoutingParams](
            title="Message Routing",
            template="""
            To: ${recipient}
            Subject: ${subject}
            """,
            key="message-routing",
            default_params=RoutingParams(subject="(optional subject)"),
            children=[tone_section, content_section],
        ),
    ]

    return Prompt(ns="tests/prompts", key="compose-email", sections=sections)


def test_prompt_integration_renders_expected_markdown() -> None:
    prompt = build_compose_prompt()

    rendered = prompt.render(
        RoutingParams(recipient="Jordan", subject="Q2 sync"),
        ToneParams(tone="warm"),
        ContentParams(summary="Top takeaways from yesterday's meeting."),
    )

    assert rendered.text == "\n\n".join(
        [
            "## Message Routing\n\nTo: Jordan\nSubject: Q2 sync",
            "### Tone\n\nTarget tone: warm",
            "### Content Guidance\n\nInclude the following summary:\nTop takeaways from yesterday's meeting.",
        ]
    )


def test_prompt_integration_handles_disabled_sections() -> None:
    prompt = build_compose_prompt()

    rendered = prompt.render(
        RoutingParams(recipient="Avery"),
        ToneParams(tone="direct"),
        ContentParams(summary="   \n"),
    )

    assert "Content Guidance" not in rendered.text
    assert "Target tone: direct" in rendered.text


def test_prompt_module_public_exports() -> None:
    for symbol in ("Prompt", "Section", "MarkdownSection"):
        assert hasattr(prompt, symbol), f"prompt module missing export: {symbol}"
    assert "Prompt" in prompt.__all__
    assert "Section" in prompt.__all__
    assert "MarkdownSection" in prompt.__all__
