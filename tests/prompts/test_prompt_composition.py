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

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import TypeVar, cast

import pytest

from weakincentives.deadlines import Deadline
from weakincentives.prompt import (
    DelegationParams,
    DelegationPrompt,
    MarkdownSection,
    ParentPromptParams,
    ParentPromptSection,
    Prompt,
    PromptRenderError,
    RecapParams,
    RecapSection,
    Tool,
)
from weakincentives.prompt.composition import _merge_tool_param_descriptions
from weakincentives.prompt.overrides import ToolParamDescription
from weakincentives.prompt.protocols import PromptProtocol

ParentProtocolT = TypeVar("ParentProtocolT")


def _as_prompt_protocol[ParentProtocolT](
    prompt: Prompt[ParentProtocolT],
) -> PromptProtocol[ParentProtocolT]:
    return cast(PromptProtocol[ParentProtocolT], prompt)


@dataclass
class ParentResult:
    message: str


@dataclass
class DelegationPlan:
    steps: tuple[str, ...]


@dataclass
class ParentSectionParams:
    guidance: str


@dataclass
class FilesystemParams:
    path: str


@dataclass
class FilesystemResult:
    contents: str


def _build_parent_prompt() -> tuple[
    Prompt[ParentResult], Tool[FilesystemParams, FilesystemResult]
]:
    tool = Tool[FilesystemParams, FilesystemResult](
        name="filesystem_inspect",
        description="Inspect filesystem contents.",
        handler=None,
    )
    section = MarkdownSection[ParentSectionParams](
        title="Investigation",
        key="investigation",
        template=(
            "You are investigating the repository for ${guidance}."
            "\nUse filesystem_inspect when you need directory listings."
        ),
        tools=[tool],
    )
    prompt = Prompt[ParentResult](
        ns="tests/prompts",
        key="parent",
        sections=(section,),
    )
    return prompt, tool


def test_delegation_prompt_renders_required_sections() -> None:
    parent_prompt, tool = _build_parent_prompt()
    rendered_parent = parent_prompt.render(ParentSectionParams(guidance="clues"))
    parent_with_descriptions = replace(
        rendered_parent,
        _tool_param_descriptions=MappingProxyType(
            {
                tool.name: MappingProxyType(
                    {"path": ToolParamDescription(description="Absolute path to inspect.")}
                )
            }
        ),
    )

    delegation = DelegationPrompt[ParentResult, DelegationPlan](
        _as_prompt_protocol(parent_prompt),
        parent_with_descriptions,
    )

    assert delegation.prompt.key == "parent-wrapper"

    rendered = delegation.render(
        DelegationParams(
            reason="The parent agent needs a focused investigation.",
            expected_result="A prioritized list of next actions.",
            may_delegate_further="no",
            recap_lines=("Prioritize areas touched in the latest commits.",),
        )
    )

    assert rendered.output_type is DelegationPlan
    assert rendered.container == "object"
    assert rendered.allow_extra_keys is False
    assert rendered.tools == (tool,)
    assert rendered.tool_param_descriptions == {
        tool.name: {
            "path": ToolParamDescription(description="Absolute path to inspect.")
        }
    }

    text = rendered.text
    assert "# Delegation Summary" in text
    summary_segment = text.split("<!-- PARENT PROMPT START -->", 1)[0]
    assert "## Response Format" not in summary_segment
    assert "## Parent Prompt (Verbatim)" in text
    assert "<!-- PARENT PROMPT START -->" in text
    assert rendered_parent.text in text
    assert "## Recap" in text
    assert "- Prioritize areas touched in the latest commits." in text


def test_delegation_prompt_with_response_format_instructions() -> None:
    parent_prompt, _ = _build_parent_prompt()
    rendered_parent = parent_prompt.render(
        ParentSectionParams(guidance="access patterns")
    )

    delegation = DelegationPrompt[ParentResult, DelegationPlan](
        _as_prompt_protocol(parent_prompt),
        rendered_parent,
        include_response_format=True,
    )

    rendered = delegation.render(
        DelegationParams(
            reason="Specialize on filesystem enumeration.",
            expected_result="Detailed plan for the next commit.",
            may_delegate_further="no",
            recap_lines=("Keep notes concise.",),
        ),
        parent=ParentPromptParams(body=rendered_parent.text),
    )

    assert "## Response Format" in rendered.text
    assert "Return ONLY a single fenced JSON code block." in rendered.text
    assert "an object" in rendered.text
    assert rendered.text.index("## Delegation Summary") < rendered.text.index(
        "## Parent Prompt (Verbatim)"
    )


def test_delegation_prompt_allows_explicit_recap_override() -> None:
    parent_prompt, _ = _build_parent_prompt()
    rendered_parent = parent_prompt.render(ParentSectionParams(guidance="logs"))

    delegation = DelegationPrompt[ParentResult, DelegationPlan](
        _as_prompt_protocol(parent_prompt),
        rendered_parent,
    )

    rendered = delegation.render(
        DelegationParams(
            reason="Need assistance reviewing logs.",
            expected_result="Summary of suspicious findings.",
            may_delegate_further="yes",
            recap_lines=("Focus on log rotation issues.",),
        ),
        recap=RecapParams(bullets=("Inspect authentication failures.",)),
    )

    assert "- Inspect authentication failures." in rendered.text
    assert "- Focus on log rotation issues." not in rendered.text


def test_delegation_prompt_requires_specialization() -> None:
    parent_prompt, _ = _build_parent_prompt()
    rendered_parent = parent_prompt.render(ParentSectionParams(guidance="clues"))

    with pytest.raises(TypeError):
        DelegationPrompt(_as_prompt_protocol(parent_prompt), rendered_parent)


def test_delegation_prompt_skips_fallback_when_parent_freeform() -> None:
    section = MarkdownSection[ParentSectionParams](
        title="Investigation",
        key="investigation",
        template="Focus on ${guidance}.",
    )
    prompt = Prompt(
        ns="tests/prompts",
        key="parent-freeform",
        sections=(section,),
    )
    rendered_parent = prompt.render(ParentSectionParams(guidance="logs"))
    assert rendered_parent.container is None

    delegation = DelegationPrompt[ParentResult, DelegationPlan](
        _as_prompt_protocol(prompt),
        rendered_parent,
        include_response_format=True,
    )
    rendered = delegation.render(
        DelegationParams(
            reason="Gather raw observations.",
            expected_result="A quick log summary.",
            may_delegate_further="no",
            recap_lines=("Capture high-level anomalies.",),
        ),
        parent=ParentPromptParams(body=rendered_parent.text),
    )

    assert "## Response Format" not in rendered.text


def test_parent_prompt_section_requires_params() -> None:
    section = ParentPromptSection()

    with pytest.raises(PromptRenderError):
        section.render(None, depth=0)


def test_recap_section_requires_params() -> None:
    section = RecapSection()

    with pytest.raises(PromptRenderError):
        section.render(None, depth=0)


def test_delegation_prompt_empty_recap_uses_placeholder() -> None:
    parent_prompt, _ = _build_parent_prompt()
    rendered_parent = parent_prompt.render(ParentSectionParams(guidance="signals"))

    delegation = DelegationPrompt[ParentResult, DelegationPlan](
        _as_prompt_protocol(parent_prompt),
        rendered_parent,
    )

    rendered = delegation.render(
        DelegationParams(
            reason="Capture key signals.",
            expected_result="List of signal hypotheses.",
            may_delegate_further="yes",
            recap_lines=(),
        )
    )

    assert "## Recap" in rendered.text
    recap_segment = rendered.text.split("## Recap", 1)[-1]
    assert "- " not in recap_segment


def test_delegation_prompt_inherits_deadline() -> None:
    parent_prompt, _ = _build_parent_prompt()
    rendered_parent = parent_prompt.render(ParentSectionParams(guidance="signals"))
    deadline = Deadline(datetime.now(UTC) + timedelta(seconds=5))
    rendered_parent_with_deadline = replace(rendered_parent, deadline=deadline)

    delegation = DelegationPrompt[ParentResult, DelegationPlan](
        _as_prompt_protocol(parent_prompt),
        rendered_parent_with_deadline,
    )

    rendered_child = delegation.render(
        DelegationParams(
            reason="Inspect deadlines.",
            expected_result="A confirmation.",
            may_delegate_further="no",
            recap_lines=("Ensure deadlines propagate.",),
        )
    )

    assert rendered_child.deadline is deadline


def test_merge_tool_param_descriptions_combines_entries() -> None:
    parent = MappingProxyType(
        {
            "filesystem_inspect": MappingProxyType(
                {"path": ToolParamDescription(description="Absolute path to inspect.")}
            )
        }
    )
    rendered = {
        "filesystem_inspect": {
            "mode": ToolParamDescription(description="Traversal mode.")
        },
        "repo_scan": {
            "branch": ToolParamDescription(description="Branch to examine.")
        },
    }

    merged = _merge_tool_param_descriptions(parent, rendered)

    assert merged["filesystem_inspect"] == {
        "path": ToolParamDescription(description="Absolute path to inspect."),
        "mode": ToolParamDescription(description="Traversal mode."),
    }
    assert merged["repo_scan"] == {
        "branch": ToolParamDescription(description="Branch to examine.")
    }
