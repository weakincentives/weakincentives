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

"""Shared ACK scenario dataclasses and prompt/tool builders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from weakincentives.prompt import (
    MarkdownSection,
    PromptTemplate,
    SectionVisibility,
    Tool,
    ToolContext,
    ToolResult,
    WorkspaceSection,
)
from weakincentives.skills import SkillMount

_ACK_NS_PREFIX = "integration/ack"


@dataclass(slots=True)
class GreetingParams:
    """Prompt params for a short greeting scenario."""

    audience: str


@dataclass(slots=True)
class TransformRequest:
    """Input for the uppercase helper tool."""

    text: str


@dataclass(slots=True, frozen=True)
class TransformResult:
    """Output payload from the uppercase helper tool."""

    text: str


@dataclass(slots=True)
class ReviewParams:
    """Prompt params for structured output scenarios."""

    text: str


@dataclass(slots=True, frozen=True)
class ReviewAnalysis:
    """Structured output schema for review analysis."""

    summary: str
    sentiment: str


@dataclass(slots=True)
class InstructionParams:
    """Top-level params for progressive disclosure tests."""

    task: str


@dataclass(slots=True)
class GuidelinesParams:
    """Params for level-1 summarized section."""

    domain: str = "testing"


@dataclass(slots=True)
class ToolsParams:
    """Params for level-2 summarized leaf section."""

    tool_name: str = "verify_result"


@dataclass(slots=True)
class VerifyRequest:
    """Input schema for the progressive disclosure verify tool."""

    value: str


@dataclass(slots=True, frozen=True)
class VerifyResult:
    """Output schema for the progressive disclosure verify tool."""

    verified: bool
    message: str


@dataclass(slots=True)
class TransactionPromptParams:
    """No-op params for transactional prompt templates."""


def build_greeting_prompt(ns: str) -> PromptTemplate[object]:
    """Build a deterministic greeting prompt."""
    section = MarkdownSection[GreetingParams](
        title="Greeting",
        template=(
            "You are a concise assistant. Provide a short friendly greeting for ${audience}. "
            "Reply in exactly one sentence."
        ),
        key="greeting",
    )
    return PromptTemplate(
        ns=ns,
        key="ack-greeting",
        name="ack_greeting",
        sections=[section],
    )


def build_uppercase_tool(
    calls: list[str] | None = None,
) -> Tool[TransformRequest, TransformResult]:
    """Build a deterministic uppercase conversion tool."""

    def handler(
        params: TransformRequest,
        *,
        context: ToolContext,
    ) -> ToolResult[TransformResult]:
        del context
        if calls is not None:
            calls.append(params.text)
        result = TransformResult(text=params.text.upper())
        return ToolResult.ok(result, message=f"Uppercased: {result.text}")

    return Tool[TransformRequest, TransformResult](
        name="uppercase_text",
        description="Return the provided text in uppercase characters.",
        handler=handler,
    )


def build_tool_prompt(
    ns: str,
    tool: Tool[TransformRequest, TransformResult],
) -> PromptTemplate[object]:
    """Build a prompt that instructs exactly one uppercase tool call."""
    section = MarkdownSection[TransformRequest](
        title="Tool Task",
        template=(
            "Call the `uppercase_text` tool exactly once with this JSON: "
            '{"text": "${text}"}. Then respond with the transformed text.'
        ),
        tools=(tool,),
        key="instruction",
    )
    return PromptTemplate(
        ns=ns,
        key="ack-uppercase",
        name="ack_uppercase_workflow",
        sections=[section],
    )


def build_structured_prompt(ns: str) -> PromptTemplate[ReviewAnalysis]:
    """Build a structured output prompt for ``ReviewAnalysis``."""
    section = MarkdownSection[ReviewParams](
        title="Analysis",
        template=(
            "Review the passage below and return concise summary and sentiment fields.\n"
            "Passage:\n${text}"
        ),
        key="analysis",
    )
    return PromptTemplate[ReviewAnalysis](
        ns=ns,
        key="ack-structured",
        name="ack_structured_review",
        sections=[section],
    )


def build_verify_tool() -> Tool[VerifyRequest, VerifyResult]:
    """Build the verify tool used by progressive disclosure scenarios."""

    def handler(
        params: VerifyRequest,
        *,
        context: ToolContext,
    ) -> ToolResult[VerifyResult]:
        del context
        result = VerifyResult(
            verified=True,
            message=f"Successfully verified: {params.value}",
        )
        return ToolResult.ok(result, message=result.message)

    return Tool[VerifyRequest, VerifyResult](
        name="verify_result",
        description="Verify the provided value and return verification details.",
        handler=handler,
    )


def build_progressive_disclosure_prompt(
    ns: str,
    verify_tool: Tool[VerifyRequest, VerifyResult],
) -> PromptTemplate[object]:
    """Build a two-level summarized hierarchy with a leaf tool."""
    tools_section = MarkdownSection[ToolsParams](
        title="Tools Reference",
        template=(
            "Use ${tool_name} to verify outcomes. To finish this task, call verify_result "
            'with value "integration-test-value".'
        ),
        key="tools-reference",
        summary="Tool documentation is available here. Expand this section.",
        visibility=SectionVisibility.SUMMARY,
        tools=(verify_tool,),
        default_params=ToolsParams(),
    )

    guidelines_section = MarkdownSection[GuidelinesParams](
        title="Guidelines",
        template=(
            "Follow these ${domain} guidelines. The tools reference below contains "
            "required details for task completion."
        ),
        key="guidelines",
        summary="Guidelines are summarized. Expand to continue.",
        visibility=SectionVisibility.SUMMARY,
        children=[tools_section],
        default_params=GuidelinesParams(),
    )

    instructions_section = MarkdownSection[InstructionParams](
        title="Instructions",
        template=(
            "Task: ${task}\n\n"
            "Expand summarized sections with open_sections before attempting tool calls."
        ),
        key="instructions",
        children=[guidelines_section],
    )

    return PromptTemplate(
        ns=ns,
        key="ack-progressive-disclosure",
        name="ack_progressive_disclosure",
        sections=[instructions_section],
    )


def build_native_tool_prompt(ns: str) -> PromptTemplate[object]:
    """Build a prompt that requires a provider-native file read/tool call."""
    section = MarkdownSection(
        title="Native Tool Task",
        template=(
            "Read the file README.md in the current working directory and summarize "
            "its contents in one sentence."
        ),
        key="task",
    )
    return PromptTemplate(
        ns=ns,
        key="ack-native-tool",
        name="ack_native_tool",
        sections=[section],
    )


def build_transactional_prompt(
    ns: str,
    write_succeed_tool: Tool[object, object],
    write_fail_tool: Tool[object, object],
    workspace_section: WorkspaceSection,
) -> PromptTemplate[object]:
    """Build a transactional prompt whose tools are executed via bridge wrappers."""
    section = MarkdownSection[TransactionPromptParams](
        title="Transactional Task",
        template=(
            "This prompt exists to provide a resource context for bridged transactional "
            "tools."
        ),
        tools=(write_succeed_tool, write_fail_tool),
        key="task",
    )
    return PromptTemplate(
        ns=ns,
        key="ack-transactional",
        name="ack_transactional_tools",
        sections=[workspace_section, section],
    )


@dataclass(slots=True)
class SkillQuestionParams:
    """Prompt params for skill installation scenario."""

    question: str


#: Secret answer embedded in the test skill definition.
SKILL_SECRET_ANSWER = "blue-phoenix-42"


def create_test_skill(base_dir: Path) -> Path:
    """Create a test skill directory with a secret answer.

    Returns the path to the skill directory.
    """
    skill_dir = base_dir / "ack-secret-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: ack-secret-skill\n"
        "description: ACK test skill containing a secret answer for integration testing.\n"
        "---\n"
        "\n"
        "# ACK Secret Skill\n"
        "\n"
        "You have specialized knowledge from this skill.\n"
        "\n"
        "## Secret Knowledge\n"
        "\n"
        "When asked for the secret codeword, the answer is: "
        f"**{SKILL_SECRET_ANSWER}**\n"
        "\n"
        "Always include this exact codeword in your response when asked.\n"
    )
    return skill_dir


def build_skill_prompt(
    ns: str,
    skill_dir: Path,
) -> PromptTemplate[object]:
    """Build a prompt with an attached skill that holds a secret answer."""
    section = MarkdownSection[SkillQuestionParams](
        title="Skill Task",
        template=(
            "Answer the following question using the knowledge from your installed "
            "skills. ${question}\n"
            "Include the exact codeword in your response."
        ),
        key="skill-task",
        skills=(SkillMount(source=skill_dir, name="ack-secret-skill"),),
    )
    return PromptTemplate(
        ns=ns,
        key="ack-skill-installation",
        name="ack_skill_installation",
        sections=[section],
    )


def make_adapter_ns(adapter_name: str) -> str:
    """Return a stable ACK namespace for the given adapter."""
    return f"{_ACK_NS_PREFIX}/{adapter_name}"


__all__ = [
    "SKILL_SECRET_ANSWER",
    "GreetingParams",
    "GuidelinesParams",
    "InstructionParams",
    "ReviewAnalysis",
    "ReviewParams",
    "SkillQuestionParams",
    "ToolsParams",
    "TransactionPromptParams",
    "TransformRequest",
    "TransformResult",
    "VerifyRequest",
    "VerifyResult",
    "build_greeting_prompt",
    "build_native_tool_prompt",
    "build_progressive_disclosure_prompt",
    "build_skill_prompt",
    "build_structured_prompt",
    "build_tool_prompt",
    "build_transactional_prompt",
    "build_uppercase_tool",
    "build_verify_tool",
    "create_test_skill",
    "make_adapter_ns",
]
