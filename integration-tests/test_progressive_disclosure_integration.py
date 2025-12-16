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

"""Integration tests for progressive disclosure functionality.

These tests verify that models can use the open_sections tool to progressively
expand summarized sections and access tools registered on leaf sections.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    SectionVisibility,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.prompt.errors import SectionPath, VisibilityExpansionRequired
from weakincentives.prompt.visibility_overrides import (
    SetVisibilityOverride,
    VisibilityOverrides,
)
from weakincentives.runtime.session import Session

pytest.importorskip("openai")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ,
        reason="OPENAI_API_KEY not set; skipping OpenAI integration tests.",
    ),
    pytest.mark.timeout(60),
]

_MODEL_ENV_VAR = "OPENAI_TEST_MODEL"
_DEFAULT_MODEL = "gpt-5.1"
_PROMPT_NS = "integration/progressive-disclosure"

# Maximum number of expansion requests expected for a two-level hierarchy
# (parent + child + possible retry)
_MAX_EXPECTED_EXPANSIONS = 3


@pytest.fixture(scope="module")
def openai_model() -> str:
    """Return the model name used for integration tests."""
    return os.environ.get(_MODEL_ENV_VAR, _DEFAULT_MODEL)


@dataclass(slots=True)
class InstructionParams:
    """Top-level instruction parameters."""

    task: str


@dataclass(slots=True)
class GuidelinesParams:
    """Parameters for the guidelines section (level 1)."""

    domain: str = "testing"


@dataclass(slots=True)
class ToolsParams:
    """Parameters for the tools section (level 2 - leaf)."""

    tool_name: str = "verify_result"


@dataclass(slots=True)
class VerifyRequest:
    """Input for the verification tool on the leaf section."""

    value: str


@dataclass(slots=True)
class VerifyResult:
    """Result from the verification tool."""

    verified: bool
    message: str


def _build_verify_tool() -> Tool[VerifyRequest, VerifyResult]:
    """Build a simple verification tool to attach to the leaf section."""

    def verify_handler(
        params: VerifyRequest, *, context: ToolContext
    ) -> ToolResult[VerifyResult]:
        del context
        result = VerifyResult(
            verified=True,
            message=f"Successfully verified: {params.value}",
        )
        return ToolResult(
            message=f"Verified value: {params.value}",
            value=result,
        )

    return Tool[VerifyRequest, VerifyResult](
        name="verify_result",
        description="Verify a result value. Returns verification status and message.",
        handler=verify_handler,
    )


def _build_two_level_hierarchy_prompt(
    verify_tool: Tool[VerifyRequest, VerifyResult],
) -> PromptTemplate[object]:
    """Build a prompt with a two-level deep summarized section hierarchy.

    Structure:
    - Instructions (FULL)
      - Guidelines (SUMMARY -> needs expansion)
        - Tools Reference (SUMMARY -> needs expansion, contains verify_result tool)
    """
    # Level 2 (leaf): Tools reference with the verification tool
    tools_section = MarkdownSection[ToolsParams](
        title="Tools Reference",
        template=(
            "The ${tool_name} tool allows you to verify results.\n\n"
            "To complete the task, you MUST call verify_result with the "
            'value "integration-test-value".'
        ),
        key="tools-reference",
        summary="Tool documentation is available here. Expand to see tool usage.",
        visibility=SectionVisibility.SUMMARY,
        tools=(verify_tool,),
        default_params=ToolsParams(),
    )

    # Level 1: Guidelines section containing the tools reference
    guidelines_section = MarkdownSection[GuidelinesParams](
        title="Guidelines",
        template=(
            "Follow these ${domain} guidelines carefully.\n\n"
            "The tools reference below contains important documentation."
        ),
        key="guidelines",
        summary="Guidelines for completing the task. Expand for detailed instructions.",
        visibility=SectionVisibility.SUMMARY,
        children=[tools_section],
        default_params=GuidelinesParams(),
    )

    # Root: Instructions section (FULL visibility)
    instructions_section = MarkdownSection[InstructionParams](
        title="Instructions",
        template=(
            "Your task: ${task}\n\n"
            "IMPORTANT: You must expand summarized sections to find the tools you need. "
            "Look for sections marked as summarized and use open_sections to expand them. "
            "Once you find the verify_result tool, call it to complete the task."
        ),
        key="instructions",
        children=[guidelines_section],
    )

    return PromptTemplate(
        ns=_PROMPT_NS,
        key="two-level-hierarchy",
        name="progressive_disclosure_test",
        sections=[instructions_section],
    )


def test_progressive_disclosure_two_level_hierarchy(openai_model: str) -> None:
    """Test that a model can expand a two-level hierarchy and use a leaf tool.

    This test verifies:
    1. The model recognizes summarized sections and calls open_sections
    2. The prompt re-renders with expanded visibility
    3. After expanding the full hierarchy, the leaf tool becomes usable
    4. The model successfully calls the leaf section's tool
    """
    verify_tool = _build_verify_tool()
    prompt_template = _build_two_level_hierarchy_prompt(verify_tool)
    params = InstructionParams(task="Verify the value 'integration-test-value'")

    adapter = OpenAIAdapter(model=openai_model)

    prompt = Prompt(prompt_template).bind(params)
    session = Session()

    max_expansions = 5
    expansion_count = 0
    tool_was_called = False

    while expansion_count < max_expansions:
        try:
            response = adapter.evaluate(
                prompt,
                session=session,
            )
            # Evaluation completed without expansion request
            # Check if the model used the verify_result tool
            if response.text is not None:
                # The model completed - check for success indicators
                text_lower = response.text.lower()
                if "verified" in text_lower or "integration-test-value" in text_lower:
                    tool_was_called = True
            break

        except VisibilityExpansionRequired as e:
            # Model requested section expansion - update session state
            expansion_count += 1
            for path, visibility in e.requested_overrides.items():
                session.broadcast(
                    SetVisibilityOverride(path=path, visibility=visibility)
                )
            # Continue the loop with updated visibility in session state

    # Assertions
    assert expansion_count > 0, (
        "Expected at least one section expansion request. "
        "The model should have called open_sections for summarized sections."
    )
    assert expansion_count <= _MAX_EXPECTED_EXPANSIONS, (
        f"Expected at most {_MAX_EXPECTED_EXPANSIONS} expansion requests "
        f"(parent + child + possible retry), got {expansion_count}."
    )
    assert tool_was_called, (
        "Expected the model to successfully call the verify_result tool "
        "after expanding the summarized sections."
    )

    # Verify that the expected sections were expanded in session state
    overrides = session[VisibilityOverrides].latest()
    assert overrides is not None, "Expected VisibilityOverrides in session state"
    expected_paths: list[SectionPath] = [
        ("instructions", "guidelines"),
        ("instructions", "guidelines", "tools-reference"),
    ]
    for path in expected_paths:
        assert overrides.get(path) == SectionVisibility.FULL, (
            f"Expected section path {path} to be expanded to FULL"
        )


def test_progressive_disclosure_direct_leaf_expansion(openai_model: str) -> None:
    """Test that models can request expansion of nested sections directly.

    Some models may request expansion of both parent and child in a single call.
    This test verifies that the framework handles this correctly.
    """
    verify_tool = _build_verify_tool()
    prompt_template = _build_two_level_hierarchy_prompt(verify_tool)
    params = InstructionParams(
        task="Expand all summarized sections and call verify_result with 'direct-test'"
    )

    adapter = OpenAIAdapter(model=openai_model)

    prompt = Prompt(prompt_template).bind(params)
    session = Session()

    max_iterations = 5
    iteration = 0
    final_response = None

    while iteration < max_iterations:
        iteration += 1
        try:
            response = adapter.evaluate(
                prompt,
                session=session,
            )
            final_response = response
            break

        except VisibilityExpansionRequired as e:
            for path, visibility in e.requested_overrides.items():
                session.broadcast(
                    SetVisibilityOverride(path=path, visibility=visibility)
                )

    assert final_response is not None, (
        f"Expected a final response after {max_iterations} iterations."
    )
    assert final_response.text is not None, "Expected text response from model."

    # The leaf section should be expanded to expose the tool
    overrides = session[VisibilityOverrides].latest()
    leaf_path: SectionPath = ("instructions", "guidelines", "tools-reference")
    assert (
        overrides is not None and overrides.get(leaf_path) == SectionVisibility.FULL
    ), f"Expected leaf section {leaf_path} to be expanded."
