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

"""Validation tests for task example steps that reference None-typed tools.

These exercise the ``expected_type is type(None)`` branches in
``prompt/_validators.py``: tools that take no params (``Tool[None, ...]``) or
return no value (``Tool[..., None]``), where a step's example input/output must
itself be ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    PromptValidationError,
    TaskExample,
    TaskExamplesSection,
    TaskStep,
    Tool,
    ToolContext,
    ToolExample,
    ToolResult,
)


@dataclass(slots=True, frozen=True)
class _ReadParams:
    path: str = field(metadata={"description": "File path to read"})


@dataclass(slots=True, frozen=True)
class _ReadResult:
    content: str


def _noparams_handler(params: None, *, context: ToolContext) -> ToolResult[_ReadResult]:
    del params, context
    return ToolResult.ok(_ReadResult(content="..."), message="Done")


def _noresult_handler(params: _ReadParams, *, context: ToolContext) -> ToolResult[None]:
    del params, context
    return ToolResult[None].ok(None, message="Done")


class TestNoneTypedStepValidation:
    """Validate task example steps for tools with None-typed params or results."""

    @staticmethod
    def _noparams_tool() -> Tool[None, _ReadResult]:
        return Tool[None, _ReadResult].create(
            name="noparams_tool",
            description="Tool that takes no params.",
            handler=_noparams_handler,
        )

    @staticmethod
    def _noresult_tool() -> Tool[_ReadParams, None]:
        return Tool[_ReadParams, None].create(
            name="noresult_tool",
            description="Tool that returns no value.",
            handler=_noresult_handler,
        )

    @staticmethod
    def _build(tool: Tool[Any, Any], example: ToolExample[Any, Any]) -> None:
        tools_section = MarkdownSection(
            key="tools", title="Tools", template="Tools:", tools=[tool]
        )
        task = TaskExample(
            key="step",
            objective="Exercise None-typed step validation paths",
            outcome="Done",
            steps=[TaskStep(tool_name=tool.name, example=example)],
        )
        template = PromptTemplate.create(
            ns="test",
            key="none-typed",
            sections=[tools_section, TaskExamplesSection(examples=[task])],
        )
        Prompt(template).sections  # noqa: B018

    def test_none_params_accepts_none_input(self) -> None:
        """A no-params tool step with ``None`` input passes validation."""
        self._build(
            self._noparams_tool(),
            ToolExample(
                description="Step", input=None, output=_ReadResult(content="x")
            ),
        )

    def test_none_params_rejects_non_none_input(self) -> None:
        """A no-params tool step with a non-``None`` input is rejected."""
        with pytest.raises(PromptValidationError) as exc:
            self._build(
                self._noparams_tool(),
                ToolExample(
                    description="Step",
                    input=_ReadResult(content="unexpected"),
                    output=_ReadResult(content="x"),
                ),
            )
        assert "input type mismatch" in str(exc.value)
        assert "Expected: None" in str(exc.value)

    def test_none_result_accepts_none_output(self) -> None:
        """A None-result tool step with ``None`` output passes validation."""
        self._build(
            self._noresult_tool(),
            ToolExample(
                description="Step", input=_ReadParams(path="a.py"), output=None
            ),
        )

    def test_none_result_rejects_non_none_output(self) -> None:
        """A None-result tool step with a non-``None`` output is rejected."""
        with pytest.raises(PromptValidationError) as exc:
            self._build(
                self._noresult_tool(),
                ToolExample(
                    description="Step",
                    input=_ReadParams(path="a.py"),
                    output=_ReadResult(content="unexpected"),
                ),
            )
        assert "output type mismatch" in str(exc.value)
        assert "Expected: None" in str(exc.value)
