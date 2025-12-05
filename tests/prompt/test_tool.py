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

import types
from dataclasses import dataclass
from typing import Literal, cast

import pytest

import weakincentives.prompt.tool as tool_module
from weakincentives.prompt import (
    PromptValidationError,
    SupportsDataclass,
    SupportsDataclassOrNone,
    Tool,
    ToolContext,
    ToolExample,
    ToolResult,
)
from weakincentives.prompt.markdown import _render_example_value


@dataclass
class ExampleParams:
    query: str


@dataclass
class ExampleResult:
    value: str


@dataclass
class OtherParams:
    value: int


def test_tool_examples_are_preserved() -> None:
    example = ToolExample[ExampleParams, ExampleResult](
        description="simple lookup",
        input=ExampleParams(query="widgets"),
        output=ExampleResult(value="result"),
    )

    tool = Tool[ExampleParams, ExampleResult](
        name="lookup",
        description="Lookup information.",
        handler=None,
        examples=(example,),
    )

    assert tool.examples == (example,)


def test_tool_example_requires_ascii_description() -> None:
    example = ToolExample[ExampleParams, ExampleResult](
        description="emoji 😊",
        input=ExampleParams(query="widgets"),
        output=ExampleResult(value="result"),
    )

    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, ExampleResult](
            name="lookup",
            description="Lookup information.",
            handler=None,
            examples=(example,),
        )


def test_tool_example_input_must_match_params_type() -> None:
    example = ToolExample[ExampleParams, ExampleResult](
        description="mismatch",
        input=OtherParams(value=1),  # type: ignore[arg-type]
        output=ExampleResult(value="result"),
    )

    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, ExampleResult](
            name="lookup",
            description="Lookup information.",
            handler=None,
            examples=(example,),
        )


def test_tool_example_output_must_match_sequence_when_result_is_array() -> None:
    valid_example = ToolExample[ExampleParams, list[ExampleResult]](
        description="sequence output",
        input=ExampleParams(query="widgets"),
        output=[ExampleResult(value="first"), ExampleResult(value="second")],
    )

    tool = Tool[ExampleParams, list[ExampleResult]](
        name="batch_lookup",
        description="Lookup multiple results.",
        handler=None,
        examples=(valid_example,),
    )

    assert tool.examples == (valid_example,)

    invalid_example = ToolExample[ExampleParams, list[ExampleResult]](
        description="not a sequence",
        input=ExampleParams(query="widgets"),
        output=ExampleResult(value="lonely"),  # type: ignore[arg-type]
    )

    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, list[ExampleResult]](
            name="batch_lookup",
            description="Lookup multiple results.",
            handler=None,
            examples=(invalid_example,),
        )


def test_tool_example_rejects_blank_description() -> None:
    example = ToolExample[ExampleParams, ExampleResult](
        description="   ",
        input=ExampleParams(query="widgets"),
        output=ExampleResult(value="result"),
    )

    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, ExampleResult](
            name="lookup",
            description="Lookup information.",
            handler=None,
            examples=(example,),
        )


def test_tool_example_input_requires_dataclass_instance() -> None:
    example = ToolExample[ExampleParams, ExampleResult](
        description="non dataclass",
        input={"query": "widgets"},  # type: ignore[arg-type]
        output=ExampleResult(value="result"),
    )

    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, ExampleResult](
            name="lookup",
            description="Lookup information.",
            handler=None,
            examples=(example,),
        )


def test_tool_example_output_validates_sequence_items() -> None:
    example = ToolExample[ExampleParams, list[ExampleResult]](
        description="wrong items",
        input=ExampleParams(query="widgets"),
        output=[ExampleParams(query="widgets")],  # type: ignore[list-item]
    )

    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, list[ExampleResult]](
            name="batch_lookup",
            description="Lookup multiple results.",
            handler=None,
            examples=(example,),
        )


def test_tool_example_output_requires_dataclass_instance_for_object_results() -> None:
    example = ToolExample[ExampleParams, ExampleResult](
        description="wrong output type",
        input=ExampleParams(query="widgets"),
        output={"value": "result"},  # type: ignore[arg-type]
    )

    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, ExampleResult](
            name="lookup",
            description="Lookup information.",
            handler=None,
            examples=(example,),
        )


def test_tool_accepts_none_params_type() -> None:
    def handler(
        params: Literal[None], *, context: ToolContext
    ) -> ToolResult[ExampleResult]:
        return ToolResult(message="ok", value=ExampleResult(value="result"))

    tool = Tool[None, ExampleResult](
        name="noop",
        description="No parameters required.",
        handler=handler,
    )

    assert tool.params_type is type(None)


def test_tool_accepts_none_result_type() -> None:
    def handler(params: ExampleParams, *, context: ToolContext) -> ToolResult[None]:
        return ToolResult(message="done", value=None)

    tool = Tool[ExampleParams, None](
        name="no_result",
        description="Does not return a payload.",
        handler=handler,
    )

    assert tool.result_type is type(None)


def test_tool_example_input_requires_none_when_params_none() -> None:
    example = ToolExample[None, ExampleResult](
        description="invalid input",
        input=cast(None, ExampleParams(query="oops")),
        output=ExampleResult(value="result"),
    )

    with pytest.raises(PromptValidationError):
        Tool[None, ExampleResult](
            name="noop",
            description="No parameters required.",
            handler=None,
            examples=(example,),
        )


def test_tool_example_output_requires_none_when_result_none() -> None:
    example = ToolExample[ExampleParams, None](
        description="invalid output",
        input=ExampleParams(query="widgets"),
        output=cast(None, ExampleResult(value="result")),
    )

    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, None](
            name="noop",
            description="No result returned.",
            handler=None,
            examples=(example,),
        )


def test_coerce_none_type_accepts_union_alias() -> None:
    coerced = tool_module._coerce_none_type(SupportsDataclassOrNone)

    assert coerced is SupportsDataclass


def test_coerce_none_type_returns_none_when_union_is_all_none() -> None:
    coerced = tool_module._coerce_none_type(type(None) | None)

    assert coerced is type(None)


def test_coerce_none_type_handles_union_type_with_only_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_module, "get_origin", lambda _: types.UnionType)
    monkeypatch.setattr(tool_module, "get_args", lambda _: (type(None), type(None)))

    coerced = tool_module._coerce_none_type(object())

    assert coerced is type(None)


def test_tool_normalizes_none_result_annotation() -> None:
    result_type, container = Tool._normalize_result_annotation(
        None,
        ExampleParams,
    )

    assert result_type is type(None)
    assert container == "object"


def test_render_example_value_serializes_none() -> None:
    assert _render_example_value(None) == "null"


def test_tool_examples_accept_none_params_and_result() -> None:
    example = ToolExample[None, None](
        description="noop",
        input=None,
        output=None,
    )

    tool = Tool[None, None](
        name="noop",
        description="Does nothing.",
        handler=None,
        examples=(example,),
    )

    assert tool.examples == (example,)


def test_tool_examples_must_be_tool_example_instances() -> None:
    with pytest.raises(PromptValidationError):
        Tool[ExampleParams, ExampleResult](
            name="lookup",
            description="Lookup information.",
            handler=None,
            examples=("not-an-example",),  # type: ignore[arg-type]
        )
