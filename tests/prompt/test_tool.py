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
import typing
from dataclasses import dataclass
from typing import Annotated, Literal, cast

import pytest

import weakincentives.prompt.tool as tool_module
from weakincentives.prompt import (
    PromptValidationError,
    Tool,
    ToolContext,
    ToolExample,
    ToolResult,
)
from weakincentives.prompt._render_tool_examples import _render_example_value
from weakincentives.types import SupportsDataclass, SupportsDataclassOrNone


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


def test_tool_wrap_builds_tool_from_handler() -> None:
    def lookup(
        params: ExampleParams, *, context: ToolContext
    ) -> ToolResult[ExampleResult]:
        """Lookup information."""

        return ToolResult(message="ok", value=ExampleResult(value=params.query))

    tool = Tool.wrap(lookup)

    assert tool.name == "lookup"
    assert tool.description == "Lookup information."
    assert tool.handler is lookup
    assert tool.params_type is ExampleParams
    assert tool.result_type is ExampleResult


def test_tool_wrap_requires_docstring() -> None:
    def docless(
        params: ExampleParams, *, context: ToolContext
    ) -> ToolResult[ExampleResult]:
        return ToolResult(message="ok", value=ExampleResult(value=params.query))

    with pytest.raises(PromptValidationError):
        Tool.wrap(docless)


def test_tool_wrap_enforces_handler_constraints() -> None:
    def invalid_handler(
        params: ExampleParams, context: ToolContext
    ) -> ToolResult[ExampleResult]:
        return ToolResult(message="ok", value=ExampleResult(value=params.query))

    with pytest.raises(PromptValidationError):
        Tool.wrap(invalid_handler)


def test_tool_wrap_requires_parameter_annotation() -> None:
    def missing_annotation(
        params: object, *, context: ToolContext
    ) -> ToolResult[ExampleResult]:
        """Missing param annotation."""

        return ToolResult(message="ok", value=ExampleResult(value="result"))

    missing_annotation.__annotations__.pop("params", None)

    with pytest.raises(PromptValidationError):
        Tool.wrap(missing_annotation)


def test_tool_wrap_supports_annotated_params() -> None:
    def annotated_params(
        params: Annotated[ExampleParams, "meta"], *, context: ToolContext
    ) -> ToolResult[ExampleResult]:
        """Annotated params."""

        return ToolResult(message="ok", value=ExampleResult(value=params.query))

    tool = Tool.wrap(annotated_params)

    assert tool.params_type is ExampleParams


def test_tool_wrap_supports_literal_params() -> None:
    def literal_params(
        params: ExampleParams, *, context: ToolContext
    ) -> ToolResult[ExampleResult]:
        """Literal params."""

        return ToolResult(message="ok", value=ExampleResult(value=params.query))

    literal_params.__annotations__["params"] = Literal[ExampleParams]  # type: ignore[invalid-type-form]

    tool = Tool.wrap(literal_params)

    assert tool.params_type is ExampleParams


def test_tool_wrap_supports_annotated_return() -> None:
    def annotated_return(
        params: ExampleParams, *, context: ToolContext
    ) -> Annotated[ToolResult[ExampleResult], "meta"]:
        """Annotated return."""

        return ToolResult(message="ok", value=ExampleResult(value=params.query))

    tool = Tool.wrap(annotated_return)

    assert tool.result_type is ExampleResult


def test_tool_wrap_requires_return_annotation() -> None:
    def missing_return(
        params: ExampleParams, *, context: ToolContext
    ) -> ToolResult[ExampleResult]:
        """Missing return annotation."""

        return ToolResult(message="ok", value=ExampleResult(value=params.query))

    missing_return.__annotations__.pop("return", None)

    with pytest.raises(PromptValidationError):
        Tool.wrap(missing_return)


def test_tool_wrap_requires_toolresult_return() -> None:
    def wrong_return(params: ExampleParams, *, context: ToolContext) -> ExampleResult:
        """Wrong return annotation."""

        return ExampleResult(value=params.query)

    with pytest.raises(PromptValidationError):
        Tool.wrap(wrong_return)  # type: ignore[arg-type]


def test_tool_wrap_requires_result_argument() -> None:
    def unparameterized_result(
        params: ExampleParams, *, context: ToolContext
    ) -> ToolResult:
        """ToolResult without parameter."""

        return ToolResult(message="ok", value=ExampleResult(value=params.query))

    unparameterized_result.__annotations__["return"] = ToolResult

    with pytest.raises(PromptValidationError):
        Tool.wrap(unparameterized_result)


def test_tool_wrap_handles_empty_toolresult_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class ExampleParams:
        value: int

    def handler(
        params: ExampleParams, *, context: ToolContext
    ) -> ToolResult[int]:  # pragma: no cover - exercised via wrap
        """Docstring for Tool.wrap coverage."""

        return ToolResult(message="ok", value=params.value)

    handler.__annotations__["return"] = ToolResult[int]

    def fake_get_args(annotation: object) -> tuple[object, ...]:
        return () if annotation == ToolResult[int] else typing.get_args(annotation)

    monkeypatch.setattr(tool_module, "get_args", fake_get_args)

    with pytest.raises(
        PromptValidationError,
        match=r"Tool handler return annotation must be ToolResult\[ResultT\].",
    ):
        Tool.wrap(handler)
