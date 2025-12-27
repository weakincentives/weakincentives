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

from dataclasses import dataclass
from typing import Annotated, Any, cast

import pytest

from weakincentives.prompt.errors import PromptValidationError
from weakincentives.prompt.tool import Tool, ToolContext, ToolHandler, ToolResult


@dataclass
class ExampleParams:
    message: str


@dataclass
class ExampleResult:
    message: str


@dataclass
class OtherResult:
    payload: str


_UNSAFE_TOOL = cast(Any, Tool)


def _example_handler(
    params: ExampleParams, *, context: ToolContext
) -> ToolResult[ExampleResult]:
    del context
    return ToolResult(
        message=params.message, value=ExampleResult(message=params.message)
    )


def test_tool_infers_param_and_result_types() -> None:
    tool = Tool[ExampleParams, ExampleResult](
        name="echo",
        description="Echo the provided message.",
        handler=_example_handler,
    )

    assert tool.params_type is ExampleParams
    assert tool.result_type is ExampleResult


def test_tool_initialises_with_clean_name_and_description() -> None:
    tool = Tool[ExampleParams, ExampleResult](
        name="lookup_entity",
        description="  Fetch structured entity info.  ",
        handler=_example_handler,
    )

    assert tool.name == "lookup_entity"
    assert tool.description == "Fetch structured entity info."
    assert tool.params_type is ExampleParams
    assert tool.result_type is ExampleResult
    assert tool.handler is _example_handler


def test_tool_requires_generic_type_arguments() -> None:
    with pytest.raises(PromptValidationError) as error_info:
        Tool(name="lookup_entity", description="Fetch info", handler=_example_handler)

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "type_arguments"


def test_tool_class_getitem_requires_two_type_arguments() -> None:
    with pytest.raises(TypeError):
        _UNSAFE_TOOL[ExampleParams]


def test_tool_class_getitem_requires_type_objects() -> None:
    specialized = Tool[ExampleParams, "not-a-type"]

    with pytest.raises(PromptValidationError):
        specialized(
            name="lookup_entity",
            description="Fetch info",
            handler=None,
        )


def test_tool_class_getitem_requires_params_type_argument() -> None:
    with pytest.raises(TypeError):
        _UNSAFE_TOOL["not-a-type", ExampleResult]


def test_tool_class_getitem_rejects_extra_type_arguments() -> None:
    with pytest.raises(TypeError):
        _UNSAFE_TOOL[ExampleParams, ExampleResult, ExampleResult]


def test_tool_subclass_resolves_generic_arguments() -> None:
    class _SubclassTool(Tool[ExampleParams, ExampleResult]):
        pass

    tool = _SubclassTool(
        name="lookup_entity",
        description="Fetch info",
        handler=_example_handler,
    )

    assert tool.params_type is ExampleParams
    assert tool.result_type is ExampleResult


def test_tool_rejects_name_with_surrounding_whitespace() -> None:
    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name=" lookup ",
            description="Fetch info",
            handler=_example_handler,
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "lookup"


def test_tool_rejects_name_with_invalid_characters() -> None:
    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="Lookup",
            description="Fetch info",
            handler=_example_handler,
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "Lookup"


def test_tool_rejects_empty_name() -> None:
    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="",
            description="Fetch info",
            handler=_example_handler,
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == ""


@pytest.mark.parametrize(
    "bad_description",
    ["", "   ", "a" * 201, "déjà vu"],
)
def test_tool_rejects_invalid_descriptions(bad_description: str) -> None:
    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="lookup_entity",
            description=bad_description,
            handler=_example_handler,
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "description"


def test_tool_accepts_variadic_tuple_result_type() -> None:
    def handler(
        params: ExampleParams, *, context: ToolContext
    ) -> ToolResult[tuple[ExampleResult, ...]]:
        del context
        value = (ExampleResult(message=params.message),)
        result = ToolResult(message=params.message, value=value)
        return cast(ToolResult[tuple[ExampleResult, ...]], result)

    tool = Tool[ExampleParams, tuple[ExampleResult, ...]](
        name="collect_examples",
        description="Collect multiple examples.",
        handler=handler,
    )

    assert tool.result_container == "array"
    assert tool.result_type is ExampleResult


def test_tool_accepts_list_result_type() -> None:
    def handler(
        params: ExampleParams, *, context: ToolContext
    ) -> ToolResult[list[ExampleResult]]:
        del context
        return ToolResult(
            message=params.message,
            value=[ExampleResult(message=params.message)],
        )

    tool = Tool[ExampleParams, list[ExampleResult]](
        name="collect_list",
        description="Collect examples in a list.",
        handler=handler,
    )

    assert tool.result_container == "array"
    assert tool.result_type is ExampleResult


def test_tool_rejects_tuple_result_without_ellipsis() -> None:
    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, tuple[ExampleResult, ExampleResult]](
            name="bad_tuple",
            description="Invalid tuple result.",
            handler=cast(
                ToolHandler[ExampleParams, tuple[ExampleResult, ExampleResult]],
                _example_handler,
            ),
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "ResultT"


def test_tool_rejects_sequence_result_with_non_type_element() -> None:
    with pytest.raises(PromptValidationError) as error_info:
        _UNSAFE_TOOL[ExampleParams, tuple[None, ...]](
            name="bad_tuple_element",
            description="Invalid tuple element.",
            handler=None,
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "ResultT"


def test_tool_class_getitem_rejects_unsupported_result_annotation() -> None:
    with pytest.raises(PromptValidationError) as error_info:
        _UNSAFE_TOOL[ExampleParams, dict[str, ExampleResult]](
            name="bad_mapping",
            description="Unsupported mapping result.",
            handler=None,
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "ResultT"


def test_tool_normalize_result_annotation_rejects_unknown_annotation() -> None:
    with pytest.raises(PromptValidationError) as error_info:
        Tool._normalize_result_annotation(
            ExampleResult | None,
            ExampleParams,
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "ResultT"


# Handler validation tests removed: pyright strict mode catches signature mismatches
# at development time. Runtime TypeErrors are caught at call site in tool_executor.py


def test_tool_accepts_annotated_param_and_return() -> None:
    def handler(
        params: Annotated[ExampleParams, "meta"],
        *,
        context: Annotated[ToolContext, "meta"],
    ) -> Annotated[ToolResult[ExampleResult], "meta"]:
        del context
        return ToolResult(
            message=params.message, value=ExampleResult(message=params.message)
        )

    tool = Tool[ExampleParams, ExampleResult](
        name="lookup_entity",
        description="Fetch structured entity info.",
        handler=handler,
    )

    assert tool.handler is handler
