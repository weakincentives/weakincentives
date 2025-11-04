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
from typing import TYPE_CHECKING, Annotated

import pytest

from weakincentives.prompt.errors import PromptValidationError
from weakincentives.prompt.tool import Tool, ToolContext, ToolResult


@dataclass
class ExampleParams:
    message: str


@dataclass
class ExampleResult:
    message: str


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
        Tool(name="lookup_entity", description="Fetch info", handler=_example_handler)  # type: ignore[call-arg]

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "type_arguments"


def test_tool_class_getitem_requires_two_type_arguments() -> None:
    with pytest.raises(TypeError):
        Tool[ExampleParams]  # type: ignore[index]


def test_tool_class_getitem_requires_type_objects() -> None:
    with pytest.raises(TypeError):
        Tool[ExampleParams, "not-a-type"]  # type: ignore[index]


def test_tool_class_getitem_rejects_extra_type_arguments() -> None:
    with pytest.raises(TypeError):
        Tool[ExampleParams, ExampleResult, ExampleResult]  # type: ignore[index]


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


def test_tool_rejects_non_dataclass_params_generic() -> None:
    def handler(_: str, *, context: ToolContext) -> ToolResult[ExampleResult]:
        del context
        return ToolResult(message="msg", value=ExampleResult(message="msg"))

    with pytest.raises(PromptValidationError) as error_info:
        Tool[str, ExampleResult](  # type: ignore[type-var]
            name="lookup_entity",
            description="Fetch info",
            handler=handler,
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.dataclass_type is str
    assert error.placeholder == "ParamsT"


def test_tool_rejects_non_dataclass_result_generic() -> None:
    def handler(params: ExampleParams, *, context: ToolContext) -> ToolResult[str]:
        del context
        return ToolResult(message=params.message, value=params.message)

    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, str](  # type: ignore[type-var]
            name="lookup_entity",
            description="Fetch info",
            handler=handler,
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.dataclass_type is str
    assert error.placeholder == "ResultT"


def test_tool_rejects_handler_when_not_callable() -> None:
    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="lookup_entity",
            description="Fetch info",
            handler="not callable",  # type: ignore[arg-type]
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "handler"


def test_tool_rejects_handler_missing_param_annotation() -> None:
    def handler(
        params: ExampleParams,
        *,
        context: ToolContext,
    ) -> ToolResult[ExampleResult]:
        del context
        return ToolResult(
            message=params.message, value=ExampleResult(message=params.message)
        )

    del handler.__annotations__["params"]

    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="lookup_entity",
            description="Fetch info",
            handler=handler,  # type: ignore[arg-type]
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "handler"


def test_tool_rejects_handler_with_wrong_param_annotation() -> None:
    def handler(params: str, *, context: ToolContext) -> ToolResult[ExampleResult]:
        del context
        return ToolResult(message=params, value=ExampleResult(message=params))

    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="lookup_entity",
            description="Fetch info",
            handler=handler,  # type: ignore[arg-type]
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "handler"


def test_tool_rejects_handler_with_multiple_params() -> None:
    def handler(
        first: ExampleParams,
        second: ExampleParams,
        *,
        context: ToolContext,
    ) -> ToolResult[ExampleResult]:
        del context
        combined = first.message + second.message
        return ToolResult(message=combined, value=ExampleResult(message=combined))

    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="lookup_entity",
            description="Fetch info",
            handler=handler,  # type: ignore[arg-type]
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "handler"


def test_tool_rejects_handler_missing_context_parameter() -> None:
    def handler(params: ExampleParams) -> ToolResult[ExampleResult]:
        return ToolResult(
            message=params.message, value=ExampleResult(message=params.message)
        )

    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="lookup_entity",
            description="Fetch info",
            handler=handler,  # type: ignore[arg-type]
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "handler"


def test_tool_rejects_handler_with_positional_context_parameter() -> None:
    def handler(
        params: ExampleParams, context: ToolContext
    ) -> ToolResult[ExampleResult]:
        del context
        return ToolResult(
            message=params.message, value=ExampleResult(message=params.message)
        )

    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="lookup_entity",
            description="Fetch info",
            handler=handler,  # type: ignore[arg-type]
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "handler"


def test_tool_rejects_handler_with_keyword_only_param() -> None:
    def handler(
        *, params: ExampleParams, context: ToolContext
    ) -> ToolResult[ExampleResult]:
        del context
        return ToolResult(
            message=params.message, value=ExampleResult(message=params.message)
        )

    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="lookup_entity",
            description="Fetch info",
            handler=handler,  # type: ignore[arg-type]
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "handler"


def test_tool_rejects_handler_with_context_default() -> None:
    def handler(
        params: ExampleParams,
        *,
        context: ToolContext = None,  # type: ignore[assignment]
    ) -> ToolResult[ExampleResult]:
        del context
        return ToolResult(
            message=params.message, value=ExampleResult(message=params.message)
        )

    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="lookup_entity",
            description="Fetch info",
            handler=handler,  # type: ignore[arg-type]
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "handler"


def test_tool_rejects_handler_with_wrong_context_name() -> None:
    def handler(
        params: ExampleParams, *, ctx: ToolContext
    ) -> ToolResult[ExampleResult]:
        del ctx
        return ToolResult(
            message=params.message, value=ExampleResult(message=params.message)
        )

    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="lookup_entity",
            description="Fetch info",
            handler=handler,  # type: ignore[arg-type]
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "handler"


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


def test_tool_rejects_handler_missing_return_annotation() -> None:
    def handler(params: ExampleParams, *, context: ToolContext):  # type: ignore[no-untyped-def]  # noqa: ANN202
        del context
        return ToolResult(
            message=params.message, value=ExampleResult(message=params.message)
        )

    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="lookup_entity",
            description="Fetch info",
            handler=handler,  # type: ignore[arg-type]
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "return"


def test_tool_rejects_handler_missing_context_annotation() -> None:
    def handler(params: ExampleParams, *, context) -> ToolResult[ExampleResult]:  # type: ignore[no-untyped-def]  # noqa: ANN001
        del context
        return ToolResult(
            message=params.message, value=ExampleResult(message=params.message)
        )

    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="lookup_entity",
            description="Fetch info",
            handler=handler,  # type: ignore[arg-type]
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "handler"


def test_tool_rejects_handler_with_wrong_context_annotation() -> None:
    def handler(params: ExampleParams, *, context: object) -> ToolResult[ExampleResult]:
        del context
        return ToolResult(
            message=params.message, value=ExampleResult(message=params.message)
        )

    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="lookup_entity",
            description="Fetch info",
            handler=handler,  # type: ignore[arg-type]
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "handler"


def test_tool_handler_type_hint_fallback_handles_name_errors() -> None:
    if TYPE_CHECKING:

        class MissingType: ...

    def handler(
        params: MissingType, *, context: ToolContext
    ) -> ToolResult[ExampleResult]:  # type: ignore[name-defined]
        del context
        return ToolResult(message="oops", value=ExampleResult(message="oops"))

    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="lookup_entity",
            description="Fetch info",
            handler=handler,  # type: ignore[arg-type]
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "handler"


def test_tool_rejects_handler_with_wrong_return_annotation() -> None:
    def handler(params: ExampleParams, *, context: ToolContext) -> ToolResult[str]:
        del context
        return ToolResult(message=params.message, value=params.message)

    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="lookup_entity",
            description="Fetch info",
            handler=handler,  # type: ignore[arg-type]
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "return"
