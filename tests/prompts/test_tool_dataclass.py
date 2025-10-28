from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, cast

import pytest

from weakincentives.prompts.errors import PromptValidationError
from weakincentives.prompts.tool import Tool, ToolResult


@dataclass
class ExampleParams:
    message: str


@dataclass
class ExampleResult:
    message: str


def _example_handler(params: ExampleParams) -> ToolResult[ExampleResult]:
    return ToolResult(
        message=params.message, payload=ExampleResult(message=params.message)
    )


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
    def handler(_: str) -> ToolResult[ExampleResult]:
        return ToolResult(message="msg", payload=ExampleResult(message="msg"))

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
    def handler(params: ExampleParams) -> ToolResult[str]:
        return ToolResult(message=params.message, payload=params.message)

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
    def handler(params):  # type: ignore[no-untyped-def]
        return ToolResult(
            message=params.message, payload=ExampleResult(message=params.message)
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


def test_tool_rejects_handler_with_wrong_param_annotation() -> None:
    def handler(params: str) -> ToolResult[ExampleResult]:
        return ToolResult(message=params, payload=ExampleResult(message=params))

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
        first: ExampleParams, second: ExampleParams
    ) -> ToolResult[ExampleResult]:  # pragma: no cover - signature invalid
        combined = first.message + second.message
        return ToolResult(message=combined, payload=ExampleResult(message=combined))

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
    def handler(*, params: ExampleParams) -> ToolResult[ExampleResult]:
        return ToolResult(
            message=params.message, payload=ExampleResult(message=params.message)
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
    ) -> Annotated[ToolResult[ExampleResult], "meta"]:
        return ToolResult(
            message=params.message, payload=ExampleResult(message=params.message)
        )

    tool = Tool[ExampleParams, ExampleResult](
        name="lookup_entity",
        description="Fetch structured entity info.",
        handler=handler,
    )

    assert tool.handler is handler


def test_tool_rejects_handler_missing_return_annotation() -> None:
    def handler(params: ExampleParams):  # type: ignore[no-untyped-def]
        return ToolResult(
            message=params.message, payload=ExampleResult(message=params.message)
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


def test_tool_rejects_handler_with_wrong_return_annotation() -> None:
    def handler(params: ExampleParams) -> ToolResult[str]:
        return ToolResult(message=params.message, payload=params.message)

    with pytest.raises(PromptValidationError) as error_info:
        Tool[ExampleParams, ExampleResult](
            name="lookup_entity",
            description="Fetch info",
            handler=handler,  # type: ignore[arg-type]
        )

    error = error_info.value
    assert isinstance(error, PromptValidationError)
    assert error.placeholder == "return"
