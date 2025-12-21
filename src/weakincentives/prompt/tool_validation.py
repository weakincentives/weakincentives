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

# pyright: reportImportCycles=false

"""Validation functions for Tool handler signatures, names, and examples.

This module contains validation logic extracted from the Tool class to keep
the dataclass focused on type inference and tool definition.
"""

from __future__ import annotations

import inspect
import re
import types
from collections.abc import Callable, Sequence as SequenceABC
from dataclasses import is_dataclass
from typing import (
    Annotated,
    Any,
    Final,
    Literal,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from ..types.dataclass import SupportsDataclass, SupportsToolResult
from .errors import PromptValidationError

_NAME_MIN_LENGTH: Final = 1
_NAME_MAX_LENGTH: Final = 64
_DESCRIPTION_MIN_LENGTH: Final = 1
_DESCRIPTION_MAX_LENGTH: Final = 200
_EXPECTED_TYPE_ARGUMENTS: Final = 2
_HANDLER_PARAMETER_COUNT: Final = 2
_VARIADIC_TUPLE_LENGTH: Final = 2
_NONE_TYPE: Final = type(None)
_POSITIONAL_PARAMETER_KINDS: Final = {
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
}

_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"^[a-z0-9_-]{{{_NAME_MIN_LENGTH},{_NAME_MAX_LENGTH}}}$"
)

type ParamsType = type[SupportsDataclass] | type[None]
type ResultType = type[SupportsDataclass] | type[None]


# ---------------------------------------------------------------------------
# Type Coercion Helpers
# ---------------------------------------------------------------------------


def coerce_none_type(candidate: object) -> object:
    """Coerce None and Union[T, None] to type(None) or the non-None type."""
    if candidate is None:
        return _NONE_TYPE

    origin = get_origin(candidate)
    if origin is types.UnionType:
        args = get_args(candidate)
        if args and all(isinstance(arg, type) for arg in args):
            non_none = [arg for arg in args if arg is not _NONE_TYPE]
            if non_none:
                return non_none[0]
            return _NONE_TYPE

    return candidate


# ---------------------------------------------------------------------------
# Name and Description Validation
# ---------------------------------------------------------------------------


def validate_name(raw_name: str, params_type: ParamsType) -> str:
    """Validate and normalize a tool name.

    Args:
        raw_name: The raw name string to validate.
        params_type: The params type for error context.

    Returns:
        The validated and normalized name.

    Raises:
        PromptValidationError: If the name is invalid.
    """
    stripped_name = raw_name.strip()
    if raw_name != stripped_name:
        normalized_name = stripped_name
        raise PromptValidationError(
            "Tool name must not contain surrounding whitespace.",
            dataclass_type=params_type,
            placeholder=normalized_name,
        )

    name_clean = raw_name
    if not name_clean:
        raise PromptValidationError(
            "Tool name must match the OpenAI function name constraints (1-64 lowercase ASCII letters, digits, underscores, or hyphens).",
            dataclass_type=params_type,
            placeholder=stripped_name,
        )
    if len(name_clean) > _NAME_MAX_LENGTH or not _NAME_PATTERN.fullmatch(name_clean):
        raise PromptValidationError(
            f"Tool name must match the OpenAI function name constraints (pattern: {_NAME_PATTERN.pattern}).",
            dataclass_type=params_type,
            placeholder=name_clean,
        )

    return name_clean


def validate_description(raw_description: str, params_type: ParamsType) -> str:
    """Validate and normalize a tool description.

    Args:
        raw_description: The raw description string to validate.
        params_type: The params type for error context.

    Returns:
        The validated and normalized description.

    Raises:
        PromptValidationError: If the description is invalid.
    """
    description_clean = raw_description.strip()
    if not description_clean or len(description_clean) > _DESCRIPTION_MAX_LENGTH:
        raise PromptValidationError(
            "Tool description must be 1-200 ASCII characters.",
            dataclass_type=params_type,
            placeholder="description",
        )
    try:
        _ = description_clean.encode("ascii")
    except UnicodeEncodeError as error:
        raise PromptValidationError(
            "Tool description must be ASCII.",
            dataclass_type=params_type,
            placeholder="description",
        ) from error
    return description_clean


# ---------------------------------------------------------------------------
# Example Validation
# ---------------------------------------------------------------------------


def validate_example_description(description: str, params_type: ParamsType) -> None:
    """Validate a tool example description.

    Args:
        description: The example description to validate.
        params_type: The params type for error context.

    Raises:
        PromptValidationError: If the description is invalid.
    """
    description_clean = description.strip()
    if not description_clean or len(description_clean) > _DESCRIPTION_MAX_LENGTH:
        raise PromptValidationError(
            "Tool example description must be 1-200 ASCII characters.",
            dataclass_type=params_type,
            placeholder="description",
        )
    try:
        _ = description_clean.encode("ascii")
    except UnicodeEncodeError as error:
        raise PromptValidationError(
            "Tool example description must be ASCII.",
            dataclass_type=params_type,
            placeholder="description",
        ) from error


def is_dataclass_instance(candidate: object) -> bool:
    """Check if an object is a dataclass instance (not a class)."""
    return is_dataclass(candidate) and not isinstance(candidate, type)


def validate_example_input(
    example_input: object,
    params_type: ParamsType,
) -> None:
    """Validate a tool example input.

    Args:
        example_input: The example input to validate.
        params_type: The expected params type.

    Raises:
        PromptValidationError: If the input is invalid.
    """
    if params_type is _NONE_TYPE:
        if example_input is not None:
            raise PromptValidationError(
                "Tool example input must be None when ParamsT is None.",
                dataclass_type=params_type,
                placeholder="examples",
            )
        return
    if not is_dataclass_instance(example_input):
        raise PromptValidationError(
            "Tool example input must be a ParamsT dataclass instance.",
            dataclass_type=params_type,
            placeholder="examples",
        )
    if type(example_input) is not params_type:
        raise PromptValidationError(
            "Tool example input must match the tool params type.",
            dataclass_type=params_type,
            placeholder="examples",
        )


def validate_example_output(
    example_output: object,
    params_type: ParamsType,
    result_type: ResultType,
    result_container: Literal["object", "array"],
) -> None:
    """Validate a tool example output.

    Args:
        example_output: The example output to validate.
        params_type: The params type for error context.
        result_type: The expected result type.
        result_container: Whether the result is an "object" or "array".

    Raises:
        PromptValidationError: If the output is invalid.
    """
    if result_type is _NONE_TYPE:
        if example_output is not None:
            raise PromptValidationError(
                "Tool example output must be None when ResultT is None.",
                dataclass_type=params_type,
                placeholder="examples",
            )
        return
    if result_container == "array":
        if not isinstance(example_output, SequenceABC) or isinstance(
            example_output, (str, bytes, bytearray)
        ):
            raise PromptValidationError(
                "Tool example output must be a sequence of ResultT dataclass instances.",
                dataclass_type=params_type,
                placeholder="examples",
            )

        sequence_output = cast(SequenceABC[object], example_output)
        for item in sequence_output:
            if not is_dataclass_instance(item) or type(item) is not result_type:
                raise PromptValidationError(
                    "Tool example output must be a sequence of ResultT dataclass instances.",
                    dataclass_type=params_type,
                    placeholder="examples",
                )
        return

    if (
        not is_dataclass_instance(example_output)
        or type(example_output) is not result_type
    ):
        raise PromptValidationError(
            "Tool example output must be a ResultT dataclass instance.",
            dataclass_type=params_type,
            placeholder="examples",
        )


def validate_examples[ParamsT: SupportsDataclass | None, ResultT: SupportsToolResult](
    examples: tuple[Any, ...],
    params_type: ParamsType,
    result_type: ResultType,
    result_container: Literal["object", "array"],
) -> tuple[Any, ...]:
    """Validate all tool examples.

    Args:
        examples: The examples tuple to validate.
        params_type: The expected params type.
        result_type: The expected result type.
        result_container: Whether the result is an "object" or "array".

    Returns:
        The validated examples tuple.

    Raises:
        PromptValidationError: If any example is invalid.
    """
    # Import here to avoid circular import
    from .tool import ToolExample

    examples_value = cast(tuple[object, ...], examples)
    if not examples_value:
        return ()

    normalized_examples: list[Any] = []
    for example in examples_value:
        if not isinstance(example, ToolExample):
            raise PromptValidationError(
                "Tool examples must be ToolExample instances.",
                dataclass_type=params_type,
                placeholder="examples",
            )

        typed_example = cast(Any, example)

        validate_example_description(typed_example.description, params_type)
        validate_example_input(typed_example.input, params_type)
        validate_example_output(
            typed_example.output, params_type, result_type, result_container
        )

        normalized_examples.append(typed_example)

    return tuple(normalized_examples)


# ---------------------------------------------------------------------------
# Handler Signature Validation
# ---------------------------------------------------------------------------


def validate_parameter_count(
    parameters: list[inspect.Parameter],
    params_type: ParamsType,
) -> tuple[inspect.Parameter, inspect.Parameter]:
    """Validate handler has exactly two parameters.

    Args:
        parameters: The list of handler parameters.
        params_type: The params type for error context.

    Returns:
        A tuple of (params_parameter, context_parameter).

    Raises:
        PromptValidationError: If the parameter count is wrong.
    """
    if len(parameters) != _HANDLER_PARAMETER_COUNT:
        raise PromptValidationError(
            "Tool handler must accept exactly one positional argument and the keyword-only 'context' parameter.",
            dataclass_type=params_type,
            placeholder="handler",
        )

    parameter = parameters[0]
    context_parameter = parameters[1]
    return parameter, context_parameter


def validate_parameter_kind(
    parameter: inspect.Parameter,
    params_type: ParamsType,
) -> None:
    """Validate the params parameter is positional.

    Args:
        parameter: The params parameter to validate.
        params_type: The params type for error context.

    Raises:
        PromptValidationError: If the parameter kind is wrong.
    """
    if parameter.kind not in _POSITIONAL_PARAMETER_KINDS:
        raise PromptValidationError(
            "Tool handler parameter must be positional.",
            dataclass_type=params_type,
            placeholder="handler",
        )


def validate_context_parameter(
    context_parameter: inspect.Parameter,
    params_type: ParamsType,
) -> None:
    """Validate the context parameter requirements.

    Args:
        context_parameter: The context parameter to validate.
        params_type: The params type for error context.

    Raises:
        PromptValidationError: If the context parameter is invalid.
    """
    if context_parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
        raise PromptValidationError(
            "Tool handler must declare a keyword-only 'context' parameter.",
            dataclass_type=params_type,
            placeholder="handler",
        )
    if context_parameter.default is not inspect.Signature.empty:
        raise PromptValidationError(
            "Tool handler 'context' parameter must not define a default value.",
            dataclass_type=params_type,
            placeholder="handler",
        )
    if context_parameter.name != "context":
        raise PromptValidationError(
            "Tool handler must name the keyword-only context parameter 'context'.",
            dataclass_type=params_type,
            placeholder="handler",
        )


def resolve_annotations(
    callable_handler: Callable[..., Any],
) -> dict[str, object]:
    """Resolve type hints for a handler, falling back on failure.

    Args:
        callable_handler: The handler callable to inspect.

    Returns:
        A dictionary of resolved type hints, or empty dict on failure.
    """
    try:
        return get_type_hints(callable_handler, include_extras=True)
    except Exception:
        return {}


def validate_parameter_annotation(
    annotation: object,
    params_type: ParamsType,
) -> None:
    """Validate the params parameter annotation matches ParamsT.

    Args:
        annotation: The parameter annotation to validate.
        params_type: The expected params type.

    Raises:
        PromptValidationError: If the annotation is invalid.
    """
    if annotation is inspect.Parameter.empty:
        raise PromptValidationError(
            "Tool handler parameter must be annotated with ParamsT.",
            dataclass_type=params_type,
            placeholder="handler",
        )
    annotation_to_validate = annotation
    if get_origin(annotation_to_validate) is Annotated:
        annotation_to_validate = get_args(annotation_to_validate)[0]
    literal_origin = get_origin(annotation_to_validate)
    if literal_origin is Literal:
        literal_args = get_args(annotation_to_validate)
        annotation_to_validate = (
            literal_args[0] if literal_args else annotation_to_validate
        )
    if params_type is _NONE_TYPE and annotation_to_validate in {
        None,
        _NONE_TYPE,
    }:
        return
    if annotation_to_validate is not params_type:
        raise PromptValidationError(
            "Tool handler parameter annotation must match ParamsT.",
            dataclass_type=params_type,
            placeholder="handler",
        )


def validate_context_annotation(
    context_annotation: object,
    params_type: ParamsType,
) -> None:
    """Validate the context parameter annotation is ToolContext.

    Args:
        context_annotation: The context annotation to validate.
        params_type: The params type for error context.

    Raises:
        PromptValidationError: If the annotation is invalid.
    """
    # Import here to avoid circular import
    from .tool import ToolContext

    if context_annotation is inspect.Parameter.empty:
        raise PromptValidationError(
            "Tool handler must annotate the 'context' parameter with ToolContext.",
            dataclass_type=params_type,
            placeholder="handler",
        )
    context_annotation_to_validate = context_annotation
    if get_origin(context_annotation_to_validate) is Annotated:
        context_annotation_to_validate = get_args(context_annotation_to_validate)[0]
    if context_annotation_to_validate is not ToolContext:
        raise PromptValidationError(
            "Tool handler 'context' annotation must be ToolContext.",
            dataclass_type=params_type,
            placeholder="handler",
        )


def validate_return_annotation(
    hints: dict[str, object],
    signature: inspect.Signature,
    result_annotation: object,
    params_type: ParamsType,
) -> None:
    """Validate the handler return annotation is ToolResult[ResultT].

    Args:
        hints: The resolved type hints dictionary.
        signature: The handler signature.
        result_annotation: The expected result annotation.
        params_type: The params type for error context.

    Raises:
        PromptValidationError: If the return annotation is invalid.
    """
    # Import here to avoid circular import
    from .tool_result import ToolResult

    return_annotation = hints.get("return", signature.return_annotation)
    if return_annotation is inspect.Signature.empty:
        raise PromptValidationError(
            "Tool handler must annotate its return value with ToolResult[ResultT].",
            dataclass_type=params_type,
            placeholder="return",
        )
    return_annotation_to_validate = return_annotation
    if get_origin(return_annotation_to_validate) is Annotated:
        return_annotation_to_validate = get_args(return_annotation_to_validate)[0]

    origin = get_origin(return_annotation_to_validate)
    if origin is ToolResult:
        result_args_raw = get_args(return_annotation_to_validate)
        if result_args_raw and matches_result_annotation(
            result_args_raw[0],
            result_annotation,
        ):
            return
    raise PromptValidationError(
        "Tool handler return annotation must be ToolResult[ResultT].",
        dataclass_type=params_type,
        placeholder="return",
    )


def validate_handler[ResultT](
    handler: object,
    params_type: ParamsType,
    result_annotation: object,
) -> None:
    """Validate a tool handler's signature and annotations.

    Args:
        handler: The handler callable to validate.
        params_type: The expected params type.
        result_annotation: The expected result annotation.

    Raises:
        PromptValidationError: If the handler is invalid.
    """
    # Import here to avoid circular import
    from .tool_result import ToolResult

    if not callable(handler):
        raise PromptValidationError(
            "Tool handler must be callable.",
            dataclass_type=params_type,
            placeholder="handler",
        )

    callable_handler = cast(Callable[..., ToolResult[ResultT]], handler)
    signature = inspect.signature(callable_handler)
    parameters = list(signature.parameters.values())

    parameter, context_parameter = validate_parameter_count(
        parameters,
        params_type,
    )

    validate_parameter_kind(parameter, params_type)
    validate_context_parameter(context_parameter, params_type)

    hints = resolve_annotations(callable_handler)
    annotation = hints.get(parameter.name, parameter.annotation)
    validate_parameter_annotation(annotation, params_type)

    context_annotation = hints.get(context_parameter.name, context_parameter.annotation)
    validate_context_annotation(context_annotation, params_type)

    validate_return_annotation(
        hints,
        signature,
        result_annotation,
        params_type,
    )


def validate_handler_if_present(
    handler: object | None,
    params_type: ParamsType,
    raw_result_annotation: object,
) -> None:
    """Validate handler if it exists.

    Args:
        handler: The handler to validate, or None.
        params_type: The expected params type.
        raw_result_annotation: The expected result annotation.
    """
    if handler is None:
        return
    validate_handler(
        handler,
        params_type,
        raw_result_annotation,
    )


# ---------------------------------------------------------------------------
# Result Annotation Handling
# ---------------------------------------------------------------------------


def normalize_result_annotation(
    annotation: object,
    params_type: ParamsType,
) -> tuple[ResultType, Literal["object", "array"]]:
    """Normalize a result annotation to a type and container.

    Args:
        annotation: The result annotation to normalize.
        params_type: The params type for error context.

    Returns:
        A tuple of (result_type, container_kind).

    Raises:
        PromptValidationError: If the annotation is invalid.
    """
    if annotation is None:
        return _NONE_TYPE, "object"
    if isinstance(annotation, type):
        return cast(ResultType, annotation), "object"

    origin = get_origin(annotation)
    if origin not in {list, tuple, SequenceABC}:
        raise PromptValidationError(
            "Tool ResultT must be a dataclass type or a sequence of dataclasses.",
            dataclass_type=params_type,
            placeholder="ResultT",
        )

    args = get_args(annotation)
    if origin is tuple and (
        len(args) != _VARIADIC_TUPLE_LENGTH or args[1] is not Ellipsis
    ):
        raise PromptValidationError(
            "Variadic Tuple[ResultT, ...] is required for Tool sequence results.",
            dataclass_type=params_type,
            placeholder="ResultT",
        )
    # list[T] and Sequence[T] always have exactly one type arg
    element = args[0]

    if not isinstance(element, type):
        raise PromptValidationError(
            "Tool ResultT must be a dataclass type or a sequence of dataclasses.",
            dataclass_type=params_type,
            placeholder="ResultT",
        )

    return cast(ResultType, element), "array"


def matches_result_annotation(candidate: object, expected: object) -> bool:
    """Check if a candidate result annotation matches the expected one.

    Args:
        candidate: The candidate annotation to check.
        expected: The expected annotation to match against.

    Returns:
        True if the annotations match, False otherwise.
    """
    candidate = coerce_none_type(candidate)
    expected = coerce_none_type(expected)
    if candidate is expected:
        return True

    candidate_origin = get_origin(candidate)
    expected_origin = get_origin(expected)

    if candidate_origin is None or expected_origin is None:
        return False

    sequence_origins = {list, tuple, SequenceABC}
    if candidate_origin in sequence_origins and expected_origin in sequence_origins:
        candidate_args = get_args(candidate)
        expected_args = get_args(expected)
        candidate_element = (
            candidate_args[0]
            if candidate_origin is not tuple
            else candidate_args[0]
            if len(candidate_args) == _VARIADIC_TUPLE_LENGTH
            else None
        )
        expected_element = (
            expected_args[0]
            if expected_origin is not tuple
            else expected_args[0]
            if len(expected_args) == _VARIADIC_TUPLE_LENGTH
            else None
        )
        return candidate_element is expected_element

    return False


# ---------------------------------------------------------------------------
# Wrapped Tool Resolution Helpers
# ---------------------------------------------------------------------------


def resolve_wrapped_description(
    fn: Callable[..., Any],
) -> str:
    """Resolve the description from a handler's docstring.

    Args:
        fn: The handler function to inspect.

    Returns:
        The docstring as the description.

    Raises:
        PromptValidationError: If no docstring is present.
    """
    description = inspect.getdoc(fn)
    if description is None:
        raise PromptValidationError(
            "Tool handler must define a docstring to use as the description.",
            placeholder="description",
        )
    return description


def resolve_wrapped_params_type(
    parameter: inspect.Parameter, hints: dict[str, object]
) -> ParamsType:
    """Resolve the params type from a handler parameter.

    Args:
        parameter: The parameter to inspect.
        hints: The resolved type hints dictionary.

    Returns:
        The resolved params type.

    Raises:
        PromptValidationError: If the parameter is not annotated.
    """
    annotation = hints.get(parameter.name, parameter.annotation)
    if annotation is inspect.Parameter.empty:
        raise PromptValidationError(
            "Tool handler parameter must be annotated with ParamsT.",
            placeholder="handler",
        )

    params_annotation = annotation
    if get_origin(params_annotation) is Annotated:
        params_annotation = get_args(params_annotation)[0]
    if get_origin(params_annotation) is Literal:
        literal_args = get_args(params_annotation)
        params_annotation = literal_args[0] if literal_args else params_annotation

    return cast(ParamsType, coerce_none_type(params_annotation))


def resolve_wrapped_result_annotation(
    signature: inspect.Signature,
    hints: dict[str, object],
    params_type: ParamsType,
) -> SupportsToolResult:
    """Resolve the result annotation from a handler return type.

    Args:
        signature: The handler signature.
        hints: The resolved type hints dictionary.
        params_type: The params type for error context.

    Returns:
        The resolved result annotation.

    Raises:
        PromptValidationError: If the return type is invalid.
    """
    # Import here to avoid circular import
    from .tool_result import ToolResult

    return_annotation = hints.get("return", signature.return_annotation)
    if return_annotation is inspect.Signature.empty:
        raise PromptValidationError(
            "Tool handler must annotate its return value with ToolResult[ResultT].",
            dataclass_type=params_type,
            placeholder="return",
        )
    result_annotation = return_annotation
    if get_origin(result_annotation) is Annotated:
        result_annotation = get_args(result_annotation)[0]

    if get_origin(result_annotation) is not ToolResult:
        raise PromptValidationError(
            "Tool handler return annotation must be ToolResult[ResultT].",
            dataclass_type=params_type,
            placeholder="return",
        )

    try:
        result_arg = next(iter(cast(tuple[object, ...], get_args(result_annotation))))
    except StopIteration as error:
        raise PromptValidationError(
            "Tool handler return annotation must be ToolResult[ResultT].",
            dataclass_type=params_type,
            placeholder="return",
        ) from error

    return cast(SupportsToolResult, coerce_none_type(result_arg))


__all__ = [
    "ParamsType",
    "ResultType",
    "coerce_none_type",
    "is_dataclass_instance",
    "matches_result_annotation",
    "normalize_result_annotation",
    "resolve_annotations",
    "resolve_wrapped_description",
    "resolve_wrapped_params_type",
    "resolve_wrapped_result_annotation",
    "validate_context_annotation",
    "validate_context_parameter",
    "validate_description",
    "validate_example_description",
    "validate_example_input",
    "validate_example_output",
    "validate_examples",
    "validate_handler",
    "validate_handler_if_present",
    "validate_name",
    "validate_parameter_annotation",
    "validate_parameter_count",
    "validate_parameter_kind",
    "validate_return_annotation",
]
