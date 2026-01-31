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

"""Tool validation logic extracted for reuse and testability.

This module provides the ``ToolValidator`` class for validating tool definitions
against API constraints. Validation is separated from the Tool dataclass to:

1. Keep Tool as a simple data container
2. Enable reuse of validation (e.g., for override validation)
3. Isolate type resolution and validation logic for testing
"""

from __future__ import annotations

import inspect
import re
import types
from collections.abc import Callable, Sequence as SequenceABC
from dataclasses import dataclass
from typing import (
    Annotated,
    Final,
    Literal,
    Protocol,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from ..types.dataclass import (
    SupportsDataclass,
    SupportsToolResult,
    is_dataclass_instance,
)
from .errors import PromptValidationError
from .tool_result import ToolResult


class _ToolExampleLike(Protocol):
    """Protocol for ToolExample-like objects used in validation."""

    @property
    def description(self) -> str: ...

    @property
    def input(self) -> object: ...

    @property
    def output(self) -> object: ...


_NAME_MIN_LENGTH: Final = 1
_NAME_MAX_LENGTH: Final = 64
_DESCRIPTION_MAX_LENGTH: Final = 200
_EXPECTED_TYPE_ARGUMENTS: Final = 2
_HANDLER_PARAMETER_COUNT: Final = 2
_VARIADIC_TUPLE_LENGTH: Final = 2
_NONE_TYPE: Final = type(None)

_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"^[a-z0-9_-]{{{_NAME_MIN_LENGTH},{_NAME_MAX_LENGTH}}}$"
)

type ParamsType = type[SupportsDataclass] | type[None]
type ResultType = type[SupportsDataclass] | type[None]


def coerce_none_type(candidate: object) -> object:
    """Normalize None and union types containing None.

    Converts:
    - ``None`` literal to ``type(None)``
    - Union types like ``T | None`` to the non-None type ``T``
    - All-None unions to ``type(None)``
    """
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


def normalize_specialization(item: object) -> tuple[object, object]:
    """Parse and normalize Tool[ParamsT, ResultT] type arguments."""
    if not isinstance(item, tuple):
        raise TypeError("Tool[...] expects two type arguments (ParamsT, ResultT).")
    normalized = cast(SequenceABC[object], item)
    if len(normalized) != _EXPECTED_TYPE_ARGUMENTS:
        raise TypeError("Tool[...] expects two type arguments (ParamsT, ResultT).")
    return coerce_none_type(normalized[0]), coerce_none_type(normalized[1])


@dataclass(slots=True, frozen=True)
class ToolValidator:
    """Validates tool definitions against API constraints.

    This class encapsulates all validation logic for tools, including:
    - Name validation (pattern matching, length constraints)
    - Description validation (length, ASCII encoding)
    - Example validation (type matching, dataclass instances)
    - Type resolution from handler signatures

    Example::

        validator = ToolValidator()
        validated_name = validator.validate_name("my_tool", params_type=MyParams)
        validated_desc = validator.validate_description("Does something.", params_type=MyParams)
    """

    def validate_name(self, name: str, *, params_type: ParamsType) -> str:
        """Validate and normalize a tool name.

        Args:
            name: Raw tool name to validate.
            params_type: Parameter type for error context.

        Returns:
            Validated name (unchanged if valid).

        Raises:
            PromptValidationError: If name has whitespace, is empty, or violates pattern.
        """
        raw_name = name
        stripped_name = raw_name.strip()
        if raw_name != stripped_name:
            raise PromptValidationError(
                "Tool name must not contain surrounding whitespace.",
                dataclass_type=params_type,
                placeholder=stripped_name,
            )

        name_clean = raw_name
        if not name_clean:
            raise PromptValidationError(
                (
                    "Tool name must match the OpenAI function name constraints "
                    "(1-64 lowercase ASCII letters, digits, underscores, or hyphens)."
                ),
                dataclass_type=params_type,
                placeholder=stripped_name,
            )
        if len(name_clean) > _NAME_MAX_LENGTH or not _NAME_PATTERN.fullmatch(
            name_clean
        ):
            raise PromptValidationError(
                (
                    f"Tool name must match the OpenAI function name constraints "
                    f"(pattern: {_NAME_PATTERN.pattern})."
                ),
                dataclass_type=params_type,
                placeholder=name_clean,
            )

        return name_clean

    def validate_description(self, description: str, *, params_type: ParamsType) -> str:
        """Validate and normalize a tool description.

        Args:
            description: Raw description to validate.
            params_type: Parameter type for error context.

        Returns:
            Stripped description if valid.

        Raises:
            PromptValidationError: If description is empty, too long, or non-ASCII.
        """
        description_clean = description.strip()
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

    def validate_example_description(
        self, description: str, *, params_type: ParamsType
    ) -> None:
        """Validate an example description.

        Args:
            description: Example description to validate.
            params_type: Parameter type for error context.

        Raises:
            PromptValidationError: If description is empty, too long, or non-ASCII.
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

    def validate_example_input(
        self,
        example_input: object,
        *,
        params_type: ParamsType,
    ) -> None:
        """Validate that example input matches the expected params type.

        Args:
            example_input: Input value to validate.
            params_type: Expected parameter type.

        Raises:
            PromptValidationError: If input is not None when params is None,
                not a dataclass instance, or wrong type.
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
        self,
        example_output: object,
        *,
        params_type: ParamsType,
        result_type: ResultType,
        result_container: Literal["object", "array"],
    ) -> None:
        """Validate that example output matches the expected result type.

        Args:
            example_output: Output value to validate.
            params_type: Parameter type for error context.
            result_type: Expected result element type.
            result_container: Whether result is "object" or "array".

        Raises:
            PromptValidationError: If output doesn't match expected type.
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

    def validate_examples(
        self,
        examples: tuple[object, ...],
        *,
        params_type: ParamsType,
        result_type: ResultType,
        result_container: Literal["object", "array"],
    ) -> tuple[object, ...]:
        """Validate all examples for a tool.

        Args:
            examples: Tuple of examples to validate.
            params_type: Expected parameter type.
            result_type: Expected result element type.
            result_container: Whether result is "object" or "array".

        Returns:
            Validated examples tuple.

        Raises:
            PromptValidationError: If any example is invalid.
        """
        if not examples:
            return ()

        normalized_examples: list[object] = []
        for example in examples:
            # Check for ToolExample duck type (has description, input, output)
            if not (
                hasattr(example, "description")
                and hasattr(example, "input")
                and hasattr(example, "output")
            ):
                raise PromptValidationError(
                    "Tool examples must be ToolExample instances.",
                    dataclass_type=params_type,
                    placeholder="examples",
                )

            # Cast to protocol for type-safe attribute access
            typed_example = cast(_ToolExampleLike, example)

            self.validate_example_description(
                typed_example.description,
                params_type=params_type,
            )
            self.validate_example_input(
                typed_example.input,
                params_type=params_type,
            )
            self.validate_example_output(
                typed_example.output,
                params_type=params_type,
                result_type=result_type,
                result_container=result_container,
            )

            normalized_examples.append(example)

        return tuple(normalized_examples)

    def validate_parameter_count(
        self,
        parameters: list[inspect.Parameter],
        *,
        params_type: ParamsType,
    ) -> tuple[inspect.Parameter, inspect.Parameter]:
        """Validate that handler has exactly params and context parameters.

        Args:
            parameters: List of handler parameters.
            params_type: Parameter type for error context.

        Returns:
            Tuple of (params_parameter, context_parameter).

        Raises:
            PromptValidationError: If parameter count is wrong.
        """
        if len(parameters) != _HANDLER_PARAMETER_COUNT:
            raise PromptValidationError(
                (
                    "Tool handler must accept exactly one positional argument "
                    "and the keyword-only 'context' parameter."
                ),
                dataclass_type=params_type,
                placeholder="handler",
            )

        parameter = parameters[0]
        context_parameter = parameters[1]
        return parameter, context_parameter


@dataclass(slots=True, frozen=True)
class TypeResolver:
    """Resolves type annotations from handler signatures and Tool specializations."""

    def resolve_annotations(
        self,
        callable_handler: Callable[..., object],
    ) -> dict[str, object]:
        """Safely resolve type hints from a callable.

        Args:
            callable_handler: Handler function to inspect.

        Returns:
            Type hints dictionary, empty if resolution fails.
        """
        try:
            return get_type_hints(callable_handler, include_extras=True)
        except Exception:
            return {}

    def normalize_result_annotation(
        self,
        annotation: SupportsToolResult,
        *,
        params_type: ParamsType,
    ) -> tuple[ResultType, Literal["object", "array"]]:
        """Normalize a result type annotation to (element_type, container_type).

        Args:
            annotation: Raw result type annotation.
            params_type: Parameter type for error context.

        Returns:
            Tuple of (result_element_type, container_kind).

        Raises:
            PromptValidationError: If annotation is not a valid result type.
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

    def resolve_wrapped_description(
        self,
        fn: Callable[..., object],
    ) -> str:
        """Extract description from handler docstring.

        Args:
            fn: Handler function with docstring.

        Returns:
            Docstring content.

        Raises:
            PromptValidationError: If handler has no docstring.
        """
        description = inspect.getdoc(fn)
        if description is None:
            raise PromptValidationError(
                "Tool handler must define a docstring to use as the description.",
                placeholder="description",
            )
        return description

    def resolve_wrapped_params_type(
        self, parameter: inspect.Parameter, hints: dict[str, object]
    ) -> ParamsType:
        """Resolve parameter type from handler annotation.

        Args:
            parameter: Handler's params parameter.
            hints: Resolved type hints.

        Returns:
            Resolved parameter type.

        Raises:
            PromptValidationError: If parameter is not annotated.
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
        self,
        signature: inspect.Signature,
        hints: dict[str, object],
        *,
        params_type: ParamsType,
    ) -> SupportsToolResult:
        """Resolve result type from handler return annotation.

        Args:
            signature: Handler signature.
            hints: Resolved type hints.
            params_type: Parameter type for error context.

        Returns:
            Resolved result type annotation.

        Raises:
            PromptValidationError: If return is not annotated as ToolResult[T].
        """
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
            result_arg = next(
                iter(cast(tuple[object, ...], get_args(result_annotation)))
            )
        except StopIteration as error:
            raise PromptValidationError(
                "Tool handler return annotation must be ToolResult[ResultT].",
                dataclass_type=params_type,
                placeholder="return",
            ) from error

        return cast(SupportsToolResult, coerce_none_type(result_arg))


# Module-level singletons for convenience
_DEFAULT_VALIDATOR: Final = ToolValidator()
_DEFAULT_RESOLVER: Final = TypeResolver()


def get_validator() -> ToolValidator:
    """Return the default ToolValidator instance."""
    return _DEFAULT_VALIDATOR


def get_type_resolver() -> TypeResolver:
    """Return the default TypeResolver instance."""
    return _DEFAULT_RESOLVER


__all__ = [
    "ParamsType",
    "ResultType",
    "ToolValidator",
    "TypeResolver",
    "coerce_none_type",
    "get_type_resolver",
    "get_validator",
    "normalize_specialization",
]
