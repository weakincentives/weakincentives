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

import inspect
import re
from collections.abc import Callable, Sequence as SequenceABC
from dataclasses import dataclass, field, is_dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    Literal,
    Protocol,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from ..deadlines import Deadline
from ._types import SupportsDataclass, SupportsToolResult
from .errors import PromptValidationError
from .tool_result import ToolResult

_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9_-]{1,64}$")


if TYPE_CHECKING:
    from ..runtime.events._types import EventBus
    from ..runtime.session.protocols import SessionProtocol
    from .protocols import (
        PromptProtocol,
        ProviderAdapterProtocol,
        RenderedPromptProtocol,
    )

ParamsT_contra = TypeVar("ParamsT_contra", bound=SupportsDataclass, contravariant=True)
ResultT_co = TypeVar("ResultT_co", bound=SupportsToolResult)


@dataclass(slots=True, frozen=True)
class ToolExample[ParamsT: SupportsDataclass, ResultT: SupportsToolResult]:
    """Representative invocation for a tool documenting inputs and outputs."""

    description: str
    input: ParamsT
    output: ResultT


@dataclass(slots=True, frozen=True)
class ToolContext:
    """Immutable container exposing prompt execution state to handlers."""

    prompt: PromptProtocol[Any]
    rendered_prompt: RenderedPromptProtocol[Any] | None
    adapter: ProviderAdapterProtocol[Any]
    session: SessionProtocol
    event_bus: EventBus
    deadline: Deadline | None = None


def _normalize_specialization(item: object) -> tuple[object, object]:
    if not isinstance(item, tuple):
        raise TypeError("Tool[...] expects two type arguments (ParamsT, ResultT).")
    normalized = cast(SequenceABC[object], item)
    if len(normalized) != 2:
        raise TypeError("Tool[...] expects two type arguments (ParamsT, ResultT).")
    return normalized[0], normalized[1]


class ToolHandler(Protocol[ParamsT_contra, ResultT_co]):
    """Callable protocol implemented by tool handlers."""

    def __call__(
        self, params: ParamsT_contra, *, context: ToolContext
    ) -> ToolResult[ResultT_co]: ...


@dataclass(slots=True)
class Tool[ParamsT: SupportsDataclass, ResultT: SupportsToolResult]:
    """Describe a callable tool exposed by prompt sections."""

    name: str
    description: str
    handler: ToolHandler[ParamsT, ResultT] | None
    examples: tuple[ToolExample[ParamsT, ResultT], ...] = field(
        default_factory=tuple,
    )
    params_type: type[ParamsT] = field(init=False, repr=False)
    result_type: type[SupportsDataclass] = field(init=False, repr=False)
    result_container: Literal["object", "array"] = field(
        init=False,
        repr=False,
    )
    _result_annotation: ResultT = field(init=False, repr=False)
    accepts_overrides: bool = True

    def __post_init__(self) -> None:
        params_type, raw_result_annotation = self._resolve_type_arguments()

        result_type, result_container = self._normalize_result_annotation(
            raw_result_annotation,
            params_type,
        )

        self.params_type = cast(type[ParamsT], params_type)
        self.result_type = result_type
        self.result_container = result_container
        self._result_annotation = raw_result_annotation

        self.name = self._validate_name(params_type)
        self.description = self._validate_description(params_type)
        self.examples = self._validate_examples(
            params_type,
            result_type,
        )

        self._validate_handler_if_present(
            params_type,
            raw_result_annotation,
        )

    def _resolve_type_arguments(
        self,
    ) -> tuple[type[SupportsDataclass], ResultT]:
        params_attr = getattr(self, "params_type", None)
        params_type: type[SupportsDataclass] | None = (
            params_attr if isinstance(params_attr, type) else None
        )
        raw_result_annotation = getattr(self, "_result_annotation", None)
        if params_type is None or raw_result_annotation is None:
            origin = getattr(self, "__orig_class__", None)
            if origin is not None:  # pragma: no cover - interpreter-specific path
                args = get_args(origin)
                if len(args) == 2:
                    params_arg, result_arg = args
                    if isinstance(params_arg, type):
                        params_type = params_arg
                        raw_result_annotation = cast(ResultT, result_arg)
        if params_type is None or raw_result_annotation is None:
            raise PromptValidationError(
                "Tool must be instantiated with concrete type arguments.",
                placeholder="type_arguments",
            )

        return params_type, cast(ResultT, raw_result_annotation)

    def _validate_name(self, params_type: type[SupportsDataclass]) -> str:
        raw_name = self.name
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
        if len(name_clean) > 64 or not _NAME_PATTERN.fullmatch(name_clean):
            raise PromptValidationError(
                "Tool name must match the OpenAI function name constraints (pattern: ^[a-z0-9_-]{1,64}$).",
                dataclass_type=params_type,
                placeholder=name_clean,
            )

        return name_clean

    def _validate_description(self, params_type: type[SupportsDataclass]) -> str:
        description_clean = self.description.strip()
        if not description_clean or len(description_clean) > 200:
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

    @staticmethod
    def _validate_example_description(
        description: str, params_type: type[SupportsDataclass]
    ) -> None:
        description_clean = description.strip()
        if not description_clean or len(description_clean) > 200:
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

    @staticmethod
    def _is_dataclass_instance(candidate: object) -> bool:
        return is_dataclass(candidate) and not isinstance(candidate, type)

    def _validate_example_input(
        self,
        example_input: object,
        params_type: type[SupportsDataclass],
    ) -> None:
        if not self._is_dataclass_instance(example_input):
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

    def _validate_example_output(
        self,
        example_output: object,
        params_type: type[SupportsDataclass],
        result_type: type[SupportsDataclass],
    ) -> None:
        if self.result_container == "array":
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
                if (
                    not self._is_dataclass_instance(item)
                    or type(item) is not result_type
                ):
                    raise PromptValidationError(
                        "Tool example output must be a sequence of ResultT dataclass instances.",
                        dataclass_type=params_type,
                        placeholder="examples",
                    )
            return

        if (
            not self._is_dataclass_instance(example_output)
            or type(example_output) is not result_type
        ):
            raise PromptValidationError(
                "Tool example output must be a ResultT dataclass instance.",
                dataclass_type=params_type,
                placeholder="examples",
            )

    def _validate_examples(
        self,
        params_type: type[SupportsDataclass],
        result_type: type[SupportsDataclass],
    ) -> tuple[ToolExample[ParamsT, ResultT], ...]:
        examples_value = cast(tuple[object, ...], self.examples)
        if not examples_value:
            return ()

        normalized_examples: list[ToolExample[ParamsT, ResultT]] = []
        for example in examples_value:
            if not isinstance(example, ToolExample):
                raise PromptValidationError(
                    "Tool examples must be ToolExample instances.",
                    dataclass_type=params_type,
                    placeholder="examples",
                )

            typed_example = cast(ToolExample[ParamsT, ResultT], example)

            self._validate_example_description(typed_example.description, params_type)
            self._validate_example_input(typed_example.input, params_type)
            self._validate_example_output(
                typed_example.output, params_type, result_type
            )

            normalized_examples.append(typed_example)

        return tuple(normalized_examples)

    def _validate_handler_if_present(
        self,
        params_type: type[SupportsDataclass],
        raw_result_annotation: object,
    ) -> None:
        handler = self.handler
        if handler is None:
            return
        self._validate_handler(
            handler,
            params_type,
            raw_result_annotation,
        )

    def _validate_handler(
        self,
        handler: object,
        params_type: type[SupportsDataclass],
        result_annotation: object,
    ) -> None:
        if not callable(handler):
            raise PromptValidationError(
                "Tool handler must be callable.",
                dataclass_type=params_type,
                placeholder="handler",
            )

        callable_handler = cast(Callable[..., ToolResult[ResultT]], handler)
        signature = inspect.signature(callable_handler)
        parameters = list(signature.parameters.values())

        parameter, context_parameter = self._validate_parameter_count(
            parameters,
            params_type,
        )

        self._validate_parameter_kind(parameter, params_type)
        self._validate_context_parameter(context_parameter, params_type)

        hints = self._resolve_annotations(callable_handler)
        annotation = hints.get(parameter.name, parameter.annotation)
        self._validate_parameter_annotation(annotation, params_type)

        context_annotation = hints.get(
            context_parameter.name, context_parameter.annotation
        )
        self._validate_context_annotation(context_annotation, params_type)

        self._validate_return_annotation(
            hints,
            signature,
            result_annotation,
            params_type,
        )

    @staticmethod
    def _validate_parameter_count(
        parameters: list[inspect.Parameter],
        params_type: type[SupportsDataclass],
    ) -> tuple[inspect.Parameter, inspect.Parameter]:
        if len(parameters) != 2:
            raise PromptValidationError(
                "Tool handler must accept exactly one positional argument and the keyword-only 'context' parameter.",
                dataclass_type=params_type,
                placeholder="handler",
            )

        parameter = parameters[0]
        context_parameter = parameters[1]
        return parameter, context_parameter

    @staticmethod
    def _validate_parameter_kind(
        parameter: inspect.Parameter,
        params_type: type[SupportsDataclass],
    ) -> None:
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise PromptValidationError(
                "Tool handler parameter must be positional.",
                dataclass_type=params_type,
                placeholder="handler",
            )

    @staticmethod
    def _validate_context_parameter(
        context_parameter: inspect.Parameter,
        params_type: type[SupportsDataclass],
    ) -> None:
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

    @staticmethod
    def _resolve_annotations(
        callable_handler: Callable[..., ToolResult[ResultT]],
    ) -> dict[str, object]:
        try:
            return get_type_hints(callable_handler, include_extras=True)
        except Exception:
            return {}

    @staticmethod
    def _validate_parameter_annotation(
        annotation: object,
        params_type: type[SupportsDataclass],
    ) -> None:
        if annotation is inspect.Parameter.empty:
            raise PromptValidationError(
                "Tool handler parameter must be annotated with ParamsT.",
                dataclass_type=params_type,
                placeholder="handler",
            )
        if get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        if annotation is not params_type:
            raise PromptValidationError(
                "Tool handler parameter annotation must match ParamsT.",
                dataclass_type=params_type,
                placeholder="handler",
            )

    @staticmethod
    def _validate_context_annotation(
        context_annotation: object,
        params_type: type[SupportsDataclass],
    ) -> None:
        if context_annotation is inspect.Parameter.empty:
            raise PromptValidationError(
                "Tool handler must annotate the 'context' parameter with ToolContext.",
                dataclass_type=params_type,
                placeholder="handler",
            )
        if get_origin(context_annotation) is Annotated:
            context_annotation = get_args(context_annotation)[0]
        if context_annotation is not ToolContext:
            raise PromptValidationError(
                "Tool handler 'context' annotation must be ToolContext.",
                dataclass_type=params_type,
                placeholder="handler",
            )

    def _validate_return_annotation(
        self,
        hints: dict[str, object],
        signature: inspect.Signature,
        result_annotation: object,
        params_type: type[SupportsDataclass],
    ) -> None:
        return_annotation = hints.get("return", signature.return_annotation)
        if return_annotation is inspect.Signature.empty:
            raise PromptValidationError(
                "Tool handler must annotate its return value with ToolResult[ResultT].",
                dataclass_type=params_type,
                placeholder="return",
            )
        if get_origin(return_annotation) is Annotated:
            return_annotation = get_args(return_annotation)[0]

        origin = get_origin(return_annotation)
        if origin is ToolResult:
            result_args_raw = get_args(return_annotation)
            if result_args_raw and self._matches_result_annotation(
                result_args_raw[0],
                result_annotation,
            ):
                return
        raise PromptValidationError(
            "Tool handler return annotation must be ToolResult[ResultT].",
            dataclass_type=params_type,
            placeholder="return",
        )

    @staticmethod
    def _normalize_result_annotation(
        annotation: ResultT,
        params_type: type[SupportsDataclass],
    ) -> tuple[type[SupportsDataclass], Literal["object", "array"]]:
        if isinstance(annotation, type):
            return cast(type[SupportsDataclass], annotation), "object"

        origin = get_origin(annotation)
        if origin not in {list, tuple, SequenceABC}:
            raise PromptValidationError(
                "Tool ResultT must be a dataclass type or a sequence of dataclasses.",
                dataclass_type=params_type,
                placeholder="ResultT",
            )

        args = get_args(annotation)
        element: object | None = None
        if origin is tuple:
            if len(args) != 2 or args[1] is not Ellipsis:
                raise PromptValidationError(
                    "Variadic Tuple[ResultT, ...] is required for Tool sequence results.",
                    dataclass_type=params_type,
                    placeholder="ResultT",
                )
            element = args[0]
        elif len(args) == 1:
            element = args[0]

        if not isinstance(element, type):
            raise PromptValidationError(
                "Tool ResultT must be a dataclass type or a sequence of dataclasses.",
                dataclass_type=params_type,
                placeholder="ResultT",
            )

        return cast(type[SupportsDataclass], element), "array"

    @staticmethod
    def _matches_result_annotation(candidate: object, expected: object) -> bool:
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
                if len(candidate_args) == 2
                else None
            )
            expected_element = (
                expected_args[0]
                if expected_origin is not tuple
                else expected_args[0]
                if len(expected_args) == 2
                else None
            )
            return candidate_element is expected_element

        return False

    @classmethod
    def __class_getitem__(
        cls, item: object
    ) -> type[Tool[SupportsDataclass, SupportsToolResult]]:
        params_candidate, result_candidate = _normalize_specialization(item)
        if not isinstance(params_candidate, type):
            raise TypeError("Tool ParamsT type argument must be a type.")
        params_type = cast(type[SupportsDataclass], params_candidate)
        result_annotation = cast(ResultT, result_candidate)

        class _SpecializedTool(cls):
            def __post_init__(self) -> None:
                self.params_type = cast(type[ParamsT], params_type)
                self._result_annotation = result_annotation
                super().__post_init__()

        _SpecializedTool.__name__ = cls.__name__
        _SpecializedTool.__qualname__ = cls.__qualname__
        _SpecializedTool.__module__ = cls.__module__
        return cast(
            "type[Tool[SupportsDataclass, SupportsToolResult]]",
            _SpecializedTool,
        )


__all__ = ["Tool", "ToolContext", "ToolExample", "ToolHandler", "ToolResult"]
