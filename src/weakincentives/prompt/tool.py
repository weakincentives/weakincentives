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
from collections.abc import Callable
from dataclasses import dataclass, field, is_dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    Protocol,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from ._types import SupportsDataclass
from .errors import PromptValidationError
from .tool_result import ToolResult

_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9_-]{1,64}$")


if TYPE_CHECKING:
    from ..events._types import EventBus

ParamsT_contra = TypeVar("ParamsT_contra", bound=SupportsDataclass, contravariant=True)
ResultT_co = TypeVar("ResultT_co", bound=SupportsDataclass)


@dataclass(slots=True, frozen=True)
class ToolContext:
    """Immutable container exposing prompt execution state to handlers."""

    prompt: Any
    rendered_prompt: Any
    adapter: Any
    session: Any
    event_bus: EventBus


class ToolHandler(Protocol[ParamsT_contra, ResultT_co]):
    """Callable protocol implemented by tool handlers."""

    def __call__(
        self, params: ParamsT_contra, *, context: ToolContext
    ) -> ToolResult[ResultT_co]: ...


@dataclass(slots=True)
class Tool[ParamsT: SupportsDataclass, ResultT: SupportsDataclass]:
    """Describe a callable tool exposed by prompt sections."""

    name: str
    description: str
    handler: ToolHandler[ParamsT, ResultT] | None
    params_type: type[Any] = field(init=False, repr=False)
    result_type: type[Any] = field(init=False, repr=False)
    accepts_overrides: bool = True

    def __post_init__(self) -> None:
        params_type = cast(
            type[SupportsDataclass] | None, getattr(self, "params_type", None)
        )
        result_type = cast(
            type[SupportsDataclass] | None, getattr(self, "result_type", None)
        )
        if params_type is None or result_type is None:
            origin = getattr(self, "__orig_class__", None)
            if origin is not None:  # pragma: no cover - interpreter-specific path
                args = get_args(origin)
                if len(args) == 2 and all(isinstance(arg, type) for arg in args):
                    params_type = cast(type[SupportsDataclass], args[0])
                    result_type = cast(type[SupportsDataclass], args[1])
        if params_type is None or result_type is None:
            raise PromptValidationError(
                "Tool must be instantiated with concrete type arguments.",
                placeholder="type_arguments",
            )

        if not is_dataclass(params_type):
            raise PromptValidationError(
                "Tool ParamsT must be a dataclass type.",
                dataclass_type=params_type,
                placeholder="ParamsT",
            )
        if not is_dataclass(result_type):
            raise PromptValidationError(
                "Tool ResultT must be a dataclass type.",
                dataclass_type=result_type,
                placeholder="ResultT",
            )

        self.params_type = params_type
        self.result_type = result_type

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

        handler = self.handler
        if handler is not None:
            self._validate_handler(handler, params_type, result_type)

        self.name = name_clean
        self.description = description_clean

    def _validate_handler(
        self,
        handler: object,
        params_type: type[SupportsDataclass],
        result_type: type[SupportsDataclass],
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

        if len(parameters) != 2:
            raise PromptValidationError(
                "Tool handler must accept exactly one positional argument and the keyword-only 'context' parameter.",
                dataclass_type=params_type,
                placeholder="handler",
            )

        parameter = parameters[0]
        context_parameter = parameters[1]

        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise PromptValidationError(
                "Tool handler parameter must be positional.",
                dataclass_type=params_type,
                placeholder="handler",
            )

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

        try:
            hints = get_type_hints(callable_handler, include_extras=True)
        except Exception:
            hints = {}

        annotation = hints.get(parameter.name, parameter.annotation)
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

        context_annotation = hints.get(
            context_parameter.name, context_parameter.annotation
        )
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
            if result_args_raw == (result_type,):
                return
        raise PromptValidationError(
            "Tool handler return annotation must be ToolResult[ResultT].",
            dataclass_type=params_type,
            placeholder="return",
        )

    @classmethod
    def __class_getitem__(
        cls, item: object
    ) -> type[Tool[SupportsDataclass, SupportsDataclass]]:
        if not isinstance(item, tuple):
            raise TypeError("Tool[...] expects two type arguments (ParamsT, ResultT).")
        typed_item = cast(tuple[Any, Any], item)
        try:
            params_candidate, result_candidate = typed_item
        except ValueError as error:
            raise TypeError(
                "Tool[...] expects two type arguments (ParamsT, ResultT)."
            ) from error
        if not isinstance(params_candidate, type) or not isinstance(
            result_candidate, type
        ):
            raise TypeError("Tool[...] type arguments must be types.")
        params_type = cast(type[SupportsDataclass], params_candidate)
        result_type = cast(type[SupportsDataclass], result_candidate)

        class _SpecializedTool(cls):
            def __post_init__(self) -> None:
                self.params_type = params_type
                self.result_type = result_type
                Tool.__post_init__(cast("Tool[Any, Any]", self))  # pyright: ignore[reportUnknownMemberType]

        _SpecializedTool.__name__ = cls.__name__
        _SpecializedTool.__qualname__ = cls.__qualname__
        _SpecializedTool.__module__ = cls.__module__
        return cast(
            "type[Tool[SupportsDataclass, SupportsDataclass]]",
            _SpecializedTool,
        )


__all__ = ["Tool", "ToolContext", "ToolHandler", "ToolResult"]
