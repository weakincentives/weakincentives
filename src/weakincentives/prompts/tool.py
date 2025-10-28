from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field, is_dataclass
from typing import Annotated, Any, Generic, TypeVar, cast, get_args, get_origin, get_type_hints

from .errors import PromptValidationError

_NAME_PATTERN = re.compile(r"^[a-z0-9_]{1,64}$")

ParamsT = TypeVar("ParamsT")
ResultT = TypeVar("ResultT")
ResultPayloadT = TypeVar("ResultPayloadT")


@dataclass(slots=True)
class ToolResult(Generic[ResultPayloadT]):
    """Structured response emitted by a tool handler."""

    message: str
    payload: ResultPayloadT


@dataclass(slots=True)
class Tool(Generic[ParamsT, ResultT]):
    """Describe a callable tool exposed by prompt sections."""

    name: str
    description: str
    handler: Callable[[ParamsT], ToolResult[ResultT]] | None
    params_type: type[Any] = field(init=False, repr=False)
    result_type: type[Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        params_type = cast(type[Any] | None, getattr(self, "params_type", None))
        result_type = cast(type[Any] | None, getattr(self, "result_type", None))
        if params_type is None or result_type is None:
            origin = getattr(self, "__orig_class__", None)
            if origin is not None:  # pragma: no cover - defensive fallback
                args = get_args(origin)
                if len(args) == 2 and all(isinstance(arg, type) for arg in args):
                    params_type = cast(type[Any], args[0])
                    result_type = cast(type[Any], args[1])
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
                "Tool name must be non-empty lowercase ASCII up to 64 characters.",
                dataclass_type=params_type,
                placeholder=stripped_name,
            )
        if len(name_clean) > 64 or not _NAME_PATTERN.fullmatch(name_clean):
            raise PromptValidationError(
                "Tool name must use lowercase ASCII letters, digits, or underscores.",
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
            description_clean.encode("ascii")
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
        handler: Callable[[ParamsT], ToolResult[ResultT]],
        params_type: type[Any],
        result_type: type[Any],
    ) -> None:
        if not callable(handler):
            raise PromptValidationError(
                "Tool handler must be callable.",
                dataclass_type=params_type,
                placeholder="handler",
            )

        signature = inspect.signature(handler)
        parameters = list(signature.parameters.values())

        if len(parameters) != 1:
            raise PromptValidationError(
                "Tool handler must accept exactly one argument.",
                dataclass_type=params_type,
                placeholder="handler",
            )

        parameter = parameters[0]
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise PromptValidationError(
                "Tool handler parameter must be positional.",
                dataclass_type=params_type,
                placeholder="handler",
            )

        try:
            hints = get_type_hints(handler, include_extras=True)
        except Exception:  # pragma: no cover - fallback for invalid hints
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
    ) -> type["Tool[Any, Any]"]:
        if not isinstance(item, tuple):
            raise TypeError("Tool[...] expects two type arguments (ParamsT, ResultT).")
        typed_item = cast(tuple[Any, Any], item)
        try:
            params_candidate, result_candidate = typed_item
        except ValueError as error:
            raise TypeError("Tool[...] expects two type arguments (ParamsT, ResultT).") from error
        if not isinstance(params_candidate, type) or not isinstance(result_candidate, type):
            raise TypeError("Tool[...] type arguments must be types.")
        params_type = cast(type[Any], params_candidate)
        result_type = cast(type[Any], result_candidate)

        class _SpecializedTool(cls):  # type: ignore[misc]
            def __post_init__(self) -> None:  # type: ignore[override]
                self.params_type = params_type
                self.result_type = result_type
                super().__post_init__()

        _SpecializedTool.__name__ = cls.__name__
        _SpecializedTool.__qualname__ = cls.__qualname__
        _SpecializedTool.__module__ = cls.__module__
        return _SpecializedTool

__all__ = ["Tool", "ToolResult"]
