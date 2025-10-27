from __future__ import annotations

import inspect
import re
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, is_dataclass
from string import Template
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from .errors import PromptRenderError, PromptValidationError
from .section import Section

_NAME_PATTERN = re.compile(r"^[a-z0-9_]{1,64}$")


@dataclass(slots=True)
class ToolResult[ResultPayloadT]:
    """Structured response emitted by a tool handler."""

    message: str
    payload: ResultPayloadT


@dataclass(slots=True)
class Tool[ParamsT, ResultT]:
    """Describe a callable tool exposed by prompt sections."""

    name: str
    description: str
    handler: Callable[[ParamsT], ToolResult[ResultT]] | None
    params_type: type[ParamsT] = field(init=False, repr=False)
    result_type: type[ResultT] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        params_type = getattr(self, "params_type", None)
        result_type = getattr(self, "result_type", None)
        if params_type is None or result_type is None:
            raise PromptValidationError(
                "Tool must be instantiated with concrete type arguments.",
                placeholder="type_arguments",
            )

        if not isinstance(params_type, type) or not is_dataclass(params_type):
            raise PromptValidationError(
                "Tool ParamsT must be a dataclass type.",
                dataclass_type=params_type
                if isinstance(params_type, type)
                else type(params_type),
                placeholder="ParamsT",
            )
        if not isinstance(result_type, type) or not is_dataclass(result_type):
            raise PromptValidationError(
                "Tool ResultT must be a dataclass type.",
                dataclass_type=result_type
                if isinstance(result_type, type)
                else type(result_type),
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
        params_type: type[ParamsT],
        result_type: type[ResultT],
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
        if annotation is inspect._empty:
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
        if return_annotation is inspect._empty:
            raise PromptValidationError(
                "Tool handler must annotate its return value with ToolResult[ResultT].",
                dataclass_type=params_type,
                placeholder="return",
            )
        if get_origin(return_annotation) is Annotated:
            return_annotation = get_args(return_annotation)[0]

        origin = get_origin(return_annotation)
        if origin is ToolResult:
            result_args = get_args(return_annotation)
            if len(result_args) == 1 and result_args[0] is result_type:
                return
        raise PromptValidationError(
            "Tool handler return annotation must be ToolResult[ResultT].",
            dataclass_type=params_type,
            placeholder="return",
        )

    @classmethod
    def __class_getitem__(cls, item: object) -> type[Tool[Any, Any]]:
        params_type, result_type = cls._normalize_generic_arguments(item)

        class _SpecializedTool(cls):  # type: ignore[misc]
            def __post_init__(self) -> None:  # type: ignore[override]
                self.params_type = params_type
                self.result_type = result_type
                super().__post_init__()

        _SpecializedTool.__name__ = cls.__name__
        _SpecializedTool.__qualname__ = cls.__qualname__
        _SpecializedTool.__module__ = cls.__module__
        return _SpecializedTool

    @staticmethod
    def _normalize_generic_arguments(item: object) -> tuple[type[Any], type[Any]]:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Tool[...] expects two type arguments (ParamsT, ResultT).")
        params_type, result_type = item
        if not isinstance(params_type, type) or not isinstance(result_type, type):
            raise TypeError("Tool[...] type arguments must be types.")
        return params_type, result_type


class ToolsSection[ParamsT](Section[ParamsT]):
    """Section that documents tools without emitting per-tool markdown."""

    def __init__(
        self,
        *,
        title: str,
        tools: Sequence[Tool[Any, Any]],
        params: type[ParamsT],
        description: str | None = None,
        defaults: ParamsT | None = None,
        children: Sequence[Section[Any]] | None = None,
        enabled: Callable[[ParamsT], bool] | None = None,
    ) -> None:
        super().__init__(
            title=title,
            params=params,
            defaults=defaults,
            children=children,
            enabled=enabled,
        )
        if not tools:
            raise ValueError("ToolsSection requires at least one Tool instance.")

        normalized_tools: list[Tool[Any, Any]] = []
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError("ToolsSection tools must be Tool instances.")
            normalized_tools.append(tool)

        self._tools = tuple(normalized_tools)

        description_text = None
        if description is not None:
            stripped_description = textwrap.dedent(description).strip()
            if stripped_description:
                description_text = stripped_description
        self._description_template = (
            Template(description_text) if description_text is not None else None
        )

    def render(self, params: ParamsT, depth: int) -> str:
        heading_level = "#" * (depth + 2)
        heading = f"{heading_level} {self.title.strip()}"

        if self._description_template is None:
            return heading

        context = vars(params)
        missing = self.placeholder_names() - set(context)
        if missing:
            placeholder = sorted(missing)[0]
            raise PromptRenderError(
                "Missing placeholder during render.",
                placeholder=placeholder,
            )

        try:
            description = self._description_template.safe_substitute(context)
        except (
            KeyError
        ) as error:  # pragma: no cover - defensive parity with TextSection
            placeholder = str(error.args[0]) if error.args else None
            raise PromptRenderError(
                "Missing placeholder during render.",
                placeholder=placeholder,
            ) from error

        if description:
            return f"{heading}\n\n{description.strip()}"
        return heading

    def placeholder_names(self) -> set[str]:
        template = self._description_template
        if template is None:
            return set()
        placeholders: set[str] = set()
        for match in template.pattern.finditer(template.template):
            named = match.group("named")
            if named:
                placeholders.add(named)
                continue
            braced = match.group("braced")
            if braced:
                placeholders.add(braced)
        return placeholders

    def tools(self) -> tuple[Tool[Any, Any], ...]:
        return self._tools


__all__ = ["Tool", "ToolResult", "ToolsSection"]
