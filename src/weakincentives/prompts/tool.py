from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, is_dataclass
import re
from typing import Generic, TypeVar

from .errors import PromptValidationError

_ParamsT = TypeVar("_ParamsT")
_HandlerReturnT = TypeVar("_HandlerReturnT")

_NAME_PATTERN = re.compile(r"^[a-z0-9_]{1,64}$")


@dataclass(slots=True)
class Tool(Generic[_ParamsT]):
    """Describe a callable tool exposed by prompt sections."""

    name: str
    description: str
    params: type[_ParamsT]
    handler: (
        Callable[[_ParamsT], _HandlerReturnT]
        | Callable[[_ParamsT], Awaitable[_HandlerReturnT]]
        | None
    ) = None

    def __post_init__(self) -> None:
        params_type = self.params
        if not isinstance(params_type, type) or not is_dataclass(params_type):
            raise PromptValidationError(
                "Tool params must be a dataclass type.",
                dataclass_type=params_type
                if isinstance(params_type, type)
                else type(params_type),
                placeholder="params",
            )

        raw_name = self.name
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

        self.name = name_clean
        self.description = description_clean


__all__ = ["Tool"]
