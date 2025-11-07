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
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Final, cast

if TYPE_CHECKING:
    from .tool import Tool

from ._types import SupportsDataclass

_SECTION_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9][a-z0-9._-]{0,63}$"
)


class Section[ParamsT: SupportsDataclass](ABC):
    """Abstract building block for prompt content."""

    _params_type: ClassVar[type[SupportsDataclass] | None] = None

    def __init__(
        self,
        *,
        title: str,
        key: str,
        default_params: ParamsT | None = None,
        children: Sequence[object] | None = None,
        enabled: Callable[[ParamsT], bool] | Callable[[], bool] | None = None,
        tools: Sequence[object] | None = None,
        accepts_overrides: bool = True,
    ) -> None:
        super().__init__()
        params_candidate = getattr(self.__class__, "_params_type", None)
        params_type = cast(
            type[ParamsT] | None,
            params_candidate if isinstance(params_candidate, type) else None,
        )

        self.params_type: type[ParamsT] | None = params_type
        self.param_type: type[ParamsT] | None = params_type
        self.title = title
        self.key = self._normalize_key(key)
        self.default_params = default_params
        self.accepts_overrides = accepts_overrides

        if self.params_type is None and self.default_params is not None:
            raise TypeError("Section without parameters cannot define default_params.")

        normalized_children: list[Section[SupportsDataclass]] = []
        for child in children or ():
            if not isinstance(child, Section):
                raise TypeError("Section children must be Section instances.")
            normalized_children.append(cast(Section[SupportsDataclass], child))
        self.children: tuple[Section[SupportsDataclass], ...] = tuple(
            normalized_children
        )
        self._enabled: Callable[[ParamsT | None], bool] | None = (
            self._normalize_enabled(enabled, params_type)
        )
        self._tools = self._normalize_tools(tools)

    def is_enabled(self, params: ParamsT | None) -> bool:
        """Return True when the section should render for the given params."""

        if self._enabled is None:
            return True
        return bool(self._enabled(params))

    @abstractmethod
    def render(self, params: ParamsT | None, depth: int) -> str:
        """Produce markdown output for the section at the supplied depth."""

    def placeholder_names(self) -> set[str]:
        """Return placeholder identifiers used by the section template."""

        return set()

    def tools(self) -> tuple[Tool[SupportsDataclass, SupportsDataclass], ...]:
        """Return the tools exposed by this section."""

        return self._tools

    def original_body_template(self) -> str | None:
        """Return the template text that participates in hashing, when available."""

        return None

    @classmethod
    def __class_getitem__(cls, item: object) -> type[Section[SupportsDataclass]]:
        params_type = cls._normalize_generic_argument(item)
        specialized = cast(
            "type[Section[SupportsDataclass]]",
            type(cls.__name__, (cls,), {}),
        )
        specialized.__name__ = cls.__name__
        specialized.__qualname__ = cls.__qualname__
        specialized.__module__ = cls.__module__
        specialized._params_type = cast(type[SupportsDataclass], params_type)
        return specialized

    @staticmethod
    def _normalize_key(key: str) -> str:
        normalized = key.strip().lower()
        if not normalized:
            raise ValueError("Section key must be a non-empty string.")
        if not _SECTION_KEY_PATTERN.match(normalized):
            raise ValueError("Section key must match ^[a-z0-9][a-z0-9._-]{0,63}$.")
        return normalized

    @staticmethod
    def _normalize_generic_argument(item: object) -> object:
        if isinstance(item, tuple):
            raise TypeError("Section[...] expects a single type argument.")
        return item

    @staticmethod
    def _normalize_tools(
        tools: Sequence[object] | None,
    ) -> tuple[Tool[SupportsDataclass, SupportsDataclass], ...]:
        if not tools:
            return ()

        from .tool import Tool

        normalized: list[Tool[SupportsDataclass, SupportsDataclass]] = []
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError("Section tools must be Tool instances.")
            normalized.append(cast(Tool[SupportsDataclass, SupportsDataclass], tool))
        return tuple(normalized)

    @staticmethod
    def _normalize_enabled(
        enabled: Callable[[ParamsT], bool] | Callable[[], bool] | None,
        params_type: type[SupportsDataclass] | None,
    ) -> Callable[[ParamsT | None], bool] | None:
        if enabled is None:
            return None
        if params_type is None and not _callable_requires_positional_argument(enabled):
            zero_arg = cast(Callable[[], bool], enabled)

            def _without_params(_: ParamsT | None) -> bool:
                return bool(zero_arg())

            return _without_params

        coerced = cast(Callable[[ParamsT], bool], enabled)

        def _with_params(value: ParamsT | None) -> bool:
            return bool(coerced(cast(ParamsT, value)))

        return _with_params


def _callable_requires_positional_argument(callback: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if (
            parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and parameter.default is inspect.Signature.empty
        ):
            return True
    return False


__all__ = ["Section"]
