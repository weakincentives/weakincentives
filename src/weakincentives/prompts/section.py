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

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from .tool import Tool

from ._types import SupportsDataclass


class Section[ParamsT: SupportsDataclass](ABC):
    """Abstract building block for prompt content."""

    _params_type: ClassVar[type[SupportsDataclass] | None] = None

    def __init__(
        self,
        *,
        title: str,
        defaults: ParamsT | None = None,
        children: Sequence[object] | None = None,
        enabled: Callable[[ParamsT], bool] | None = None,
        tools: Sequence[object] | None = None,
    ) -> None:
        params_type = cast(
            type[ParamsT] | None, getattr(self.__class__, "_params_type", None)
        )
        if params_type is None:
            raise TypeError(
                "Section must be instantiated with a concrete ParamsT type."
            )

        self.params_type: type[ParamsT] = params_type
        self.params: type[ParamsT] = params_type
        self.title = title
        self.defaults = defaults

        normalized_children: list[Section[SupportsDataclass]] = []
        for child in children or ():
            if not isinstance(child, Section):
                raise TypeError("Section children must be Section instances.")
            normalized_children.append(cast(Section[SupportsDataclass], child))
        self.children: tuple[Section[SupportsDataclass], ...] = tuple(
            normalized_children
        )
        self._enabled = enabled
        self._tools = self._normalize_tools(tools)

    def is_enabled(self, params: ParamsT) -> bool:
        """Return True when the section should render for the given params."""

        if self._enabled is None:
            return True
        return bool(self._enabled(params))

    @abstractmethod
    def render(self, params: ParamsT, depth: int) -> str:
        """Produce markdown output for the section at the supplied depth."""

    def placeholder_names(self) -> set[str]:
        """Return placeholder identifiers used by the section template."""

        return set()

    def tools(self) -> tuple[Tool[SupportsDataclass, SupportsDataclass], ...]:
        """Return the tools exposed by this section."""

        return self._tools

    @classmethod
    def __class_getitem__(cls, item: object) -> type[Section[SupportsDataclass]]:
        params_type = cls._normalize_generic_argument(item)

        class _SpecializedSection(cls):  # type: ignore[misc]
            pass

        _SpecializedSection.__name__ = cls.__name__
        _SpecializedSection.__qualname__ = cls.__qualname__
        _SpecializedSection.__module__ = cls.__module__
        _SpecializedSection._params_type = cast(type[SupportsDataclass], params_type)
        return _SpecializedSection  # type: ignore[return-value]

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


__all__ = ["Section"]
