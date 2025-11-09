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

from ._predicate_utils import normalize_enabled_predicate
from ._normalization import normalize_component_key
from ._types import SupportsDataclass


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
        return normalize_component_key(key, owner="Section")

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
        return normalize_enabled_predicate(enabled, params_type)


__all__ = ["Section"]
