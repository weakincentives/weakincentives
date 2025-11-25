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
from typing import TYPE_CHECKING, ClassVar, Self, TypeVar, cast

if TYPE_CHECKING:
    from .tool import Tool

from ._enabled_predicate import EnabledPredicate, normalize_enabled_predicate
from ._generic_params_specializer import GenericParamsSpecializer
from ._normalization import normalize_component_key
from ._types import SupportsDataclass, SupportsToolResult

SectionParamsT = TypeVar("SectionParamsT", bound=SupportsDataclass, covariant=True)


class Section(GenericParamsSpecializer[SectionParamsT], ABC):
    """Abstract building block for prompt content."""

    _generic_owner_name: ClassVar[str | None] = "Section"

    def __init__(
        self,
        *,
        title: str,
        key: str,
        default_params: SectionParamsT | None = None,
        children: Sequence[Section[SupportsDataclass]] | None = None,
        enabled: EnabledPredicate | None = None,
        tools: Sequence[object] | None = None,
        accepts_overrides: bool = True,
    ) -> None:
        super().__init__()
        params_candidate = getattr(self.__class__, "_params_type", None)
        candidate_type = (
            params_candidate if isinstance(params_candidate, type) else None
        )
        params_type = cast(type[SupportsDataclass] | None, candidate_type)

        self.params_type: type[SectionParamsT] | None = cast(
            type[SectionParamsT] | None, params_type
        )
        self.param_type: type[SectionParamsT] | None = self.params_type
        self.title = title
        self.key = self._normalize_key(key)
        self.default_params = default_params
        self.accepts_overrides = accepts_overrides

        if self.params_type is None and self.default_params is not None:
            raise TypeError("Section without parameters cannot define default_params.")

        normalized_children: list[Section[SupportsDataclass]] = []
        raw_children: Sequence[object] = cast(Sequence[object], children or ())
        for child in raw_children:
            if not isinstance(child, Section):
                raise TypeError("Section children must be Section instances.")
            normalized_children.append(cast(Section[SupportsDataclass], child))
        self.children = tuple(normalized_children)
        self._enabled: Callable[[SupportsDataclass | None], bool] | None = (
            normalize_enabled_predicate(enabled, params_type)
        )
        self._tools = self._normalize_tools(tools)

    def is_enabled(self, params: SupportsDataclass | None) -> bool:
        """Return True when the section should render for the given params."""

        if self._enabled is None:
            return True
        return bool(self._enabled(params))

    @abstractmethod
    def render(self, params: SupportsDataclass | None, depth: int, number: str) -> str:
        """Produce markdown output for the section at the supplied depth."""

    def placeholder_names(self) -> set[str]:
        """Return placeholder identifiers used by the section template."""

        return set()

    @abstractmethod
    def clone(self, **kwargs: object) -> Self:
        """Return a deep copy of the section and its children."""

    def tools(self) -> tuple[Tool[SupportsDataclass, SupportsToolResult], ...]:
        """Return the tools exposed by this section."""

        return self._tools

    def original_body_template(self) -> str | None:
        """Return the template text that participates in hashing, when available."""

        return None

    @staticmethod
    def _normalize_key(key: str) -> str:
        return normalize_component_key(key, owner="Section")

    @staticmethod
    def _normalize_tools(
        tools: Sequence[object] | None,
    ) -> tuple[Tool[SupportsDataclass, SupportsToolResult], ...]:
        if not tools:
            return ()

        from .tool import Tool

        normalized: list[Tool[SupportsDataclass, SupportsToolResult]] = []
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError("Section tools must be Tool instances.")
            normalized.append(cast(Tool[SupportsDataclass, SupportsToolResult], tool))
        return tuple(normalized)


__all__ = ["Section"]
