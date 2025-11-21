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
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, TypeVar, cast

if TYPE_CHECKING:
    from .tool import Tool

from ._generic_params_specializer import GenericParamsSpecializer
from ._normalization import normalize_component_key
from ._types import SupportsDataclass, SupportsToolResult

SectionParamsT = TypeVar("SectionParamsT", bound=SupportsDataclass, covariant=True)

EnabledPredicate = Callable[[SupportsDataclass], bool] | Callable[[], bool]


@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class Section(GenericParamsSpecializer[SectionParamsT], ABC):
    """Abstract building block for prompt content."""

    title: str
    key: str
    default_params: SectionParamsT | None = None
    children: tuple["Section[SupportsDataclass]", ...] = ()
    enabled: EnabledPredicate | None = None
    accepts_overrides: bool = True

    _enabled: Callable[[SupportsDataclass | None], bool] | None = field(
        init=False, repr=False, default=None
    )
    _tools: tuple["Tool[SupportsDataclass, SupportsToolResult]", ...] = field(
        init=False, repr=False, default_factory=tuple
    )
    params_type: type[SectionParamsT] | None = field(
        init=False, repr=False, default=None
    )
    param_type: type[SectionParamsT] | None = field(
        init=False, repr=False, default=None
    )

    _generic_owner_name: ClassVar[str | None] = "Section"

    def __init__(
        self,
        *,
        title: str,
        key: str,
        default_params: SectionParamsT | None = None,
        children: Sequence["Section[SupportsDataclass]"] | None = None,
        enabled: EnabledPredicate | None = None,
        tools: Sequence[object] | None = None,
        accepts_overrides: bool = True,
    ) -> None:
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "default_params", default_params)
        object.__setattr__(self, "children", tuple(children or ()))
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "accepts_overrides", accepts_overrides)

        self.__post_init__(tools)

    def __post_init__(self, tools: Sequence[object] | None = None) -> None:
        params_candidate = getattr(self.__class__, "_params_type", None)
        candidate_type = (
            params_candidate if isinstance(params_candidate, type) else None
        )
        params_type = cast(type[SupportsDataclass] | None, candidate_type)

        object.__setattr__(
            self, "params_type", cast(type[SectionParamsT] | None, params_type)
        )
        object.__setattr__(
            self, "param_type", cast(type[SectionParamsT] | None, params_type)
        )
        object.__setattr__(self, "key", self._normalize_key(self.key))

        if params_type is None and self.default_params is not None:
            raise TypeError("Section without parameters cannot define default_params.")

        normalized_children: list[Section[SupportsDataclass]] = []
        raw_children: Sequence[object] = cast(Sequence[object], self.children or ())
        for child in raw_children:
            if not isinstance(child, Section):
                raise TypeError("Section children must be Section instances.")
            normalized_children.append(cast(Section[SupportsDataclass], child))
        object.__setattr__(self, "children", tuple(normalized_children))
        object.__setattr__(
            self,
            "_enabled",
            self._normalize_enabled(self.enabled, params_type),
        )
        object.__setattr__(self, "_tools", self._normalize_tools(tools))

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

    @staticmethod
    def _normalize_enabled(
        enabled: EnabledPredicate | None,
        params_type: type[SupportsDataclass] | None,
    ) -> Callable[[SupportsDataclass | None], bool] | None:
        if enabled is None:
            return None
        if params_type is None and not _callable_requires_positional_argument(enabled):
            zero_arg = cast(Callable[[], bool], enabled)

            def _without_params(_: SupportsDataclass | None) -> bool:
                return bool(zero_arg())

            return _without_params

        coerced = cast(Callable[[SupportsDataclass], bool], enabled)

        def _with_params(value: SupportsDataclass | None) -> bool:
            return bool(coerced(cast(SupportsDataclass, value)))

        return _with_params


def _callable_requires_positional_argument(callback: EnabledPredicate) -> bool:
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
