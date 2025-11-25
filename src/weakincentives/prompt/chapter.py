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

"""Chapter primitives controlling coarse-grained prompt visibility."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Self, TypeVar, cast

from ..serde import clone as clone_dataclass
from ._enabled_predicate import EnabledPredicate, normalize_enabled_predicate
from ._generic_params_specializer import GenericParamsSpecializer
from ._normalization import normalize_component_key
from ._types import SupportsDataclass
from .section import Section


class ChaptersExpansionPolicy(StrEnum):
    """Strategies describing how adapters may open prompt chapters."""

    ALL_INCLUDED = "all_included"
    INTENT_CLASSIFIER = "intent_classifier"


ChapterParamsT = TypeVar("ChapterParamsT", bound=SupportsDataclass, covariant=True)


@dataclass
class Chapter(GenericParamsSpecializer[ChapterParamsT]):
    """Container grouping sections under a shared visibility boundary."""

    key: str
    title: str
    description: str | None = None
    sections: tuple[Section[SupportsDataclass], ...] = ()
    default_params: ChapterParamsT | None = None
    enabled: EnabledPredicate | None = None
    _enabled_callable: Callable[[SupportsDataclass | None], bool] | None = field(
        init=False, repr=False, default=None
    )

    _generic_owner_name: ClassVar[str | None] = "Chapter"

    def __post_init__(self) -> None:
        params_candidate = getattr(self.__class__, "_params_type", None)
        candidate_type = (
            params_candidate if isinstance(params_candidate, type) else None
        )
        params_type = cast(type[SupportsDataclass] | None, candidate_type)
        self.key = self._normalize_key(self.key)

        self.sections = tuple(self.sections or ())

        self._enabled_callable = normalize_enabled_predicate(self.enabled, params_type)

        if params_type is None:
            if self.default_params is not None:
                raise TypeError(
                    "Chapter without parameters cannot define default_params."
                )
        elif self.default_params is not None and not isinstance(
            self.default_params, params_type
        ):
            raise TypeError(
                "Chapter default_params must match the declared ParamsT type."
            )

    @property
    def params_type(self) -> type[ChapterParamsT] | None:
        params_candidate = getattr(self.__class__, "_params_type", None)
        candidate_type = (
            params_candidate if isinstance(params_candidate, type) else None
        )
        return cast(type[ChapterParamsT] | None, candidate_type)

    @property
    def param_type(self) -> type[ChapterParamsT] | None:
        return self.params_type

    def is_enabled(self, params: SupportsDataclass | None) -> bool:
        """Return True when the chapter should open for the provided params."""

        if self._enabled_callable is None:
            return True
        if params is None and self.param_type is not None:
            raise TypeError("Chapter parameters are required for enabled predicates.")
        return bool(self._enabled_callable(params))

    @staticmethod
    def _normalize_key(key: str) -> str:
        return normalize_component_key(key, owner="Chapter")

    def clone(self, **kwargs: object) -> Self:
        cloned_sections = tuple(section.clone(**kwargs) for section in self.sections)
        cloned_default = (
            clone_dataclass(self.default_params)
            if self.default_params is not None
            else None
        )

        cls: type[Any] = type(self)
        clone = cls(
            key=self.key,
            title=self.title,
            description=self.description,
            sections=cloned_sections,
            default_params=cloned_default,
            enabled=self.enabled,
        )
        return cast(Self, clone)


__all__ = ["Chapter", "ChaptersExpansionPolicy"]
