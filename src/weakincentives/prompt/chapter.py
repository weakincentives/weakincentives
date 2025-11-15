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

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, TypeVar, cast

from ._generic_params_specializer import GenericParamsSpecializer
from ._normalization import normalize_component_key
from ._types import SupportsDataclass
from .section import Section


class ChaptersExpansionPolicy(StrEnum):
    """Strategies describing how adapters may open prompt chapters."""

    ALL_INCLUDED = "all_included"
    INTENT_CLASSIFIER = "intent_classifier"


ChapterParamsT = TypeVar("ChapterParamsT", bound=SupportsDataclass, covariant=True)
EnabledPredicate = Callable[[SupportsDataclass], bool] | Callable[[], bool]


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

        self._enabled_callable = self._normalize_enabled(self.enabled, params_type)

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


__all__ = ["Chapter", "ChaptersExpansionPolicy"]
