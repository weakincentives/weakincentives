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
from typing import Any, ClassVar, cast

from ._normalization import normalize_component_key
from ._types import SupportsDataclass
from .section import Section


class ChaptersExpansionPolicy(StrEnum):
    """Strategies describing how adapters may open prompt chapters."""

    ALL_INCLUDED = "all_included"
    LLM_TOOL = "llm_tool"
    INTENT_CLASSIFIER = "intent_classifier"


@dataclass
class Chapter[ParamsT: SupportsDataclass]:
    """Container grouping sections under a shared visibility boundary."""

    key: str
    title: str
    description: str | None = None
    sections: tuple[Section[Any], ...] = ()
    default_params: ParamsT | None = None
    enabled: Callable[[ParamsT], bool] | Callable[[], bool] | None = None
    _enabled_callable: Callable[[ParamsT | None], bool] | None = field(
        init=False, repr=False, default=None
    )

    _params_type: ClassVar[type[SupportsDataclass] | None] = None

    def __post_init__(self) -> None:
        params_candidate = getattr(self.__class__, "_params_type", None)
        params_type = cast(
            type[ParamsT] | None,
            params_candidate if isinstance(params_candidate, type) else None,
        )
        self.key = self._normalize_key(self.key)

        normalized_sections: list[Section[SupportsDataclass]] = []
        for section in self.sections or ():
            if not isinstance(section, Section):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError("Chapter sections must be Section instances.")
            normalized_sections.append(cast(Section[SupportsDataclass], section))
        self.sections = tuple(normalized_sections)

        self._enabled_callable: Callable[[ParamsT | None], bool] | None = (
            self._normalize_enabled(self.enabled, params_type)
        )

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
    def params_type(self) -> type[ParamsT] | None:
        params_candidate = getattr(self.__class__, "_params_type", None)
        return cast(
            type[ParamsT] | None,
            params_candidate if isinstance(params_candidate, type) else None,
        )

    @property
    def param_type(self) -> type[ParamsT] | None:
        return self.params_type

    def is_enabled(self, params: ParamsT | None) -> bool:
        """Return True when the chapter should open for the provided params."""

        if self._enabled_callable is None:
            return True
        if params is None and self.param_type is not None:
            raise TypeError("Chapter parameters are required for enabled predicates.")
        return bool(self._enabled_callable(params))

    @classmethod
    def __class_getitem__(cls, item: object) -> type[Chapter[SupportsDataclass]]:
        params_type = cls._normalize_generic_argument(item)
        specialized = cast(
            "type[Chapter[SupportsDataclass]]",
            type(cls.__name__, (cls,), {}),
        )
        specialized.__name__ = cls.__name__
        specialized.__qualname__ = cls.__qualname__
        specialized.__module__ = cls.__module__
        specialized._params_type = cast(type[SupportsDataclass], params_type)
        return specialized

    @staticmethod
    def _normalize_key(key: str) -> str:
        return normalize_component_key(key, owner="Chapter")

    @staticmethod
    def _normalize_generic_argument(item: object) -> object:
        if isinstance(item, tuple):
            raise TypeError("Chapter[...] expects a single type argument.")
        return item

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


def _callable_requires_positional_argument(callback: Callable[..., object]) -> bool:
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
