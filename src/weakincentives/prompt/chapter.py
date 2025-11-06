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

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Final, cast

from ._types import SupportsDataclass
from .section import Section

_CHAPTER_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9][a-z0-9._-]{0,63}$"
)


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
    enabled: Callable[[ParamsT], bool] | None = None

    _params_type: ClassVar[type[SupportsDataclass] | None] = None

    def __post_init__(self) -> None:
        params_type = cast(
            type[ParamsT] | None, getattr(self.__class__, "_params_type", None)
        )
        if params_type is None:
            raise TypeError(
                "Chapter must be instantiated with a concrete ParamsT type."
            )

        self.key = self._normalize_key(self.key)

        normalized_sections: list[Section[SupportsDataclass]] = []
        for section in self.sections or ():
            if not isinstance(section, Section):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError("Chapter sections must be Section instances.")
            normalized_sections.append(cast(Section[SupportsDataclass], section))
        self.sections = tuple(normalized_sections)

        if self.default_params is not None and not isinstance(
            self.default_params, params_type
        ):
            raise TypeError(
                "Chapter default_params must match the declared ParamsT type."
            )

    @property
    def params_type(self) -> type[ParamsT]:
        return cast(type[ParamsT], self.__class__._params_type)

    @property
    def param_type(self) -> type[ParamsT]:
        return self.params_type

    def is_enabled(self, params: ParamsT | None) -> bool:
        """Return True when the chapter should open for the provided params."""

        if self.enabled is None:
            return True
        if params is None:
            raise TypeError("Chapter parameters are required for enabled predicates.")
        return bool(self.enabled(params))

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
        normalized = key.strip().lower()
        if not normalized:
            raise ValueError("Chapter key must be a non-empty string.")
        if not _CHAPTER_KEY_PATTERN.match(normalized):
            raise ValueError("Chapter key must match ^[a-z0-9][a-z0-9._-]{0,63}$.")
        return normalized

    @staticmethod
    def _normalize_generic_argument(item: object) -> object:
        if isinstance(item, tuple):
            raise TypeError("Chapter[...] expects a single type argument.")
        return item


__all__ = ["Chapter", "ChaptersExpansionPolicy"]
