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

from typing import ClassVar, get_args, get_origin

from ._types import SupportsDataclass


class GenericParamsSpecializer[ParamsT: SupportsDataclass]:
    """Mixin providing ``ParamsT`` specialization for prompt components."""

    _params_type: ClassVar[type[SupportsDataclass] | None] = None

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._params_type = cls._extract_params_type()

    @classmethod
    def _extract_params_type(cls) -> type[SupportsDataclass] | None:
        for base in getattr(cls, "__orig_bases__", ()):  # type: ignore[attr-defined]
            origin = get_origin(base) or base
            if not isinstance(origin, type) or not issubclass(
                origin, GenericParamsSpecializer
            ):
                continue

            args = get_args(base)
            if not args:
                continue

            candidate = args[0]
            if isinstance(candidate, type):
                return candidate

        for base in cls.__mro__[1:]:
            if isinstance(base, type) and issubclass(base, GenericParamsSpecializer):
                parent_params_type = getattr(base, "_params_type", None)
                if isinstance(parent_params_type, type):
                    return parent_params_type

        return None


__all__ = ["GenericParamsSpecializer"]
