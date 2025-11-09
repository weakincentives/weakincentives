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

"""Utilities for specializing prompt primitives with parameter types."""

from __future__ import annotations

from typing import ClassVar, TypeVar, cast

from ._types import SupportsDataclass

SpecializerT = TypeVar(
    "SpecializerT",
    bound="GenericParamsSpecializer",
)


class GenericParamsSpecializer:
    """Mixin adding ``ParamsT`` specialization support."""

    _params_type: ClassVar[type[SupportsDataclass] | None] = None

    @classmethod
    def __class_getitem__(cls: type[SpecializerT], item: object) -> type[SpecializerT]:
        params_type = cls._normalize_generic_argument(item)
        specialized = cast(
            type[SpecializerT],
            type(cls.__name__, (cls,), {}),
        )
        specialized.__name__ = cls.__name__
        specialized.__qualname__ = cls.__qualname__
        specialized.__module__ = cls.__module__
        specialized._params_type = cast(type[SupportsDataclass], params_type)
        return specialized

    @classmethod
    def _normalize_generic_argument(cls, item: object) -> object:
        if isinstance(item, tuple):
            raise TypeError(f"{cls.__name__}[...] expects a single type argument.")
        return item


__all__ = ["GenericParamsSpecializer"]
