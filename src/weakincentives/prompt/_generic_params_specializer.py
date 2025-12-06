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

from typing import ClassVar, Generic, TypeVar, cast

from ._types import SupportsDataclass

ParamsT = TypeVar("ParamsT", bound=SupportsDataclass, covariant=True)

SelfClass = TypeVar(
    "SelfClass",
    bound="GenericParamsSpecializer[SupportsDataclass]",
    covariant=True,
)


class GenericParamsSpecializer(Generic[ParamsT]):
    """Mixin providing ``ParamsT`` specialization for prompt components."""

    _params_type: ClassVar[type[SupportsDataclass] | None] = None
    _generic_owner_name: ClassVar[str | None] = None

    @classmethod
    def __class_getitem__(cls: type[SelfClass], item: object) -> type[SelfClass]:
        params_type = cls._normalize_generic_argument(item)
        specialized = cast(
            "type[SelfClass]",
            type(cls.__name__, (cls,), {}),
        )
        specialized.__name__ = cls.__name__
        specialized.__qualname__ = cls.__qualname__
        specialized.__module__ = cls.__module__
        specialized._params_type = cast("type[SupportsDataclass]", params_type)
        return specialized

    @classmethod
    def _normalize_generic_argument(cls, item: object) -> object:
        if isinstance(item, tuple):
            raise TypeError(f"{cls._owner_name()}[...] expects a single type argument.")
        return item

    @classmethod
    def _owner_name(cls) -> str:
        owner_name = getattr(cls, "_generic_owner_name", None)
        if isinstance(owner_name, str) and owner_name:
            return owner_name
        name = getattr(cls, "__name__", None)
        if isinstance(name, str) and name:
            return name
        return "Component"


__all__ = ["GenericParamsSpecializer"]
