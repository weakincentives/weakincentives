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

from typing import ClassVar, Self, cast

from ._types import SupportsDataclass


class GenericParamsSpecializer[ParamsT: SupportsDataclass]:
    """Mixin providing ``ParamsT`` specialization for prompt components.

    This mixin automatically infers ``_params_type`` from generic type arguments.
    When creating a subclass like ``MySection(Section[MyParams])``, the
    ``_params_type`` class variable is automatically set to ``MyParams``.

    The inference works through two mechanisms:
    1. ``__class_getitem__`` sets ``_params_type`` on dynamic specialized classes
    2. ``__init_subclass__`` propagates ``_params_type`` to user-defined subclasses

    This means users never need to explicitly set ``_params_type``::

        @dataclass
        class MyParams:
            value: str

        # _params_type is automatically inferred - no need to set it manually
        class MySection(MarkdownSection[MyParams]):
            pass
    """

    _params_type: ClassVar[type[SupportsDataclass] | None] = None
    _generic_owner_name: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Propagate _params_type to subclasses automatically.

        When a subclass is created, this hook ensures ``_params_type`` is set
        by inheriting from the first base class in the MRO that has it defined.
        This covers dynamic specialized classes created by ``__class_getitem__``.

        If a subclass explicitly defines ``_params_type`` in its own ``__dict__``,
        that value is preserved and not overwritten.
        """
        super().__init_subclass__(**kwargs)

        # Check if subclass explicitly defines _params_type in its own __dict__
        if "_params_type" in cls.__dict__:
            return

        # Inherit _params_type from MRO (covers dynamic base classes from __class_getitem__)
        for base in cls.__mro__[1:]:
            base_params = getattr(base, "_params_type", None)
            if base_params is not None:
                cls._params_type = base_params
                return

    @classmethod
    def __class_getitem__(cls, item: object) -> type[Self]:
        params_type = cls._normalize_generic_argument(item)
        specialized = cast(
            "type[Self]",
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
