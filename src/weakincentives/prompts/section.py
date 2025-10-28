from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast

if TYPE_CHECKING:
    from .tool import Tool


class Section[ParamsT](ABC):
    """Abstract building block for prompt content."""

    _params_type: ClassVar[type[Any] | None] = None

    def __init__(
        self,
        *,
        title: str,
        defaults: ParamsT | None = None,
        children: Sequence[object] | None = None,
        enabled: Callable[[ParamsT], bool] | None = None,
        tools: Sequence[object] | None = None,
    ) -> None:
        params_type = cast(
            type[ParamsT] | None, getattr(self.__class__, "_params_type", None)
        )
        if params_type is None:
            raise TypeError(
                "Section must be instantiated with a concrete ParamsT type."
            )

        self.params_type: type[ParamsT] = params_type
        self.params: type[ParamsT] = params_type
        self.title = title
        self.defaults = defaults

        normalized_children: list[Section[Any]] = []
        for child in children or ():
            if not isinstance(child, Section):
                raise TypeError("Section children must be Section instances.")
            normalized_children.append(cast(Section[Any], child))
        self.children: tuple[Section[Any], ...] = tuple(normalized_children)
        self._enabled = enabled
        self._tools = self._normalize_tools(tools)

    def is_enabled(self, params: ParamsT) -> bool:
        """Return True when the section should render for the given params."""

        if self._enabled is None:
            return True
        return bool(self._enabled(params))

    @abstractmethod
    def render(self, params: ParamsT, depth: int) -> str:
        """Produce markdown output for the section at the supplied depth."""

    def placeholder_names(self) -> set[str]:
        """Return placeholder identifiers used by the section template."""

        return set()

    def tools(self) -> tuple[Tool[Any, Any], ...]:
        """Return the tools exposed by this section."""

        return self._tools

    @classmethod
    def __class_getitem__(cls, item: object) -> type[Section[Any]]:
        params_type = cls._normalize_generic_argument(item)

        class _SpecializedSection(cls):  # type: ignore[misc]
            pass

        _SpecializedSection.__name__ = cls.__name__
        _SpecializedSection.__qualname__ = cls.__qualname__
        _SpecializedSection.__module__ = cls.__module__
        _SpecializedSection._params_type = cast(type[Any], params_type)
        return _SpecializedSection

    @staticmethod
    def _normalize_generic_argument(item: object) -> object:
        if isinstance(item, tuple):
            raise TypeError("Section[...] expects a single type argument.")
        return item

    @staticmethod
    def _normalize_tools(
        tools: Sequence[object] | None,
    ) -> tuple[Tool[Any, Any], ...]:
        if not tools:
            return ()

        from .tool import Tool

        normalized: list[Tool[Any, Any]] = []
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError("Section tools must be Tool instances.")
            normalized.append(cast(Tool[Any, Any], tool))
        return tuple(normalized)


__all__ = ["Section"]
