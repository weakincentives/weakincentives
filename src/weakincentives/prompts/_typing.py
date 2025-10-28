from __future__ import annotations

from dataclasses import Field
from typing import Any, ClassVar, Protocol, runtime_checkable


@runtime_checkable
class SupportsDataclass(Protocol):
    """Protocol satisfied by dataclass types and instances."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


__all__ = ["SupportsDataclass"]
