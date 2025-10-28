"""Internal typing helpers for the prompts package."""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable


@runtime_checkable
class SupportsDataclass(Protocol):
    """Protocol satisfied by dataclass types and instances."""

    __dataclass_fields__: ClassVar[dict[str, Any]]


__all__ = ["SupportsDataclass"]
