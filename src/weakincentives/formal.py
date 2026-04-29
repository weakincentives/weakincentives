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

"""Lightweight metadata for formal-spec annotations.

The full TLA+ verification toolchain is out of scope; this module only
captures the metadata an extra (or a downstream verifier) can consume.
Decorate a function or class with :func:`formal_spec` and read the
attached :class:`FormalSpec` with :func:`get_formal_spec`. A registry of
all decorated targets is available for tooling that wants to audit which
parts of the codebase are covered.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeVar, final

from weakincentives.core import WinkError

T = TypeVar("T")

__all__ = [
    "FormalError",
    "FormalSpec",
    "all_formal_specs",
    "formal_spec",
    "get_formal_spec",
    "reset_registry",
]

_ATTRIBUTE = "__wink_formal_spec__"


class FormalError(WinkError):
    """Raised on misuse of the formal-spec decorator."""


@final
@dataclass(frozen=True, slots=True)
class FormalSpec:
    """Metadata attached to a formal-spec target."""

    name: str
    properties: tuple[str, ...] = ()
    description: str = ""
    target_qualname: str = ""


_REGISTRY: dict[str, FormalSpec] = {}


def formal_spec(
    *,
    name: str,
    properties: Iterable[str] = (),
    description: str | None = None,
) -> Callable[[T], T]:
    """Attach :class:`FormalSpec` metadata to a function or class.

    The decorator records the target in a module-level registry so tooling
    can iterate every annotated symbol via :func:`all_formal_specs`.
    """
    cleaned = tuple(properties)

    def decorate(target: T) -> T:
        if not callable(target) and not isinstance(target, type):
            raise FormalError(
                f"@formal_spec target must be a callable or class, "
                f"got {type(target).__name__}"
            )
        if name in _REGISTRY:
            existing = _REGISTRY[name]
            raise FormalError(
                f"duplicate formal_spec name {name!r} "
                f"(already attached to {existing.target_qualname})"
            )
        doc = getattr(target, "__doc__", None) or ""
        spec = FormalSpec(
            name=name,
            properties=cleaned,
            description=(description or doc).strip(),
            target_qualname=getattr(target, "__qualname__", repr(target)),
        )
        setattr(target, _ATTRIBUTE, spec)
        _REGISTRY[name] = spec
        return target

    return decorate


def get_formal_spec(target: object) -> FormalSpec | None:
    """Return the :class:`FormalSpec` attached to ``target``, if any."""
    candidate = getattr(target, _ATTRIBUTE, None)
    return candidate if isinstance(candidate, FormalSpec) else None


def all_formal_specs() -> tuple[FormalSpec, ...]:
    """Snapshot every registered :class:`FormalSpec`."""
    return tuple(_REGISTRY.values())


def reset_registry() -> None:
    """Clear the global registry.

    Primarily useful in tests that decorate throwaway targets and want
    each test to start from an empty registry.
    """
    _REGISTRY.clear()
