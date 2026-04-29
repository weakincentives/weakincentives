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

"""Persistent snapshots and the :class:`Snapshotable` protocol.

Snapshots round-trip session state through JSON. Slice values must be frozen
dataclasses whose fields are JSON-compatible (primitives, ``None``, tuples,
or nested frozen dataclasses). The :class:`TypeRegistry` is responsible for
mapping ``__type__`` strings to concrete classes.
"""

# ``capture`` and ``restore`` reach into ``Session``'s underscore attributes
# by design; they are co-designed within the spine package.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    cast,
    final,
    runtime_checkable,
)

from .errors import SnapshotError

if TYPE_CHECKING:
    from .session import Session

__all__ = [
    "Snapshot",
    "Snapshotable",
    "TypeRegistry",
    "capture",
    "restore",
]

_SCHEMA_VERSION = "1"


@runtime_checkable
class Snapshotable[StateT](Protocol):
    """Resources that can capture and restore opaque state."""

    def snapshot(self) -> StateT: ...

    def restore(self, state: StateT) -> None: ...


class TypeRegistry:
    """Maps stable string identifiers to frozen-dataclass types."""

    def __init__(self) -> None:
        super().__init__()
        self._by_id: dict[str, type[object]] = {}
        self._by_class: dict[type[object], str] = {}

    def register(self, cls: type[object]) -> None:
        if not is_dataclass(cls):
            raise SnapshotError(f"{cls.__name__} is not a dataclass; cannot register")
        identifier = f"{cls.__module__}.{cls.__qualname__}"
        existing = self._by_id.get(identifier)
        if existing is not None and existing is not cls:
            raise SnapshotError(f"identifier collision for {identifier!r}")
        self._by_id[identifier] = cls
        self._by_class[cls] = identifier

    def identifier_for(self, cls: type[object]) -> str:
        try:
            return self._by_class[cls]
        except KeyError as error:
            raise SnapshotError(f"type {cls.__qualname__} is not registered") from error

    def resolve(self, identifier: str) -> type[object]:
        try:
            return self._by_id[identifier]
        except KeyError as error:
            raise SnapshotError(f"unknown type identifier: {identifier!r}") from error


@final
@dataclass(frozen=True, slots=True)
class Snapshot:
    """Immutable point-in-time capture of a session's slices."""

    slices: Mapping[type[object], tuple[object, ...]]
    created_at: datetime
    schema_version: str = _SCHEMA_VERSION

    def to_json(self, *, registry: TypeRegistry) -> str:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "created_at": self.created_at.isoformat(),
            "slices": [
                {
                    "type": registry.identifier_for(slice_type),
                    "items": [_encode(item, registry) for item in items],
                }
                for slice_type, items in self.slices.items()
            ],
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str, *, registry: TypeRegistry) -> Snapshot:
        try:
            decoded: object = json.loads(raw)
        except json.JSONDecodeError as error:
            raise SnapshotError("invalid snapshot JSON") from error
        if not isinstance(decoded, dict):
            raise SnapshotError("snapshot payload must be an object")
        payload = cast("dict[str, Any]", decoded)
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise SnapshotError(
                f"unsupported snapshot schema: {payload.get('schema_version')!r}"
            )
        try:
            created_at = datetime.fromisoformat(cast("str", payload["created_at"]))
        except (KeyError, ValueError, TypeError) as error:
            raise SnapshotError("invalid created_at timestamp") from error
        slices: dict[type[object], tuple[object, ...]] = {}
        for entry in cast("list[dict[str, Any]]", payload.get("slices", [])):
            slice_type = registry.resolve(cast("str", entry["type"]))
            entry_items = cast("list[Any]", entry["items"])
            items = tuple(_decode(item, registry) for item in entry_items)
            slices[slice_type] = items
        return cls(
            slices=slices,
            created_at=created_at,
            schema_version=_SCHEMA_VERSION,
        )


# =============================================================================
# Capture / restore helpers
# =============================================================================


def capture(session: Session) -> Snapshot:
    """Capture an atomic :class:`Snapshot` of ``session``."""
    return Snapshot(
        slices=session._slice_state(),
        created_at=datetime.now(UTC),
    )


def restore(session: Session, snapshot: Snapshot) -> None:
    """Restore ``session`` to the state recorded in ``snapshot``."""
    session._restore_slice_state(dict(snapshot.slices))


# =============================================================================
# Encode / decode
# =============================================================================


def _encode(value: object, registry: TypeRegistry) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple | list):
        sequence = cast("tuple[object, ...] | list[object]", value)
        return [_encode(item, registry) for item in sequence]
    if is_dataclass(value) and not isinstance(value, type):
        encoded: dict[str, object] = {"__type__": registry.identifier_for(type(value))}
        for f in fields(value):
            encoded[f.name] = _encode(getattr(value, f.name), registry)
        return encoded
    raise SnapshotError(f"unsupported value of type {type(value).__name__} in snapshot")


def _decode(payload: Any, registry: TypeRegistry) -> object:  # noqa: ANN401
    if payload is None or isinstance(payload, str | int | float | bool):
        return payload
    if isinstance(payload, list):
        items = cast("list[Any]", payload)
        return tuple(_decode(item, registry) for item in items)
    if isinstance(payload, dict):
        mapping = cast("dict[str, Any]", payload)
        type_id = mapping.get("__type__")
        if not isinstance(type_id, str):
            raise SnapshotError("dataclass payload missing __type__ tag")
        cls = cast("type[Any]", registry.resolve(type_id))
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            kwargs[f.name] = _decode(mapping.get(f.name), registry)
        return cls(**kwargs)
    raise SnapshotError(  # pragma: no cover - JSON cannot produce other types
        f"unsupported payload: {payload!r}"
    )
