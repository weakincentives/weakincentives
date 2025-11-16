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

"""Snapshot serialization utilities for :mod:`weakincentives.runtime.session`."""

from __future__ import annotations

import json
import types
from collections.abc import Mapping
from dataclasses import dataclass, field, is_dataclass
from datetime import UTC, datetime
from importlib import import_module
from typing import Any, TypeGuard, cast, override

from ...prompt._types import SupportsDataclass
from ...serde import dump, parse
from ...types import JSONValue
from ._slice_types import SessionSlice, SessionSliceType
from .dataclasses import is_dataclass_instance

SNAPSHOT_SCHEMA_VERSION = "1"


type SnapshotState = Mapping[SessionSliceType, SessionSlice]


class SnapshotSerializationError(RuntimeError):
    """Raised when snapshot capture fails due to unsupported payloads."""


class SnapshotRestoreError(RuntimeError):
    """Raised when snapshot restoration fails due to incompatible payloads."""


def normalize_snapshot_state(
    state: Mapping[SessionSliceType, SessionSlice],
) -> SnapshotState:
    """Validate snapshot state and return an immutable copy."""

    normalized: dict[SessionSliceType, SessionSlice] = {}
    for slice_key, values in state.items():
        if not _is_dataclass_type(slice_key):
            raise ValueError("Slice keys must be dataclass types")

        slice_type = slice_key

        items: list[SupportsDataclass] = []
        for value in values:
            if not is_dataclass_instance(value):
                raise ValueError(
                    f"Slice {slice_type.__qualname__} contains non-dataclass value"
                )
            try:
                _ = dump(value)
            except Exception as error:
                raise ValueError(
                    f"Slice {slice_type.__qualname__} cannot be serialized"
                ) from error
            items.append(value)

        normalized[slice_type] = tuple(items)

    return cast(SnapshotState, types.MappingProxyType(normalized))


def _type_identifier(cls: SessionSliceType) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _resolve_type(identifier: str) -> SessionSliceType:
    module_name, _, qualname = identifier.partition(":")
    if not module_name or not qualname:
        msg = f"Invalid type identifier: {identifier!r}"
        raise SnapshotRestoreError(msg)
    module = import_module(module_name)
    target: Any = module
    for part in qualname.split("."):
        target = getattr(target, part, None)
        if target is None:
            msg = f"Type {identifier!r} could not be resolved"
            raise SnapshotRestoreError(msg)
    if not isinstance(target, type):
        msg = f"Resolved object for {identifier!r} is not a type"
        raise SnapshotRestoreError(msg)
    return cast(SessionSliceType, target)


def _ensure_timezone(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _infer_item_type(
    slice_type: SessionSliceType, values: SessionSlice
) -> SessionSliceType:
    if values:
        first_value = values[0]
        first_type = type(first_value)
        for value in values:
            if type(value) is not first_type:  # intentional identity check
                msg = (
                    "Snapshot slices must contain a single dataclass type; "
                    f"found {type(value)!r}"
                )
                raise SnapshotSerializationError(msg)
        return first_type
    return slice_type


def _is_dataclass_type(value: object) -> TypeGuard[type[SupportsDataclass]]:
    return isinstance(value, type) and is_dataclass(value)


@dataclass(slots=True, frozen=True)
class SnapshotSlicePayload:
    """Typed representation of a serialized snapshot slice entry."""

    slice_type: str
    item_type: str
    items: tuple[Mapping[str, JSONValue], ...]

    @classmethod
    def from_object(cls, obj: object) -> SnapshotSlicePayload:
        if not isinstance(obj, Mapping):
            raise SnapshotRestoreError("Slice entry must be an object")

        entry = cast(Mapping[str, JSONValue], obj)
        slice_identifier = entry.get("slice_type")
        item_identifier = entry.get("item_type")

        if not isinstance(slice_identifier, str) or not isinstance(
            item_identifier, str
        ):
            raise SnapshotRestoreError("Slice type identifiers must be strings")

        items_obj_raw = entry.get("items", [])
        if not isinstance(items_obj_raw, list):
            raise SnapshotRestoreError("Slice items must be a list")

        items_obj = items_obj_raw
        items: list[Mapping[str, JSONValue]] = []
        for item in items_obj:
            if not isinstance(item, Mapping):
                raise SnapshotRestoreError("Slice items must be objects")
            items.append(cast(Mapping[str, JSONValue], item))

        return cls(
            slice_type=slice_identifier,
            item_type=item_identifier,
            items=tuple(items),
        )


@dataclass(slots=True, frozen=True)
class SnapshotPayload:
    """Typed representation of the serialized snapshot envelope."""

    version: str
    created_at: str
    slices: tuple[SnapshotSlicePayload, ...]

    @classmethod
    def from_json(cls, raw: str) -> SnapshotPayload:
        try:
            payload_obj: JSONValue = json.loads(raw)
        except json.JSONDecodeError as error:
            raise SnapshotRestoreError("Invalid snapshot JSON") from error

        if not isinstance(payload_obj, Mapping):
            raise SnapshotRestoreError("Snapshot payload must be an object")

        payload = cast(Mapping[str, JSONValue], payload_obj)
        version = payload.get("version")
        if not isinstance(version, str):
            raise SnapshotRestoreError("Snapshot version must be a string")

        created_at = payload.get("created_at")
        if not isinstance(created_at, str):
            raise SnapshotRestoreError("Snapshot created_at must be a string")

        slices_obj = payload.get("slices", [])
        if not isinstance(slices_obj, list):
            raise SnapshotRestoreError("Snapshot slices must be a list")

        slices_source = slices_obj
        slices = tuple(
            SnapshotSlicePayload.from_object(entry) for entry in slices_source
        )
        return cls(version=version, created_at=created_at, slices=slices)


@dataclass(slots=True, frozen=True)
class Snapshot:
    """Frozen value object representing session slice state."""

    created_at: datetime
    slices: SnapshotState = field(
        default_factory=lambda: cast(
            SnapshotState,
            types.MappingProxyType({}),
        )
    )

    def __post_init__(self) -> None:
        normalized: dict[SessionSliceType, SessionSlice] = {
            slice_type: tuple(values) for slice_type, values in self.slices.items()
        }
        object.__setattr__(self, "created_at", _ensure_timezone(self.created_at))
        object.__setattr__(
            self,
            "slices",
            cast(SnapshotState, types.MappingProxyType(normalized)),
        )

    @override
    def __hash__(self) -> int:
        ordered = tuple(
            sorted(
                self.slices.items(),
                key=lambda item: _type_identifier(item[0]),
            )
        )
        return hash((self.created_at, ordered))

    def to_json(self) -> str:
        """Serialize the snapshot to a JSON string."""

        payload_slices: list[dict[str, JSONValue]] = []
        for slice_type, values in sorted(
            self.slices.items(), key=lambda item: _type_identifier(item[0])
        ):
            item_type = _infer_item_type(slice_type, values)
            try:
                serialized_items = [
                    cast(Mapping[str, JSONValue], dump(value)) for value in values
                ]
            except Exception as error:
                msg = f"Failed to serialize slice {slice_type.__qualname__}"
                raise SnapshotSerializationError(msg) from error

            payload_slices.append(
                {
                    "slice_type": _type_identifier(slice_type),
                    "item_type": _type_identifier(item_type),
                    "items": serialized_items,
                }
            )

        payload: dict[str, JSONValue] = {
            "version": SNAPSHOT_SCHEMA_VERSION,
            "created_at": self.created_at.isoformat(),
            "slices": payload_slices,
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> Snapshot:
        """Deserialize a snapshot from its JSON representation."""

        payload = SnapshotPayload.from_json(raw)

        if payload.version != SNAPSHOT_SCHEMA_VERSION:
            msg = (
                "Snapshot schema version mismatch: "
                f"expected {SNAPSHOT_SCHEMA_VERSION}, got {payload.version!r}"
            )
            raise SnapshotRestoreError(msg)

        try:
            created_at = datetime.fromisoformat(payload.created_at)
        except ValueError as error:
            raise SnapshotRestoreError("Invalid created_at timestamp") from error

        restored: dict[SessionSliceType, SessionSlice] = {}
        for entry in payload.slices:
            slice_type_candidate = _resolve_type(entry.slice_type)
            item_type_candidate = _resolve_type(entry.item_type)

            if not _is_dataclass_type(slice_type_candidate) or not _is_dataclass_type(
                item_type_candidate
            ):
                raise SnapshotRestoreError("Snapshot types must be dataclasses")

            slice_type = slice_type_candidate
            item_type = item_type_candidate

            restored_items: list[SupportsDataclass] = []
            for item_mapping in entry.items:
                try:
                    restored_item = parse(item_type, item_mapping)
                except Exception as error:
                    raise SnapshotRestoreError(
                        f"Failed to restore slice {slice_type.__qualname__}"
                    ) from error
                restored_items.append(restored_item)

            restored[slice_type] = tuple(restored_items)

        return cls(created_at=_ensure_timezone(created_at), slices=restored)
