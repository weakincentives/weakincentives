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

"""Snapshot serialization utilities for :mod:`weakincentives.session`."""

from __future__ import annotations

import json
import types
from collections.abc import Mapping
from dataclasses import dataclass, field, is_dataclass
from datetime import UTC, datetime
from importlib import import_module
from typing import Any, TypeGuard, cast, override

from ..prompt._types import SupportsDataclass
from ..serde import dump, parse

SNAPSHOT_SCHEMA_VERSION = "1"


class SnapshotSerializationError(RuntimeError):
    """Raised when snapshot capture fails due to unsupported payloads."""


class SnapshotRestoreError(RuntimeError):
    """Raised when snapshot restoration fails due to incompatible payloads."""


def normalize_snapshot_state(
    state: Mapping[object, tuple[object, ...]],
) -> Mapping[type[SupportsDataclass], tuple[SupportsDataclass, ...]]:
    """Validate snapshot state and return an immutable copy."""

    normalized: dict[type[SupportsDataclass], tuple[SupportsDataclass, ...]] = {}
    for slice_key, values in state.items():
        if not _is_dataclass_type(slice_key):
            raise ValueError("Slice keys must be dataclass types")

        slice_type = cast(type[SupportsDataclass], slice_key)  # pyright: ignore[reportUnnecessaryCast]

        items: list[SupportsDataclass] = []
        for value in values:
            if not _is_dataclass_instance(value):
                raise ValueError(
                    f"Slice {slice_type.__qualname__} contains non-dataclass value"
                )
            dataclass_value = cast(SupportsDataclass, value)  # pyright: ignore[reportUnnecessaryCast]
            try:
                _ = dump(dataclass_value)
            except Exception as error:
                raise ValueError(
                    f"Slice {slice_type.__qualname__} cannot be serialized"
                ) from error
            items.append(dataclass_value)

        normalized[slice_type] = tuple(items)

    return types.MappingProxyType(normalized)


def _type_identifier(cls: type[Any]) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _resolve_type(identifier: str) -> type[Any]:
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
    return target


def _ensure_timezone(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _infer_item_type(
    slice_type: type[SupportsDataclass], values: tuple[SupportsDataclass, ...]
) -> type[SupportsDataclass]:
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


def _is_dataclass_instance(value: object) -> TypeGuard[SupportsDataclass]:
    return is_dataclass(value) and not isinstance(value, type)


@dataclass(slots=True, frozen=True)
class Snapshot:
    """Frozen value object representing session slice state."""

    created_at: datetime
    slices: Mapping[type[SupportsDataclass], tuple[SupportsDataclass, ...]] = field(
        default_factory=lambda: cast(
            Mapping[type[SupportsDataclass], tuple[SupportsDataclass, ...]],
            types.MappingProxyType({}),
        )
    )

    def __post_init__(self) -> None:
        normalized: dict[type[SupportsDataclass], tuple[SupportsDataclass, ...]] = {
            slice_type: tuple(values) for slice_type, values in self.slices.items()
        }
        object.__setattr__(self, "created_at", _ensure_timezone(self.created_at))
        object.__setattr__(self, "slices", types.MappingProxyType(normalized))

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

        payload_slices: list[dict[str, object]] = []
        for slice_type, values in sorted(
            self.slices.items(), key=lambda item: _type_identifier(item[0])
        ):
            item_type = _infer_item_type(slice_type, values)
            try:
                serialized_items = [dump(value) for value in values]
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

        payload: dict[str, object] = {
            "version": SNAPSHOT_SCHEMA_VERSION,
            "created_at": self.created_at.isoformat(),
            "slices": payload_slices,
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> Snapshot:
        """Deserialize a snapshot from its JSON representation."""

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise SnapshotRestoreError("Invalid snapshot JSON") from error

        if not isinstance(payload, dict):
            raise SnapshotRestoreError("Snapshot payload must be an object")

        payload_mapping = cast(dict[str, object], payload)

        version = payload_mapping.get("version")
        if version != SNAPSHOT_SCHEMA_VERSION:
            msg = (
                "Snapshot schema version mismatch: "
                f"expected {SNAPSHOT_SCHEMA_VERSION}, got {version!r}"
            )
            raise SnapshotRestoreError(msg)

        created_at_raw = payload_mapping.get("created_at")
        if not isinstance(created_at_raw, str):
            raise SnapshotRestoreError("Snapshot created_at must be a string")
        try:
            created_at = datetime.fromisoformat(created_at_raw)
        except ValueError as error:
            raise SnapshotRestoreError("Invalid created_at timestamp") from error

        slices_payload_raw = payload_mapping.get("slices", [])
        if not isinstance(slices_payload_raw, list):
            raise SnapshotRestoreError("Snapshot slices must be a list")

        slices_payload = cast(list[object], slices_payload_raw)

        restored: dict[type[SupportsDataclass], tuple[SupportsDataclass, ...]] = {}
        for entry_obj in slices_payload:
            if not isinstance(entry_obj, Mapping):
                raise SnapshotRestoreError("Slice entry must be an object")

            entry = cast(Mapping[str, object], entry_obj)

            slice_identifier = entry.get("slice_type")
            item_identifier = entry.get("item_type")

            if not isinstance(slice_identifier, str) or not isinstance(
                item_identifier, str
            ):
                raise SnapshotRestoreError("Slice type identifiers must be strings")

            slice_type_candidate = _resolve_type(slice_identifier)
            item_type_candidate = _resolve_type(item_identifier)

            if not _is_dataclass_type(slice_type_candidate) or not _is_dataclass_type(
                item_type_candidate
            ):
                raise SnapshotRestoreError("Snapshot types must be dataclasses")

            slice_type = slice_type_candidate
            item_type = item_type_candidate

            items_payload_obj = entry.get("items", [])

            if not isinstance(items_payload_obj, list):
                raise SnapshotRestoreError("Slice items must be a list")

            items_payload = cast(list[object], items_payload_obj)

            restored_items: list[SupportsDataclass] = []
            for item_obj in items_payload:
                if not isinstance(item_obj, Mapping):
                    raise SnapshotRestoreError("Slice items must be objects")
                item_mapping = cast(Mapping[str, object], item_obj)
                try:
                    restored_item = parse(item_type, item_mapping)
                except Exception as error:
                    raise SnapshotRestoreError(
                        f"Failed to restore slice {slice_type.__qualname__}"
                    ) from error
                restored_items.append(restored_item)

            restored[slice_type] = tuple(restored_items)

        return cls(created_at=_ensure_timezone(created_at), slices=restored)
